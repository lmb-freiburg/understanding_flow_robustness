import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from dataset_utils import custom_transforms, datasets, kitti_datasets

SUM_FREQ = 100
MAX_FLOW = 400  # exclude extremly large displacements


class Logger:
    def __init__(self, model, scheduler, ckpt_dir, total_steps=0):
        self.model = model
        self.scheduler = scheduler
        self.ckpt_dir = ckpt_dir
        self.total_steps = total_steps
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [
            self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())
        ]
        training_str = "[{:6d}, {:10.7f}] ".format(
            self.total_steps + 1, self.scheduler.get_last_lr()[0]
        )
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter(self.ckpt_dir / "log")

        for k, v in self.running_loss.items():
            self.writer.add_scalar(k, v / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(self.ckpt_dir / "log")

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def multiscale_epe(
    flow_preds,
    flow_gt,
    valid,
    gamma=0.8,
    max_flow=MAX_FLOW,
    flowNetC=False,
    not_excluding=False,
    div_flow=1,
    flownetc_weighing=False,
    pwc=False,
):
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    eps = 1e-5

    if div_flow > 1:
        flow_gt = flow_gt / div_flow

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    if not_excluding:
        valid = valid >= 0.5

    if flowNetC or pwc:
        _, _, h, w = flow_preds[0].size()
        flow_gt_scaled = torch.nn.functional.interpolate(flow_gt, (h, w), mode="area")
        epe = torch.sum((flow_preds[0] - flow_gt_scaled) ** 2, dim=1).sqrt().view(-1)
        epe = epe[torch.logical_not(torch.isnan(epe))]
    else:
        epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

    for i in range(n_predictions):
        i_weight = (
            gamma**i if flowNetC or pwc else gamma ** (n_predictions - i - 1)
        )  # RAFT assigns larger weight to later refinements
        if flowNetC or pwc:
            weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original articles
            _, _, h, w = flow_preds[i].size()
            scale_x = w / flow_gt.shape[3]
            scale_y = h / flow_gt.shape[2]
            flow_gt_scaled = torch.nn.functional.interpolate(flow_gt, (h, w), mode="area")
            flow_gt_scaled[:, 0, ...] = flow_gt_scaled[:, 0, ...] * scale_x
            flow_gt_scaled[:, 1, ...] = flow_gt_scaled[:, 1, ...] * scale_y
            EPE_map = (
                torch.sum((flow_preds[i] - flow_gt_scaled) ** 2, dim=1) + eps
            ).sqrt()
            if not EPE_map[torch.logical_not(torch.isnan(EPE_map))].numel():
                continue
            if flownetc_weighing:
                flow_loss += weights[i] * torch.mean(
                    EPE_map[torch.logical_not(torch.isnan(EPE_map))]
                )
            else:
                flow_loss += i_weight * torch.mean(
                    EPE_map[torch.logical_not(torch.isnan(EPE_map))]
                )

            # valid_scaled = (flow_gt_scaled[0].abs() < 1000) & (flow_gt_scaled[1].abs() < 1000)
            # valid_scaled = valid_scaled.float()
            # mag_scaled = torch.sum(flow_gt_scaled**2, dim=1).sqrt()
            # valid_scaled = (valid_scaled >= 0.5) & (mag_scaled < max_flow)
            # flow_loss += weights[i] * (valid_scaled[:, None] * EPE_map.sum()/batch_size).mean()
        else:
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
        "loss": flow_loss.float().mean().item(),
    }

    return flow_loss, metrics


def sequence_loss(
    flow_preds,
    flow_gt,
    valid,
    gamma=0.8,
    max_flow=MAX_FLOW,
    flowNetC=False,
    pwc=False,
    not_excluding=False,
    div_flow=1,
    flownetc_weighing=False,
):
    """Loss function defined over sequence of flow predictions"""

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    if div_flow > 1:
        flow_gt = flow_gt / div_flow

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    if not_excluding:
        valid = valid >= 0.5

    if flowNetC or pwc:
        _, _, h, w = flow_preds[0].size()
        scale_x = w / flow_gt.shape[3]
        scale_y = h / flow_gt.shape[2]
        flow_gt_scaled = torch.nn.functional.interpolate(flow_gt, (h, w), mode="area")
        flow_gt_scaled[:, 0, ...] = flow_gt_scaled[:, 0, ...] * scale_x
        flow_gt_scaled[:, 1, ...] = flow_gt_scaled[:, 1, ...] * scale_y
        epe = torch.sum((flow_preds[0] - flow_gt_scaled) ** 2, dim=1).sqrt().view(-1)
    else:
        epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

    for i in range(n_predictions):
        i_weight = (
            gamma**i if flowNetC else gamma ** (n_predictions - i - 1)
        )  # RAFT assigns larger weight to later refinements
        if flowNetC or pwc:
            weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original articles
            _, _, h, w = flow_preds[i].size()
            scale_x = w / flow_gt.shape[3]
            scale_y = h / flow_gt.shape[2]
            flow_gt_scaled = torch.nn.functional.interpolate(flow_gt, (h, w), mode="area")
            flow_gt_scaled[:, 0, ...] = flow_gt_scaled[:, 0, ...] * scale_x
            flow_gt_scaled[:, 1, ...] = flow_gt_scaled[:, 1, ...] * scale_y
            i_loss = (flow_preds[i] - flow_gt_scaled).abs()
            # flow_loss += i_weight * i_loss.mean()
            # flow_loss += i_weight * torch.mean(
            #     i_loss[torch.logical_not(torch.isnan(i_loss))]
            # )
            if flownetc_weighing:
                flow_loss += weights[i] * torch.mean(
                    i_loss[torch.logical_not(torch.isnan(i_loss))]
                )
            else:
                flow_loss += i_weight * torch.mean(
                    i_loss[torch.logical_not(torch.isnan(i_loss))]
                )
        else:
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_optimizer(args, model, inner_iteration: int = 1):
    """Create the optimizer and learning rate scheduler"""
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps * inner_iteration + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    return optimizer, scheduler


def fetch_dataloader(args):
    """Create the data loader for the corresponding training set"""

    if args.adv_train:
        assert args.batch_size == 1, "Only works for batch size 1!"

        transforms = custom_transforms.Compose(
            [
                custom_transforms.Scale(h=args.image_size[0], w=args.image_size[1]),
                custom_transforms.ArrayToTensorWoNorm(),
                custom_transforms.Normalize(
                    mean=[0, 0, 0], std=[255.0, 255.0, 255.0]
                ),  # compatibility with perturb model
            ]
        )

        if args.stage == "kitti2015":
            dataset = kitti_datasets.KITTI2015(
                n_height=args.image_size[0],
                n_width=args.image_size[1],
                transform=transforms,
                finetune=args.finetune,
            )
        elif args.stage == "kitti2012":
            dataset = kitti_datasets.KITTI2012(
                n_height=args.image_size[0],
                n_width=args.image_size[1],
                transform=transforms,
                finetune=args.finetune,
            )
        else:
            raise NotImplementedError

        if args.online_subset is not None:
            dataset = torch.utils.data.Subset(dataset, args.online_subset)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            pin_memory=False,
            shuffle=True,
            num_workers=1
            if args.DEBUG or args.online_subset is not None
            else 2 * len(args.gpus),
            drop_last=True,
        )

        print("Training with %d image pairs" % len(dataset))
    else:
        TRAIN_DS = "C+T+K+S+H"
        if args.stage == "chairs":
            if args.trans_rot:
                aug_params = {
                    "crop_size": args.image_size,
                    "min_scale": -0.2,
                    "max_scale": 1.0,
                    "do_flip": True,
                    "do_trans_rot": True,
                    "translate": 10,
                    "rot_angle": 17,
                    "diff_angle": 0,
                }
            else:
                aug_params = {
                    "crop_size": args.image_size,
                    "min_scale": -0.2,
                    "max_scale": 1.0,
                    "do_flip": True,
                }
            train_dataset = datasets.FlyingChairs(aug_params, split="training")

        elif args.stage == "things":
            if args.trans_rot:
                aug_params = {
                    "crop_size": args.image_size,
                    "min_scale": -0.4,
                    "max_scale": 0.8,
                    "do_flip": True,
                    "do_trans_rot": True,
                    "translate": 10,
                    "rot_angle": 17,
                    "diff_angle": 5,
                }
            else:
                aug_params = {
                    "crop_size": args.image_size,
                    "min_scale": -0.4,
                    "max_scale": 0.8,
                    "do_flip": True,
                }
            # aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
            clean_dataset = datasets.FlyingThings3D(aug_params, dstype="frames_cleanpass")
            final_dataset = datasets.FlyingThings3D(aug_params, dstype="frames_finalpass")
            train_dataset = clean_dataset + final_dataset

        elif args.stage == "sintel":
            aug_params = {
                "crop_size": args.image_size,
                "min_scale": -0.2,
                "max_scale": 0.6,
                "do_flip": True,
            }
            things = datasets.FlyingThings3D(aug_params, dstype="frames_cleanpass")
            sintel_clean = datasets.MpiSintel(
                aug_params, split="training", dstype="clean"
            )
            sintel_final = datasets.MpiSintel(
                aug_params, split="training", dstype="final"
            )

            if TRAIN_DS == "C+T+K+S+H":
                kitti = datasets.KITTI(
                    {
                        "crop_size": args.image_size,
                        "min_scale": -0.3,
                        "max_scale": 0.5,
                        "do_flip": True,
                    }
                )
                hd1k = datasets.HD1K(
                    {
                        "crop_size": args.image_size,
                        "min_scale": -0.5,
                        "max_scale": 0.2,
                        "do_flip": True,
                    }
                )
                train_dataset = (
                    100 * sintel_clean
                    + 100 * sintel_final
                    + 200 * kitti
                    + 5 * hd1k
                    + things
                )

            elif TRAIN_DS == "C+T+K/S":
                train_dataset = 100 * sintel_clean + 100 * sintel_final + things

        elif args.stage == "kitti":
            aug_params = {
                "crop_size": args.image_size,
                "min_scale": -0.2,
                "max_scale": 0.4,
                "do_flip": False,
            }
            train_dataset = datasets.KITTI(aug_params, split="training")

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            pin_memory=False,
            shuffle=True,
            num_workers=1 if args.DEBUG else 2 * len(args.gpus),
            drop_last=True,
        )

        print("Training with %d image pairs" % len(train_dataset))
    return train_loader
