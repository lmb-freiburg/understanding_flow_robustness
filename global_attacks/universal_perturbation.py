import argparse
import json
import shutil
import warnings

import numpy as np
import torch
from path import Path
from scipy.ndimage.interpolation import zoom
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm

import global_attacks.global_constants as settings
from dataset_utils import custom_transforms

# from dataset_utils.kitti_datasets import KITTI2012
from dataset_utils.sequence_folders import SequenceFolder
from dataset_utils.utils import get_evaluation_set
from flowutils.flowlib import flow_to_image
from global_attacks.perturb_model import compute_cossim, compute_epe, compute_flow_loss
from models.utils_model import fetch_model, predict_flow
from patch_attacks.logger import AverageMeter, TermLogger
from patch_attacks.utils import tensor2array, transpose_image

warnings.filterwarnings("ignore")

best_error = -1
n_iter = 0

parser = argparse.ArgumentParser()
# Training and validation input filepaths
parser.add_argument(
    "--valset",
    type=str,
    help="Flow for validation set",
    default="kitti2015",
    choices=["kitti2015", "kitti2012"],
)
# Run settings
parser.add_argument(
    "--n_height", type=int, default=settings.N_HEIGHT, help="Height of each sample"
)
parser.add_argument(
    "--n_width", type=int, default=settings.N_WIDTH, help="Width of each sample"
)
parser.add_argument(
    "--batch_size", type=int, default=4, metavar="N", help="mini-batch size"
)
parser.add_argument(
    "--epochs", default=40, type=int, metavar="N", help="Number of total epochs to run"
)
parser.add_argument(
    "--epoch_size",
    default=100,
    type=int,
    metavar="N",
    help="manual epoch size (will match dataset size if not set)",
)
# Perturb model settings
parser.add_argument(
    "--perturb_method",
    type=str,
    default=settings.PERTURB_METHOD,
    help="Perturb method available: %s" % settings.PERTURB_METHOD_AVAILABLE,
)
parser.add_argument(
    "--perturb_mode",
    type=str,
    default="both",
    help="Perturb modes available: %s" % settings.PERTURB_MODE_AVAILABLE,
)
parser.add_argument(
    "--output_norm", default=settings.OUTPUT_NORM, help="Output norm of noise"
)
parser.add_argument(
    "--n_step",
    type=int,
    default=10,
    help="Number of steps to optimize perturbations",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=2e-3,
    help="Learning rate (alpha) to use for optimizing perturbations",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.47,
    help="Momentum (beta) used for momentum iterative fast gradient sign method",
)
parser.add_argument(
    "--probability_diverse_input",
    type=float,
    default=0.00,
    help="Probability (p) to use diverse input",
)
parser.add_argument(
    "--flow_loss",
    type=str,
    default="cossim",
    choices=["cossim", "l2", "l1", "corr"],
    help="What type of flow loss to use",
)
# Stereo model settings
parser.add_argument(
    "--disparity_method",
    type=str,
    default=settings.STEREO_METHOD,
    help="Stereo method available: %s" % settings.STEREO_METHOD_AVAILABLE,
)
parser.add_argument("--flownet", type=str, default="FlowNetC", help="Flownet model")
parser.add_argument(
    "--disparity_model_restore_path",
    type=str,
    default="",
    help="Path to restore model checkpoint",
)
# Output settings
parser.add_argument(
    "--output_path", type=str, default=settings.OUTPUT_PATH, help="Path to save outputs"
)
# Hardware settings
parser.add_argument(
    "--device", type=str, default=settings.DEVICE, help="Device to use: gpu, cpu"
)
parser.add_argument("--disparity", action="store_true", help="Run disparity estimation")
parser.add_argument("--DEBUG", action="store_true", help="Debug mode")
parser.add_argument(
    "--log-output",
    type=bool,
    default=True,
    help="will log dispnet outputs and warped imgs at validation step",
)
parser.add_argument(
    "--log_terminal", action="store_true", help="will display progressbar at terminal"
)
parser.add_argument(
    "--training-output-freq",
    type=int,
    help="frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output",
    metavar="N",
    default=50,
)
parser.add_argument(
    "--seed",
    type=int,
    default=-1,
    help="Set seed. Note that this might overwrite previous results with this seed! If not specified unused seed is used",
)
parser.add_argument(
    "--workers", default=4, type=int, metavar="N", help="Number of data loading workers"
)
parser.add_argument("--add_gaussian", action="store_true", help="Adds Gaussian noise")

args = parser.parse_args()


def run(
    # Perturb method settings
    perturb_method=settings.PERTURB_METHOD,
    perturb_mode=settings.PERTURB_MODE,
    output_norm=settings.OUTPUT_NORM,
    # Stereo model settings
    # stereo_method=settings.STEREO_METHOD,
    # stereo_model_restore_path=settings.STEREO_MODEL_RESTORE_PATH,
    # Output settings
    output_path="",
    # Hardware settings
    device=settings.DEVICE,
    args=None,
):
    if device == settings.CUDA or device == settings.GPU:
        device = torch.device(settings.CUDA)
    else:
        device = torch.device(settings.CPU)

    if args.DEBUG:
        output_path = Path(output_path) / "DEBUG"

    # Set random seed. Might overwrite previous results if set!
    if args.seed > 0:
        settings.RANDOM_SEED = args.seed
    else:
        args.seed = np.random.randint(0, 1e4)
    torch.manual_seed(settings.RANDOM_SEED)
    np.random.seed(settings.RANDOM_SEED)

    if device.type == settings.CUDA:
        torch.cuda.manual_seed(settings.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    folder_name = f"{args.learning_rate}_{args.n_step}_{args.seed}"
    if args.add_gaussian:
        folder_name += "_addG"
    output_path = (
        Path(output_path)
        / "kitti2015"
        / args.flownet
        / "universal"
        / perturb_mode
        / f"{perturb_method}_{args.flow_loss}"
        / str(output_norm)
        / folder_name
    )
    print(f"Save everything to {output_path}")
    output_path.makedirs_p()

    with open(output_path / "args.json", "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, indent=2)

    pertubation_path = output_path / "perturbations"
    try:
        shutil.rmtree(pertubation_path)
    except Exception:
        pass
    pertubation_path.makedirs_p()

    # Set up output and logging paths
    training_writer = SummaryWriter(output_path / "train")
    validation_writer = SummaryWriter(output_path / "valid_attack")

    if "di2" in args.perturb_method:
        perturb_method = perturb_method[3:]

    # Read input paths
    train_transform = custom_transforms.Compose(
        [
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(h=args.n_height, w=args.n_width),
            custom_transforms.ArrayToTensor(),
        ]
    )

    print("=> fetching scenes in datasets/KITTI/2012_prepared")
    train_set = SequenceFolder(
        root="datasets/KITTI/2012_prepared",
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=3,
    )
    # transforms = custom_transforms.Compose(
    #     [
    #         custom_transforms.Scale(h=args.n_height, w=args.n_width),
    #         custom_transforms.ArrayToTensorWoNorm(),
    #         custom_transforms.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
    #     ]
    # )
    # train_set = KITTI2012(
    #     n_height=args.n_height,
    #     n_width=args.n_width,
    #     transform=transforms,
    #     disparity=args.disparity,
    # )

    val_set = get_evaluation_set(args.n_height, args.n_width, args)

    if args.DEBUG:
        train_set_len = 8  # 32
        train_set.__len__ = train_set_len
        train_set.samples = train_set.samples[:train_set_len]

    print(f"{len(train_set)} samples in train scenes")
    print(f"{len(val_set)} samples found in valid scenes")
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    # batch size is 1 since images in kitti have different sizes
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")
    if args.disparity:
        raise NotImplementedError
        # stereo_model = StereoModel(method=stereo_method, device=device)
        # # Restore stereo model
        # stereo_model.restore_model(stereo_model_restore_path)
    else:
        flow_net = fetch_model(
            args=args,
        )
        flow_net = flow_net.cuda()

    if args.log_terminal:
        logger = TermLogger(
            n_epochs=args.epochs,
            train_size=min(len(train_loader), args.epoch_size),
            valid_size=len(val_loader),
            attack_size=args.n_step,
        )
        logger.epoch_bar.start()
    else:
        logger = None
    universal_perturbation = torch.zeros((1, 2, 3, args.n_height, args.n_width))
    for epoch in tqdm(range(args.epochs)):
        if args.log_terminal:
            logger.epoch_bar.update(epoch)
            logger.reset_train_bar()

        universal_perturbation = train(
            universal_perturbation,
            train_loader,
            flow_net,
            args,
            training_writer,
            logger,
        )
        errors, error_names = validation(
            universal_perturbation,
            val_loader,
            flow_net,
            epoch,
            args,
            validation_writer,
            logger,
        )
        error_string = ", ".join(
            f"{name} : {error:.3f}" for name, error in zip(error_names, errors)
        )
        if args.log_terminal:
            logger.valid_writer.write(f" * Avg {error_string}")
        else:
            print(f"Epoch {epoch} completed")

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        torch.save(universal_perturbation, pertubation_path / f"epoch_{epoch}")

    if args.log_terminal:
        logger.epoch_bar.finish()


def train(universal_perturbation, train_loader, model, args, writer=None, logger=None):
    global n_iter  # pylint: disable=global-statement

    universal_perturbation_var = Variable(universal_perturbation)
    model.eval()
    for i, data in enumerate(train_loader):
        if len(data) == 2:
            tgt_img, ref_img = data
        else:
            if len(data) == 4:
                tgt_img, ref_img, _, _ = data
                ref_img = [ref_img, ref_img]
            else:
                raise NotImplementedError
        tgt_img_var = Variable(tgt_img.cuda())
        ref_img_past_var = Variable(ref_img[0].cuda())
        ref_img_future_var = Variable(ref_img[1].cuda())

        flow_pred_var = predict_flow(
            model, ref_img_past_var, tgt_img_var, ref_img_future_var, args
        )

        universal_perturbation_var = universal_perturbation_var.cuda()
        if not args.add_gaussian:
            target_var = Variable(
                -1 * flow_pred_var.data.clone(), requires_grad=True
            ).cuda()
        else:
            target_var = Variable(flow_pred_var.data.clone(), requires_grad=True).cuda()
            if args.add_gaussian:
                target_var = target_var + 1 * torch.randn(target_var.shape).cuda()
        (
            adv_tgt_img_var,
            adv_ref_img_past_var,
            adv_ref_img_future_var,
            universal_perturbation_var,
        ) = attack(
            model,
            tgt_img_var,
            ref_img_future_var,
            universal_perturbation_var,
            target_var=target_var,
            args=args,
        )

        if args.training_output_freq > 0 and n_iter % args.training_output_freq == 0:
            writer.add_image(
                "train tgt image", transpose_image(tensor2array(tgt_img[0])), n_iter
            )
            writer.add_image(
                "train ref past image",
                transpose_image(tensor2array(ref_img[0][0])),
                n_iter,
            )
            writer.add_image(
                "train ref future image",
                transpose_image(tensor2array(ref_img[1][0])),
                n_iter,
            )
            writer.add_image(
                "train adv tgt image",
                transpose_image(tensor2array(adv_tgt_img_var.data.cpu()[0])),
                n_iter,
            )
            if type(model).__name__ == "Back2Future":
                writer.add_image(
                    "train adv ref past image",
                    transpose_image(tensor2array(adv_ref_img_past_var.data.cpu()[0])),
                    n_iter,
                )
            writer.add_image(
                "train adv ref future image",
                transpose_image(tensor2array(adv_ref_img_future_var.data.cpu()[0])),
                n_iter,
            )
            writer.add_image(
                "universal perturbation 0",
                transpose_image(
                    tensor2array(universal_perturbation_var.data.cpu()[0, 0])
                ),
                n_iter,
            )
            writer.add_image(
                "universal perturbation 1",
                transpose_image(
                    tensor2array(universal_perturbation_var.data.cpu()[0, 1])
                ),
                n_iter,
            )

        if args.log_terminal:
            logger.train_bar.update(i + 1)
        if i >= args.epoch_size - 1:
            break
        n_iter += 1
    return universal_perturbation_var


def attack(
    model,
    img0_var,
    img1_var,
    universal_perturbation_var,
    target_var,
    args,
):
    model.eval()
    adv_img0_var, adv_img1_var = add_universal_perturbation(
        img0_var, img1_var, universal_perturbation_var
    )
    # args.n_step = 40
    for _ in range(args.n_step):
        adv_img0_var = torch.autograd.Variable(adv_img0_var, requires_grad=True)
        adv_img1_var = torch.autograd.Variable(adv_img1_var, requires_grad=True)
        loss = compute_flow_loss(
            model,
            adv_img0_var,
            adv_img1_var,
            target_var,
            args,
        )
        loss.backward(retain_graph=True)

        if "ifgsm" in args.perturb_method:
            img0_grad = torch.sign(adv_img0_var.grad.data)
            img1_grad = (
                torch.zeros_like(img1_var)
                if adv_img1_var.grad is None
                else torch.sign(adv_img1_var.grad.data)
            )
        elif "ifgm" == args.perturb_method:
            img0_grad = adv_img0_var.grad.data
            img1_grad = adv_img1_var.grad.data
        else:
            raise NotImplementedError

        if args.perturb_mode == "both":
            noise0_output = args.learning_rate * img0_grad
            noise1_output = args.learning_rate * img1_grad
        elif args.perturb_mode == "left":
            noise0_output = args.learning_rate * img0_grad
            noise1_output = torch.zeros_like(img1_var)
        elif args.perturb_mode == "right":
            noise0_output = torch.zeros_like(img0_var)
            noise1_output = args.learning_rate * img1_grad
        else:
            raise ValueError(f"Invalid perturbation mode: {args.perturb_mode}")

        # Add to image and clamp between supports of image intensity to get perturbed image
        if not args.add_gaussian:  # gradient descent
            adv_img0_var = torch.clamp(adv_img0_var - noise0_output, 0.0, 1.0)
            adv_img1_var = torch.clamp(adv_img1_var - noise1_output, 0.0, 1.0)
        else:  # gradient ascent
            adv_img0_var = torch.clamp(adv_img0_var + noise0_output, 0.0, 1.0)
            adv_img1_var = torch.clamp(adv_img1_var + noise1_output, 0.0, 1.0)

        # Output perturbations are the difference between original and perturbed image
        noise0_output = torch.clamp(
            adv_img0_var - img0_var, -args.output_norm, args.output_norm
        )
        noise1_output = torch.clamp(
            adv_img1_var - img1_var, -args.output_norm, args.output_norm
        )

        # Add perturbations to images
        adv_img0_var = img0_var + noise0_output
        adv_img1_var = img1_var + noise1_output

        # sys.stdout.write(
        #     "Step={:3}/{:3}  Model Loss={:.10f}\r".format(
        #         step, args.n_step, loss.item()
        #     )
        # )
        # sys.stdout.flush()

    universal_perturbation = torch.stack([noise0_output, noise1_output], dim=1)
    return adv_img0_var, None, adv_img1_var, Variable(universal_perturbation)


def validation(
    universal_perturbation, val_loader, model, epoch, args, writer=None, logger=None
):
    error_names = ["epe", "adv_epe", "cos_sim", "adv_cos_sim"]
    errors = AverageMeter(i=len(error_names))
    model.eval()

    with torch.no_grad():
        for i, (ref_img_past, tgt_img, ref_img_future, flow_gt, _, _, _) in enumerate(
            val_loader
        ):
            tgt_img_var = Variable(tgt_img.cuda())
            ref_img_past_var = Variable(ref_img_past.cuda())
            ref_img_future_var = Variable(ref_img_future.cuda())
            flow_gt_var = Variable(flow_gt.cuda())

            flow_fwd = predict_flow(
                model, ref_img_past_var, tgt_img_var, ref_img_future_var, args
            )

            universal_perturbation_var = Variable(universal_perturbation).cuda()
            adv_tgt_img_var, adv_ref_img_future_var = add_universal_perturbation(
                tgt_img_var, ref_img_future_var, universal_perturbation_var
            )
            adv_flow_fwd = predict_flow(
                model, None, adv_tgt_img_var, adv_ref_img_future_var, args
            )

            epe = compute_epe(gt=flow_gt_var, pred=flow_fwd)
            adv_epe = compute_epe(gt=flow_gt_var, pred=adv_flow_fwd)
            cos_sim = compute_cossim(flow_gt_var, flow_fwd)
            adv_cos_sim = compute_cossim(flow_gt_var, adv_flow_fwd)

            errors.update([epe, adv_epe, cos_sim, adv_cos_sim])

            if args.log_output and i % 100 == 0:
                index = int(i // 10)
                if epoch == 0:
                    writer.add_image(
                        "val flow Input", transpose_image(tensor2array(tgt_img[0])), 0
                    )
                    flow_to_show = flow_gt[0][:2, :, :].cpu()
                    writer.add_image(
                        "val target Flow",
                        transpose_image(flow_to_image(tensor2array(flow_to_show))),
                        epoch,
                    )

                val_Flow_Output = (
                    transpose_image(flow_to_image(tensor2array(flow_fwd.data[0].cpu())))
                    / 255.0
                )
                val_adv_Flow_Output = (
                    transpose_image(
                        flow_to_image(tensor2array(adv_flow_fwd.data[0].cpu()))
                    )
                    / 255.0
                )
                val_Diff_Flow_Output = (
                    transpose_image(
                        flow_to_image(
                            tensor2array((adv_flow_fwd - flow_fwd).data[0].cpu())
                        )
                    )
                    / 255.0
                )
                val_adv_tgt_image = transpose_image(
                    tensor2array(adv_tgt_img_var.data.cpu()[0])
                )
                # if type(model).__name__ == "Back2Future" and False:
                #     val_adv_ref_past_image = transpose_image(
                #         tensor2array(adv_ref_img_past_var.data.cpu()[0])
                #     )
                val_adv_ref_future_image = transpose_image(
                    tensor2array(adv_ref_img_future_var.data.cpu()[0])
                )

                if "FlowNetS" in args.flownet:
                    val_Flow_Output = zoom(
                        val_Flow_Output,
                        zoom=(
                            1,
                            val_adv_tgt_image.shape[1] / val_Flow_Output.shape[1],
                            val_adv_tgt_image.shape[2] / val_Flow_Output.shape[2],
                        ),
                        order=1,
                    )
                    val_adv_Flow_Output = zoom(
                        val_adv_Flow_Output,
                        zoom=(
                            1,
                            val_adv_tgt_image.shape[1] / val_adv_Flow_Output.shape[1],
                            val_adv_tgt_image.shape[2] / val_adv_Flow_Output.shape[2],
                        ),
                        order=1,
                    )
                    val_Diff_Flow_Output = zoom(
                        val_Diff_Flow_Output,
                        zoom=(
                            1,
                            val_adv_tgt_image.shape[1] / val_Diff_Flow_Output.shape[1],
                            val_adv_tgt_image.shape[2] / val_Diff_Flow_Output.shape[2],
                        ),
                        order=1,
                    )

                if type(model).__name__ == "Back2Future":
                    val_output_viz = np.hstack(
                        (
                            val_Flow_Output,
                            val_adv_Flow_Output,
                            val_Diff_Flow_Output,
                            val_adv_tgt_image,
                            val_adv_ref_future_image,
                        )
                    )
                else:
                    val_output_viz = np.hstack(
                        (
                            val_Flow_Output,
                            val_adv_Flow_Output,
                            val_Diff_Flow_Output,
                            val_adv_tgt_image,
                            val_adv_ref_future_image,
                        )
                    )
                writer.add_image(f"val Output viz {index}", val_output_viz, epoch)

            if args.log_terminal:
                logger.valid_bar.update(i)

    return errors.avg, error_names


def add_universal_perturbation(
    image0, image1, universal_perturbation, lower_bound=0.0, upper_bound=1.0
):
    if universal_perturbation.shape[1] != 2:
        raise Exception("Universarial perturbation: first dimension must be 2!")
    return (
        torch.clamp(image0 + universal_perturbation[0, 0], lower_bound, upper_bound),
        torch.clamp(image1 + universal_perturbation[:, 1], lower_bound, upper_bound),
    )


def isfloat(x) -> bool:
    try:
        _ = float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True


def isint(x) -> bool:
    try:
        a = float(x)
        b = int(a)
    except (TypeError, ValueError):
        return False
    else:
        return a == b


if __name__ == "__main__":
    args.perturb_method = args.perturb_method.lower()
    args.perturb_mode = args.perturb_mode.lower()
    args.disparity_method = args.disparity_method.lower()
    args.device = args.device.lower()
    if args.device not in [settings.GPU, settings.CPU, settings.CUDA]:
        args.device = settings.CUDA
    args.device = settings.CUDA if args.device == settings.GPU else args.device

    if isint(args.output_norm):
        args.output_norm = int(args.output_norm)
    elif isfloat(args.output_norm):
        args.output_norm = float(args.output_norm)

    args.return_feat_maps = False

    run(
        # Perturbations Settings
        perturb_method=args.perturb_method,
        perturb_mode=args.perturb_mode,
        output_norm=args.output_norm,
        # Stereo model settings
        # stereo_method=args.disparity_method,
        # stereo_model_restore_path=args.disparity_model_restore_path,
        # Output settings
        output_path=args.output_path,
        # Hardware settings
        device=args.device,
        args=args,
    )
