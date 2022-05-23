"""
We edited this file.

Authors: Alex Wong <alexw@cs.ucla.edu>, Mukund Mundhra <mukundmundhra@cs.ucla.edu>

If this code is useful to you, please consider citing the following paper:

A. Wong, M. Mundhra, and S. Soatto. Stereopagnosia: Fooling Stereo Networks with Adversarial Perturbations.
https://arxiv.org/pdf/2009.10142.pdf

@inproceedings{wong2021stereopagnosia,
  title={Stereopagnosia: Fooling Stereo Networks with Adversarial Perturbations},
  author={Wong, Alex and Mundhra, Mukund and Soatto, Stefano},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  year={2021}
}
"""
import functools
import os
import shutil
import time
from copy import deepcopy
from typing import Final

import imagecorruptions
import numpy as np
import torch
from path import Path
from PIL import Image
from tqdm import tqdm

import global_attacks.global_constants as settings
from dataset_utils import custom_transforms
from dataset_utils.kitti_datasets import KITTI2012, KITTI2015
from flowutils.flowlib import flow_to_image
from global_attacks.log_utils import (
    create_write_folder_structure,
    log,
    validate,
    write_out_sample,
)
from global_attacks.perturb_model import PerturbationsModel, compute_cossim, compute_epe
from models.utils_model import fetch_model, predict_flow, setup_hooks
from patch_attacks.logger import AverageMeter


def run(
    # Run settings
    n_height=settings.N_HEIGHT,
    n_width=settings.N_WIDTH,
    # Perturb method settings
    perturb_method=settings.PERTURB_METHOD,
    perturb_mode=settings.PERTURB_MODE,
    output_norm=settings.OUTPUT_NORM,
    n_step=settings.N_STEP,
    learning_rate=settings.LEARNING_RATE,
    momentum=settings.MOMENTUM,
    probability_diverse_input=settings.PROBABILITY_DIVERSE_INPUT,
    # Stereo model settings
    stereo_method=settings.STEREO_METHOD,
    stereo_model_restore_path=settings.STEREO_MODEL_RESTORE_PATH,
    # Output settings
    output_path="",
    # Hardware settings
    device=settings.DEVICE,
    args=None,
):
    if args.return_feat_maps:
        global activation  # pylint: disable=global-variable-not-assigned

    if device == settings.CUDA or device == settings.GPU:
        device = torch.device(settings.CUDA)
    else:
        device = torch.device(settings.CPU)

    if args.DEBUG:
        output_path = Path(output_path) / "DEBUG"

    output_path = Path(output_path) / args.dataset / args.flownet

    if args.targeted:
        output_path = Path(output_path) / "targeted"

    if args.universal_evaluation:
        output_path = Path(output_path) / "universal"

    output_path = Path(output_path) / perturb_mode
    if args.disparity:
        output_path = Path(output_path) / perturb_method / str(output_norm)
    elif perturb_method in imagecorruptions.get_corruption_names():
        if args.homogeneous:
            output_path = (
                Path(output_path) / "homogeneous" / f"{perturb_method}" / str(output_norm)
            )
        else:
            output_path = Path(output_path) / str(perturb_method) / str(output_norm)
    else:
        if args.homogeneous:
            output_path = Path(output_path) / "homogeneous"
        if args.arbitrary_gt_index is not None:
            output_path = Path(output_path) / "arbitraryGT" / f"{args.arbitrary_gt_index}"
        if args.arbitrary_noise_index is not None:
            output_path = (
                Path(output_path) / "arbitraryNoise" / f"{args.arbitrary_noise_index}"
            )
        output_path = (
            Path(output_path) / f"{perturb_method}_{args.flow_loss}" / str(output_norm)
        )

    if args.universal_evaluation:
        if args.uniform_noise:
            output_path = Path(output_path) / "uniform"
        else:
            output_path = Path(output_path) / args.folder_name

    print(f"Save everything to {output_path}")

    output_path.makedirs_p()

    # Set random seed. Might overwrite previous results if set!
    if args.seed > 0:
        settings.RANDOM_SEED = args.seed

    # Set up output and logging paths
    if args.universal_evaluation:
        log_path = output_path / "results.txt"
    else:
        log_path = os.path.join(output_path, f"results{settings.RANDOM_SEED}.txt")
        while os.path.isfile(log_path) and not args.seed > 0:
            settings.RANDOM_SEED += 1
            log_path = os.path.join(output_path, f"results{settings.RANDOM_SEED}.txt")
    logger: Final = functools.partial(
        log, filepath=log_path
    )  # for more convenient use of logger

    if args.write_out:
        logger(f"Storing image and depth outputs into {output_path}")
        output_paths = create_write_folder_structure(output_path, args.disparity)

    if args.show_evolve:
        evolve_path: Final = output_path / "evolve"
        try:
            shutil.rmtree(evolve_path)
        except Exception:
            pass
        evolve_path.makedirs_p()

    torch.manual_seed(settings.RANDOM_SEED)
    np.random.seed(settings.RANDOM_SEED)

    if device.type == settings.CUDA:
        torch.cuda.manual_seed(settings.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True

    if "di2" in args.perturb_method:
        perturb_method = perturb_method[3:]

    # Read input paths
    if args.dataset in ["kitti2012", "kitti2015"]:
        transforms = custom_transforms.Compose(
            [
                custom_transforms.Scale(h=args.n_height, w=args.n_width),
                custom_transforms.ArrayToTensorWoNorm(),
                custom_transforms.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
            ]
        )
        if "kitti2015" == args.dataset:
            dataset = KITTI2015(
                n_height=n_height,
                n_width=n_width,
                transform=transforms,
                disparity=args.disparity,
            )
        elif "kitti2012" == args.dataset:
            dataset = KITTI2012(
                n_height=n_height,
                n_width=n_width,
                transform=transforms,
                disparity=args.disparity,
            )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented!")

    n_sample = len(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    if args.disparity:
        stereo_model = None
        raise NotImplementedError
        # stereo_model = StereoModel(method=stereo_method, device=device)
        # # Restore stereo model
        # stereo_model.restore_model(stereo_model_restore_path)
    else:
        flow_net = fetch_model(
            args=args,
            return_feat_maps=args.return_feat_maps,
        )

        flow_net.eval()
        flow_net = flow_net.cuda()
        if args.return_feat_maps:
            setup_hooks(flow_net, args)

    logger("Run settings:")
    logger(f"n_height={n_height}  n_width={n_width}  output_norm={output_norm}")
    logger(f"perturb_method={perturb_method}  perturb_mode={perturb_mode}")
    logger(f"n_step={n_step}  learning_rate={learning_rate}")
    logger(f"momentum={momentum}")
    logger(f"probability_diverse_input={probability_diverse_input}")
    logger(f"seed={settings.RANDOM_SEED}")

    logger("Stereo model settings:")
    if args.disparity:
        logger(f"stereo_method={stereo_method}")
        logger(f"stereo_model_restore_path={stereo_model_restore_path}")
    else:
        logger(f"stereo_method={args.flownet}")

    logger("Output settings:")
    logger(f"output_path={output_path}")

    logger("Running...")

    time_start = time.time()
    time_per_frame = 0.0

    disparities_origin = []
    flows_origin = []
    noises0_output = []
    noises1_output = []
    disparities_output = []
    flows_output = []
    flows_noise_output = []
    ground_truths = []

    if "arbitrary_gt_index" in args and args.arbitrary_gt_index is not None:
        result_scene_path = os.path.join(output_path, "results_scenes.csv")
        result_file_path = os.path.join(output_path, "results.csv")
        error_names = [
            "epe_true_gt",
            "epe_any_gt",
            "adv_epe_true_gt",
            "adv_epe_any_gt",
            "cos_sim_true_gt",
            "cos_sim_any_gt",
            "adv_cos_true_gt",
            "adv_cos_any_gt",
        ]
        errors = AverageMeter(i=len(error_names))
        with open(result_file_path, "w", encoding="utf-8") as f:
            f.write(
                f"{error_names[0]}, {error_names[1]}, {error_names[2]}, {error_names[3]}, {error_names[4]}, {error_names[5]}, {error_names[6]}, {error_names[7]}\n"
            )
        with open(result_scene_path, "w", encoding="utf-8") as f:
            f.write(
                f"scene, {error_names[0]}, {error_names[1]}, {error_names[2]}, {error_names[3]}, {error_names[4]}, {error_names[5]}, {error_names[6]}, {error_names[7]}\n"
            )

        dataset = dataloader.dataset
        if args.arbitrary_gt_index.isdigit():
            args.arbitrary_gt_index = int(args.arbitrary_gt_index)
            if args.arbitrary_gt_index < len(dataset):
                arbitrary_sample = dataset[args.arbitrary_gt_index]
                if len(arbitrary_sample) == 4:
                    _, _, _, arbitrary_ground_truth = arbitrary_sample
                elif len(arbitrary_sample) == 5:
                    _, _, _, flow_downsampled, valid_downsampled = arbitrary_sample
                    arbitrary_ground_truth = torch.cat(
                        (flow_downsampled, valid_downsampled[None, ...])
                    )
                else:
                    _, _, _, arbitrary_ground_truth, _, _, _ = arbitrary_sample
            else:
                raise ValueError("No such sample!")
            h, w = args.n_height, args.n_width
            flow_gt = torch.nn.functional.interpolate(
                arbitrary_ground_truth[None, ...], (h, w), mode="area"
            ).numpy()[0]
            u = flow_gt[0, :, :]
            v = flow_gt[1, :, :]
            idxUnknow = (abs(u) > 1e7) | (abs(v) > 1e7)
            u[idxUnknow] = 0
            v[idxUnknow] = 0
            rad = np.sqrt(u**2 + v**2)
            maxrad = np.max(rad)
            flow_gt = flow_to_image(flow_gt[:2, ...].transpose(1, 2, 0), maxrad) / 255.0
            output_viz_im = Image.fromarray(np.uint8(255 * flow_gt))
            output_viz_im.save(os.path.join(output_path, "arbitrary_gt.png"))
            np.save(
                os.path.join(output_path, "arbitrary_gt.npy"),
                flow_gt,
            )
        else:
            arbitrary_sample = dataset[0]
            if len(arbitrary_sample) == 4:
                _, _, _, arbitrary_ground_truth_helper = arbitrary_sample
            elif len(arbitrary_sample) == 5:
                _, _, _, flow_downsampled, valid_downsampled = arbitrary_sample
                arbitrary_ground_truth_helper = torch.cat(
                    (flow_downsampled, valid_downsampled[None, ...])
                )
            else:
                _, _, _, arbitrary_ground_truth_helper, _, _, _ = arbitrary_sample
            if args.arbitrary_gt_index == "fun":
                arbitrary_ground_truth = (
                    torch.ones_like(arbitrary_ground_truth_helper) * -5
                )
                index_map = torch.zeros_like(
                    arbitrary_ground_truth_helper[0], dtype=torch.bool
                )
                thickness = 50
                margin = 20
                # 4
                index_map[0 + margin : 256 - margin, 225 : 225 + thickness] = True
                index_map[0 + margin : 128, 125 : 125 + thickness] = True
                index_map[
                    128 - thickness // 2 : 128 + thickness // 2, 125 : 225 + thickness
                ] = True
                arbitrary_ground_truth[0, index_map] = -90
                arbitrary_ground_truth[1, index_map] = -90

                # 2
                index_map = torch.zeros_like(
                    arbitrary_ground_truth_helper[0], dtype=torch.bool
                )
                index_map[
                    0 + margin : 0 + margin + thickness, 400 : 500 + thickness
                ] = True
                index_map[
                    128 - thickness // 2 : 128 + thickness // 2, 400 : 500 + thickness
                ] = True
                index_map[
                    256 - margin - thickness : 256 - margin, 400 : 500 + thickness
                ] = True
                index_map[0 + margin : 128, 500 : 500 + thickness] = True
                index_map[128 : 256 - margin, 400 : 400 + thickness] = True
                arbitrary_ground_truth[0, index_map] = 90
                arbitrary_ground_truth[1, index_map] = 90

                arbitrary_ground_truth[2] = 1

            elif "uniform" in args.arbitrary_gt_index:
                _shape = list(arbitrary_ground_truth_helper.shape)
                factor = int(
                    args.arbitrary_gt_index[args.arbitrary_gt_index.rfind("_") + 1 :]
                )
                _shape[1] = _shape[1] // factor
                _shape[2] = _shape[2] // factor
                _arbitrary_ground_truth = torch.FloatTensor(*_shape).uniform_(-180, 180)

                arbitrary_ground_truth = torch.nn.functional.interpolate(
                    _arbitrary_ground_truth[None, ...],
                    size=arbitrary_ground_truth_helper.shape[1:],
                    mode="nearest",
                )[0]
                arbitrary_ground_truth[2] = 1

            h, w = args.n_height, args.n_width
            flow_gt = torch.nn.functional.interpolate(
                arbitrary_ground_truth[None, ...], (h, w), mode="area"
            ).numpy()[0]
            u = flow_gt[0, :, :]
            v = flow_gt[1, :, :]
            idxUnknow = (abs(u) > 1e7) | (abs(v) > 1e7)
            u[idxUnknow] = 0
            v[idxUnknow] = 0
            rad = np.sqrt(u**2 + v**2)
            maxrad = np.max(rad)
            flow_gt = flow_to_image(flow_gt[:2, ...].transpose(1, 2, 0), maxrad) / 255.0
            output_viz_im = Image.fromarray(np.uint8(255 * flow_gt))
            output_viz_im.save(os.path.join(output_path, "arbitrary_gt.png"))
            np.save(
                os.path.join(output_path, "arbitrary_gt.npy"),
                flow_gt,
            )

    elif "arbitrary_noise_index" in args and args.arbitrary_noise_index is not None:
        dataset = dataloader.dataset
        if args.arbitrary_noise_index < len(dataset):
            arbitrary_sample = dataset[args.arbitrary_noise_index]
            if len(arbitrary_sample) == 3:
                (
                    arbitrary_image0,
                    arbitrary_image1,
                    arbitrary_ground_truth,
                    arbitrary_ground_truth_downsampled,
                ) = arbitrary_sample
            elif len(arbitrary_sample) == 5:
                (
                    arbitrary_image0,
                    arbitrary_image1,
                    arbitrary_ground_truth,
                    flow_downsampled,
                    valid_downsampled,
                ) = arbitrary_sample
                arbitrary_ground_truth_downsampled = torch.cat(
                    (flow_downsampled, valid_downsampled[None, ...])
                )
            else:
                (
                    _,
                    arbitrary_image0,
                    arbitrary_image1,
                    arbitrary_ground_truth,
                    _,
                    _,
                    _,
                ) = arbitrary_sample
        else:
            raise ValueError("No such sample!")

        arbitrary_image0 = deepcopy(arbitrary_image0)[None, ...]
        arbitrary_image1 = deepcopy(arbitrary_image1)[None, ...]
        arbitrary_ground_truth = deepcopy(arbitrary_ground_truth)[None, ...]
        if device.type == settings.CUDA:
            arbitrary_image0, arbitrary_image1, arbitrary_ground_truth = (
                arbitrary_image0.cuda(),
                arbitrary_image1.cuda(),
                arbitrary_ground_truth.cuda(),
            )
        perturb_model = PerturbationsModel(
            perturb_method=perturb_method,
            perturb_mode=perturb_mode,
            output_norm=output_norm,
            n_step=n_step,
            learning_rate=learning_rate,
            momentum=momentum,
            probability_diverse_input=probability_diverse_input,
            device=device,
            disparity=args.disparity,
            targeted=args.targeted,
            show_perturbation_evolution=evolve_path if args.show_evolve else None,
            args=args,
        )
        (arbitrary_noise0_output, arbitrary_noise1_output, _, _,) = perturb_model.forward(
            model=stereo_model if args.disparity else flow_net,
            image0=arbitrary_image0,
            image1=arbitrary_image1,
            ground_truth=arbitrary_ground_truth_downsampled,
        )

    elif args.universal_evaluation:
        if args.uniform_noise:
            universal_perturbation0 = (
                torch.rand((1, 3, 256, 640)) * 2 * output_norm - output_norm
            )
            universal_perturbation1 = (
                torch.rand((1, 3, 256, 640)) * 2 * output_norm - output_norm
            )
        else:
            universal_perturbation_path = (
                output_path / "perturbations" / f"epoch_{args.epoch_number}"
            )
            universal_perturbations = torch.load(universal_perturbation_path)
            universal_perturbation0 = universal_perturbations[:, 0, ...]
            universal_perturbation1 = universal_perturbations[:, 1, ...]

    for idx, data in tqdm(enumerate(dataloader)):
        if len(data) == 4:
            image0, image1, ground_truth, ground_truth_downsampled = data
        elif len(data) == 5:
            image0, image1, ground_truth, flow_downsampled, valid_downsampled = data
            ground_truth_downsampled = torch.cat(
                (flow_downsampled, valid_downsampled[:, None, ...]), dim=1
            )
        else:
            _, image0, image1, ground_truth, _, _, _ = data

        if args.homogeneous:
            # image0 = torch.ones_like(image0) * 0.5
            # image1 = torch.ones_like(image1) * 0.5
            image1 = image0
            ground_truth = torch.zeros_like(ground_truth)

        if device.type == settings.CUDA:
            image0, image1, ground_truth, ground_truth_downsampled = (
                image0.cuda(),
                image1.cuda(),
                ground_truth.cuda(),
                ground_truth_downsampled.cuda(),
            )
            if args.universal_evaluation:
                universal_perturbation0 = universal_perturbation0.cuda()
                universal_perturbation1 = universal_perturbation1.cuda()

        if len(image0.shape) == 4 and image0.shape[0] == 1 and args.disparity:
            ground_truth = torch.unsqueeze(ground_truth, dim=0)

        ground_truths.append(np.squeeze(ground_truth.cpu().numpy()))

        # Initialize perturbations and defense
        perturb_model = PerturbationsModel(
            perturb_method=perturb_method,
            perturb_mode=perturb_mode,
            output_norm=output_norm,
            n_step=n_step,
            learning_rate=learning_rate,
            momentum=momentum,
            probability_diverse_input=probability_diverse_input,
            device=device,
            disparity=args.disparity,
            targeted=args.targeted if "targeted" in args else False,
            show_perturbation_evolution=evolve_path
            if "show_evolve" in args and args.show_evolve
            else None,
            args=args,
        )

        # Forward through through stereo network
        with torch.no_grad():
            if args.disparity:
                disparity_origin = stereo_model.forward(image0, image1)
            else:
                if "return_feat_maps" in args and args.return_feat_maps:
                    flow_origin, _ = predict_flow(
                        flow_net,
                        None,
                        image0,
                        image1,
                        args,
                        return_feat_maps=True,
                    )
                else:
                    flow_origin = predict_flow(
                        flow_net,
                        None,
                        image0,
                        image1,
                        args,
                        return_feat_maps=False,
                    )
                epe = compute_epe(gt=ground_truth, pred=flow_origin)
                _, _, h, w = flow_origin.size()

        time_per_frame_start = time.time()

        # Optimize perturbations for the stereo pair and model
        if args.universal_evaluation:
            image0_output = torch.clamp(image0 + universal_perturbation0, 0.0, 1.0)
            image1_output = torch.clamp(image1 + universal_perturbation1, 0.0, 1.0)
            noise0_output = torch.clone(universal_perturbation0)
            noise1_output = torch.clone(universal_perturbation1)
        elif "arbitrary_noise_index" in args and args.arbitrary_noise_index is not None:
            image0_output = torch.clamp(image0 + arbitrary_noise0_output, 0.0, 1.0)
            image1_output = torch.clamp(image1 + arbitrary_noise1_output, 0.0, 1.0)
        else:
            if args.disparity:
                (
                    noise0_output,
                    noise1_output,
                    image0_output,
                    image1_output,
                ) = perturb_model.forward(
                    model=stereo_model if args.disparity else flow_net,
                    image0=image0,
                    image1=image1,
                    ground_truth=disparity_origin
                    if perturb_method == "self"
                    else ground_truth,
                )
            else:
                _ground_truth = (
                    ground_truth_downsampled
                    if args.arbitrary_gt_index is None
                    else deepcopy(arbitrary_ground_truth)[None, ...].cuda()
                )
                (
                    noise0_output,
                    noise1_output,
                    image0_output,
                    image1_output,
                ) = perturb_model.forward(
                    model=stereo_model if args.disparity else flow_net,
                    image0=image0,
                    image1=image1,
                    ground_truth=flow_origin
                    if perturb_method == "self"
                    else _ground_truth,
                )

        time_per_frame = time_per_frame + (time.time() - time_per_frame_start)

        # Forward through network again
        with torch.no_grad():
            if args.disparity:
                disparity_output = stereo_model.forward(image0_output, image1_output)
                loss_func = torch.nn.L1Loss()
                loss = loss_func(disparity_output, ground_truth)
            else:
                if "return_feat_maps" in args and args.return_feat_maps:
                    flow_output, _ = predict_flow(
                        flow_net,
                        None,
                        image0_output,
                        image1_output,
                        args,
                        return_feat_maps=True,
                    )
                else:
                    flow_output = predict_flow(
                        flow_net,
                        None,
                        image0_output,
                        image1_output,
                        args,
                        return_feat_maps=False,
                    )
                loss = compute_epe(gt=ground_truth, pred=flow_output)

                if (
                    "arbitrary_noise_index" in args
                    and args.arbitrary_noise_index is not None
                ):
                    noise0_output = torch.clone(arbitrary_noise0_output)
                    noise1_output = torch.clone(arbitrary_noise1_output)

                # noise0_output_t = (noise0_output + abs(noise0_output.min()))/(noise0_output + abs(noise0_output.min())).max()
                # noise1_output_t = (noise1_output + abs(noise1_output.min()))/(noise1_output + abs(noise1_output.min())).max()
                noise0_output_t = (noise0_output - noise0_output.min()) / (
                    noise0_output.max() - noise0_output.min()
                )
                noise1_output_t = (noise1_output - noise1_output.min()) / (
                    noise1_output.max() - noise1_output.min()
                )
                if "return_feat_maps" in args and args.return_feat_maps:
                    flow_noise_output, _ = predict_flow(
                        flow_net,
                        None,
                        noise0_output_t,
                        noise1_output_t,
                        args,
                        return_feat_maps=True,
                    )
                else:
                    flow_noise_output = predict_flow(
                        flow_net,
                        None,
                        noise0_output_t,
                        noise1_output_t,
                        args,
                        return_feat_maps=False,
                    )

        # Save outputs
        noises0_output.append(
            np.transpose(np.squeeze(noise0_output.detach().cpu().numpy()), (1, 2, 0))
        )
        noises1_output.append(
            np.transpose(np.squeeze(noise1_output.detach().cpu().numpy()), (1, 2, 0))
        )
        if args.disparity:
            disparities_origin.append(np.squeeze(disparity_origin.detach().cpu().numpy()))
            disparities_output.append(np.squeeze(disparity_output.detach().cpu().numpy()))
        else:
            flows_origin.append(np.squeeze(flow_origin.detach().cpu().numpy()))
            flows_output.append(np.squeeze(flow_output.detach().cpu().numpy()))
            flows_noise_output.append(
                np.squeeze(flow_noise_output.detach().cpu().numpy())
            )

        # Log results
        time_elapse = (time.time() - time_start) / 3600

        if args.disparity:
            logger(
                "Sample={:3}/{:3}  L1 Loss={:.5f}  Time Elapsed={:.2f}h".format(
                    idx + 1, n_sample, loss.item(), time_elapse
                )
            )
        else:
            logger(
                "Sample={:3}/{:3}  EPE Before={:.5f}  EPE After={:.5f}  Time Elapsed={:.2f}h".format(
                    idx + 1, n_sample, epe, loss, time_elapse
                )
            )
        if (
            "arbitrary_gt_index" in args
            and args.arbitrary_gt_index is not None
            and args.arbitrary_gt_index != idx
        ):
            logger(
                "Sample={:3}/{:3}  Arbitrary GT EPE Before={:.5f}  Arbitrary GT EPE After={:.5f}  Time Elapsed={:.2f}h".format(
                    idx + 1,
                    n_sample,
                    compute_epe(gt=_ground_truth, pred=flow_origin),
                    compute_epe(gt=_ground_truth, pred=flow_output),
                    time_elapse,
                )
            )

            true_gt = ground_truth
            any_gt = deepcopy(arbitrary_ground_truth)[None, ...].cuda()
            epe_true_gt = compute_epe(gt=true_gt, pred=flow_origin)
            epe_any_gt = compute_epe(gt=any_gt, pred=flow_origin)
            adv_epe_true_gt = compute_epe(gt=true_gt, pred=flow_output)
            adv_epe_any_gt = compute_epe(gt=any_gt, pred=flow_output)
            cossim_true_gt = compute_cossim(gt=true_gt, pred=flow_origin)
            cossim_any_gt = compute_cossim(gt=any_gt, pred=flow_origin)
            adv_cossim_true_gt = compute_cossim(gt=true_gt, pred=flow_output)
            adv_cossim_any_gt = compute_cossim(gt=any_gt, pred=flow_output)
            with open(result_scene_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{idx}, {round(epe_true_gt, 4)},{round(epe_any_gt, 4)}, {round(adv_epe_true_gt, 4)}, {round(adv_epe_any_gt, 4)}, {round(cossim_true_gt, 4)}, {round(cossim_any_gt, 4)}, {round(adv_cossim_true_gt, 4)}, {round(adv_cossim_any_gt, 4)}\n"
                )
            errors.update(
                [
                    epe_true_gt,
                    epe_any_gt,
                    adv_epe_true_gt,
                    adv_epe_any_gt,
                    cossim_true_gt,
                    cossim_any_gt,
                    adv_cossim_true_gt,
                    adv_cossim_any_gt,
                ]
            )

        if args.write_out:
            write_out_sample(
                output_paths=output_paths,
                idx=idx,
                is_disparity=args.disparity,
                unattacked_image0=np.transpose(
                    np.squeeze(image0.detach().cpu().numpy()), (1, 2, 0)
                ),
                unattacked_image1=np.transpose(
                    np.squeeze(image1.detach().cpu().numpy()), (1, 2, 0)
                ),
                unattacked_pred=np.squeeze(disparity_origin.detach().cpu().numpy())
                if args.disparity
                else np.squeeze(flow_origin.detach().cpu().numpy()),
                ground_truth=np.squeeze(ground_truth.cpu().numpy()),
                attacked_noise0=np.transpose(
                    np.squeeze(noise0_output.detach().cpu().numpy()), (1, 2, 0)
                ),
                attacked_noise1=np.transpose(
                    np.squeeze(noise1_output.detach().cpu().numpy()), (1, 2, 0)
                ),
                attacked_image0=np.transpose(
                    np.squeeze(image0_output.detach().cpu().numpy()), (1, 2, 0)
                ),
                attacked_image1=np.transpose(
                    np.squeeze(image1_output.detach().cpu().numpy()), (1, 2, 0)
                ),
                attacked_pred=np.squeeze(disparity_output.detach().cpu().numpy())
                if args.disparity
                else np.squeeze(flow_output.detach().cpu().numpy()),
                attacked_noise_pred=None
                if args.disparity
                else np.squeeze(flow_noise_output.detach().cpu().numpy()),
                write_out_npy=args.write_out_npy,
            )

        # Clean up
        del perturb_model
        del image0, image1
        del image0_output, image1_output
        del noise0_output, noise1_output
        if args.disparity:
            del disparity_origin, disparity_output
        else:
            del flow_origin, flow_output, flow_noise_output
        del ground_truth
        del loss

        if device.type == settings.CUDA:
            torch.cuda.empty_cache()

        if "homogeneous" in args and args.homogeneous:
            break

    if "arbitrary_gt_index" in args and args.arbitrary_gt_index is not None:
        (
            avg_epe_true_gt,
            avg_epe_any_gt,
            avg_adv_epe_true_gt,
            avg_adv_epe_any_gt,
            avg_cossim_true_gt,
            avg_cossim_any_gt,
            avg_adv_cossim_true_gt,
            avg_adv_cossim_any_gt,
        ) = errors.avg
        with open(result_scene_path, "a", encoding="utf-8") as f:
            f.write(
                f"{idx}, {round(avg_epe_true_gt, 4)},{round(avg_epe_any_gt, 4)}, {round(avg_adv_epe_true_gt, 4)}, {round(avg_adv_epe_any_gt, 4)}, {round(avg_cossim_true_gt, 4)}, {round(avg_cossim_any_gt, 4)}, {round(avg_adv_cossim_true_gt, 4)}, {round(avg_adv_cossim_any_gt, 4)}\n"  # pylint: disable=undefined-loop-variable
            )
        with open(result_file_path, "a", encoding="utf-8") as f:
            f.write(
                f"avg, {round(avg_epe_true_gt, 4)},{round(avg_epe_any_gt, 4)}, {round(avg_adv_epe_true_gt, 4)}, {round(avg_adv_epe_any_gt, 4)}, {round(avg_cossim_true_gt, 4)}, {round(avg_cossim_any_gt, 4)}, {round(avg_adv_cossim_true_gt, 4)}, {round(avg_adv_cossim_any_gt, 4)}\n"
            )

    # Perform validation
    with torch.no_grad():
        validate(
            noises0_output=noises0_output,
            noises1_output=noises1_output,
            origins=disparities_origin if args.disparity else flows_origin,
            outputs=disparities_output if args.disparity else flows_output,
            ground_truths=ground_truths,
            logger=logger,
            output_path=output_path,
            is_disparity=args.disparity,
            seed=settings.RANDOM_SEED,
        )

    logger(f"Time per frame: {time_per_frame / len(dataloader)}s")
