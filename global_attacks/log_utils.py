"""
Edited from

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

Extended!
"""
import json
import os

import numpy as np
import torch
from PIL import Image

import global_attacks.eval_utils as eval_utils
from flowutils.flowlib import flow_to_image
from global_attacks.perturb_model import compute_cossim, compute_epe, compute_l1


def get_output_paths_dict(output_path):
    folder_names = [
        "image0_output",
        "image1_output",
        "noise0_output",
        "noise1_output",
        "disparity_origin",
        "disparity_output",
        "flow_origin",
        "flow_output",
        "flow_noise_output",
        "ground_truth",
        "output_viz",
    ]
    output_path_keys = [
        "image0_output_path",
        "image1_output_path",
        "noise0_output_path",
        "noise1_output_path",
        "disparity_origin_path",
        "disparity_output_path",
        "flow_origin_path",
        "flow_output_path",
        "flow_noise_output_path",
        "ground_truth_path",
        "output_vis_dir",
    ]
    output_paths = {
        key: os.path.join(output_path, folder_name)
        for key, folder_name in zip(output_path_keys, folder_names)
    }
    return output_paths


def create_write_folder_structure(output_path, is_disparity: bool = True):
    output_paths = get_output_paths_dict(output_path)
    for _, folder in output_paths.items():
        if is_disparity and "flow" in folder:
            continue
        if not is_disparity and "disparity" in folder:
            continue
        folder.makedirs_p()
    return output_paths


def write_out_sample(
    output_paths,
    idx,
    is_disparity,
    unattacked_image0,
    unattacked_image1,
    unattacked_pred,
    ground_truth,
    attacked_noise0,
    attacked_noise1,
    attacked_image0,
    attacked_image1,
    attacked_pred,
    attacked_noise_pred=None,
    write_out_npy: bool = True,
):
    # image_filename = f"{idx:05d}.png"
    numpy_filename = f"{idx:05d}.npy"

    # Save to disk
    # Image.fromarray(np.uint8(unattacked_image0 * 255.0)).save(os.path.join(output_paths['image0_output_path'], image_filename))
    # Image.fromarray(np.uint8(unattacked_image1 * 255.0)).save(os.path.join(output_paths['image1_output_path'], image_filename))

    if write_out_npy:
        np.save(
            os.path.join(output_paths["noise0_output_path"], numpy_filename),
            attacked_noise0,
        )
        np.save(
            os.path.join(output_paths["noise1_output_path"], numpy_filename),
            attacked_noise1,
        )

    # attacked_noise0 = (attacked_noise0 + abs(attacked_noise0.min()))/(attacked_noise0 + abs(attacked_noise0.min())).max()
    # attacked_noise1 = (attacked_noise1 + abs(attacked_noise1.min()))/(attacked_noise1 + abs(attacked_noise1.min())).max()
    attacked_noise0 = (attacked_noise0 - attacked_noise0.min()) / (
        attacked_noise0.max() - attacked_noise0.min()
    )
    attacked_noise1 = (attacked_noise1 - attacked_noise1.min()) / (
        attacked_noise1.max() - attacked_noise1.min()
    )

    if write_out_npy:
        np.save(
            os.path.join(output_paths["ground_truth_path"], numpy_filename), ground_truth
        )
    if not is_disparity:
        _, h, w = unattacked_pred.shape
        ground_truth = torch.nn.functional.interpolate(
            torch.from_numpy(ground_truth[None, ...]), (h, w), mode="area"
        ).numpy()[0]
        u = ground_truth[0, :, :]
        v = ground_truth[1, :, :]
        idxUnknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxUnknow] = 0
        v[idxUnknow] = 0
        rad = np.sqrt(u**2 + v**2)
        maxrad = np.max(rad)
        flow_gt = flow_to_image(ground_truth[:2, ...].transpose(1, 2, 0), maxrad) / 255.0

    if is_disparity:
        if write_out_npy:
            np.save(
                os.path.join(output_paths["disparity_origin_path"], numpy_filename),
                unattacked_pred,
            )
            np.save(
                os.path.join(output_paths["disparity_output_path"], numpy_filename),
                attacked_pred,
            )
    else:
        if write_out_npy:
            np.save(
                os.path.join(output_paths["flow_origin_path"], numpy_filename),
                unattacked_pred,
            )
            np.save(
                os.path.join(output_paths["flow_output_path"], numpy_filename),
                attacked_pred,
            )
            np.save(
                os.path.join(output_paths["flow_noise_output_path"], numpy_filename),
                attacked_noise_pred,
            )

        flow_unattacked = (
            flow_to_image(unattacked_pred.transpose(1, 2, 0), maxrad) / 255.0
        )
        flow_attacked = flow_to_image(attacked_pred.transpose(1, 2, 0), maxrad) / 255.0
        flow_noise = flow_to_image(attacked_noise_pred.transpose(1, 2, 0), maxrad) / 255.0

    if not is_disparity:
        output_viz = np.concatenate(
            (
                unattacked_image0,
                attacked_image0,
                attacked_noise0,
                flow_noise,
                flow_unattacked,
                unattacked_image1,
                attacked_image1,
                attacked_noise1,
                flow_gt,
                flow_attacked,
            ),
            1,
        )
        h, w, _ = output_viz.shape
        out_viz_arranged = np.zeros((2 * h, w // 2, 3))
        out_viz_arranged[:h, ...] = output_viz[:, : w // 2, :]
        out_viz_arranged[h:, ...] = output_viz[:, w // 2 :, :]
        if write_out_npy:
            np.save(
                output_paths["output_vis_dir"] / "viz" + str(idx).zfill(3) + ".npy",
                out_viz_arranged,
            )
        output_viz_im = Image.fromarray(np.uint8(255 * out_viz_arranged))
        output_viz_im.save(
            output_paths["output_vis_dir"] / "viz" + str(idx).zfill(3) + ".png"
        )
    else:
        raise NotImplementedError


def log(s, filepath=None, to_console=True):
    """
    Logs a string to either file or console

    Args:
        s : str
            string to log
        filepath
            output filepath for logging
        to_console : bool
            log to console
    """
    if to_console:
        print(s)
    if filepath is not None:
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
            with open(filepath, "w+", encoding="utf-8") as o:
                o.write(s + "\n")
        else:
            with open(filepath, "a+", encoding="utf-8") as o:
                o.write(s + "\n")


def validate(
    noises0_output,
    noises1_output,
    origins,
    outputs,
    ground_truths,
    logger,
    output_path,
    is_disparity,
    seed,
):

    n_sample = len(outputs)

    assert n_sample == len(noises0_output)
    assert n_sample == len(noises1_output)
    assert n_sample == len(ground_truths)

    # Noise metrics
    noise0_l0pix = np.zeros(n_sample)
    noise0_l1pix = np.zeros(n_sample)
    noise1_l0pix = np.zeros(n_sample)
    noise1_l1pix = np.zeros(n_sample)

    # Disparity metrics
    if is_disparity:
        disparity_mae_attacked = np.zeros(n_sample)
        disparity_rmse_attacked = np.zeros(n_sample)
        disparity_d1_attacked = np.zeros(n_sample)
        disparity_mae_unattacked = np.zeros(n_sample)
        disparity_rmse_unattacked = np.zeros(n_sample)
        disparity_d1_unattacked = np.zeros(n_sample)
    else:
        flow_epe_origin = np.zeros(n_sample)
        flow_epe = np.zeros(n_sample)
        flow_cossim_unattacked = np.zeros(n_sample)
        flow_cossim_attacked = np.zeros(n_sample)
        flow_l1_unattacked = np.zeros(n_sample)
        flow_l1_attacked = np.zeros(n_sample)

    data = zip(noises0_output, noises1_output, origins, outputs, ground_truths)

    for idx, (noise0_output, noise1_output, origin, output, ground_truth) in enumerate(
        data
    ):

        # Compute noise metrics
        noise0_l0pix[idx] = eval_utils.lp_norm(noise0_output, p=0)
        noise0_l1pix[idx] = eval_utils.lp_norm(noise0_output, p=1, axis=-1)
        noise1_l0pix[idx] = eval_utils.lp_norm(noise1_output, p=0)
        noise1_l1pix[idx] = eval_utils.lp_norm(noise1_output, p=1, axis=-1)

        # Mask out invalid ground truth
        if is_disparity:
            mask = np.logical_and(ground_truth > 0.0, ~np.isnan(ground_truth))
        else:
            mask = np.logical_or(ground_truth[2, ...], np.isnan(ground_truth))

        # Compute disparity metrics
        if is_disparity:
            disparity_mae_unattacked[idx] = eval_utils.mean_abs_err(
                origin[mask], ground_truth[mask]
            )
            disparity_mae_attacked[idx] = eval_utils.mean_abs_err(
                output[mask], ground_truth[mask]
            )
            disparity_rmse_unattacked[idx] = eval_utils.root_mean_sq_err(
                origin[mask], ground_truth[mask]
            )
            disparity_rmse_attacked[idx] = eval_utils.root_mean_sq_err(
                output[mask], ground_truth[mask]
            )
            disparity_d1_unattacked[idx] = eval_utils.d1_error(
                origin[mask], ground_truth[mask]
            )
            disparity_d1_attacked[idx] = eval_utils.d1_error(
                output[mask], ground_truth[mask]
            )
        else:
            flow_epe_origin[idx] = compute_epe(
                gt=torch.FloatTensor(ground_truth[None, ...]),
                pred=torch.FloatTensor(origin[None, ...]),
            )
            flow_epe[idx] = compute_epe(
                gt=torch.FloatTensor(ground_truth[None, ...]),
                pred=torch.FloatTensor(output[None, ...]),
            )
            flow_cossim_unattacked[idx] = compute_cossim(
                gt=torch.FloatTensor(ground_truth[None, ...]),
                pred=torch.FloatTensor(origin[None, ...]),
            )
            flow_cossim_attacked[idx] = compute_cossim(
                gt=torch.FloatTensor(ground_truth[None, ...]),
                pred=torch.FloatTensor(output[None, ...]),
            )
            flow_l1_unattacked[idx] = compute_l1(
                gt=torch.FloatTensor(ground_truth[None, ...]),
                pred=torch.FloatTensor(origin[None, ...]),
            )
            flow_l1_attacked[idx] = compute_l1(
                gt=torch.FloatTensor(ground_truth[None, ...]),
                pred=torch.FloatTensor(output[None, ...]),
            )

    # Noise metrics
    noise0_l0pix = np.mean(noise0_l0pix)
    noise0_l1pix = np.mean(noise0_l1pix)

    noise1_l0pix = np.mean(noise1_l0pix)
    noise1_l1pix = np.mean(noise1_l1pix)

    # Disparity metrics
    if is_disparity:
        disparity_mae_unattacked_std = np.std(disparity_mae_unattacked)
        disparity_mae_unattacked_mean = np.mean(disparity_mae_unattacked)

        disparity_mae_attacked_std = np.std(disparity_mae_attacked)
        disparity_mae_attacked_mean = np.mean(disparity_mae_attacked)

        disparity_rmse_unattacked_std = np.std(disparity_rmse_unattacked)
        disparity_rmse_unattacked_mean = np.mean(disparity_rmse_unattacked)

        disparity_rmse_attacked_std = np.std(disparity_rmse_attacked)
        disparity_rmse_attacked_mean = np.mean(disparity_rmse_attacked)

        disparity_d1_unattacked_std = np.std(disparity_d1_unattacked * 100.0)
        disparity_d1_unattacked_mean = np.mean(disparity_d1_unattacked * 100.0)

        disparity_d1_attacked_std = np.std(disparity_d1_attacked * 100.0)
        disparity_d1_attacked_mean = np.mean(disparity_d1_attacked * 100.0)
    else:
        flow_l1_unattacked_std = np.std(flow_l1_unattacked)
        flow_l1_unattacked_mean = np.mean(flow_l1_unattacked)

        flow_l1_attacked_std = np.std(flow_l1_attacked)
        flow_l1_attacked_mean = np.mean(flow_l1_attacked)

        flow_epe_std = np.std(flow_epe)
        flow_epe_mean = np.mean(flow_epe)

        flow_epe_origin_std = np.std(flow_epe_origin)
        flow_epe_origin_mean = np.mean(flow_epe_origin)

        flow_cossim_unattacked_std = np.std(flow_cossim_unattacked)
        flow_cossim_unattacked_mean = np.mean(flow_cossim_unattacked)

        flow_cossim_attacked_std = np.std(flow_cossim_attacked)
        flow_cossim_attacked_mean = np.mean(flow_cossim_attacked)

    logger("Validation results:")

    logger("{:<14}  {:>10}  {:>10}".format("Noise0:", "L0 Pixel", "L1 Pixel"))
    logger("{:<14}  {:10.4f}  {:10.4f}".format("", noise0_l0pix, noise0_l1pix))

    logger("{:<14}  {:>10}  {:>10}".format("Noise1:", "L0 Pixel", "L1 Pixel"))
    logger("{:<14}  {:10.4f}  {:10.4f}".format("", noise1_l0pix, noise1_l1pix))

    results_dict = {
        "noise0": {
            "L0_Pixel": noise0_l0pix,
            "L1_Pixel": noise0_l1pix,
        },
        "noise1": {
            "L0_Pixel": noise1_l0pix,
            "L1_Pixel": noise1_l1pix,
        },
    }

    if is_disparity:
        logger("{:<14}  {:>10}  {:>10}".format("Disparity:", "MAE", "+/-"))
        logger(
            "{:<14}  {:>10.4f}  {:>10.4f}".format(
                "", disparity_mae_unattacked_mean, disparity_mae_unattacked_std
            )
        )
        logger(
            "{:<14}  {:>10.4f}  {:>10.4f}".format(
                "", disparity_mae_attacked_mean, disparity_mae_attacked_std
            )
        )

        logger(
            "{:<14}  {:>10}  {:>10}".format(
                "",
                "RMSE",
                "+/-",
            )
        )
        logger(
            "{:<14}  {:>10.4f}  {:>10.4f}".format(
                "", disparity_rmse_unattacked_mean, disparity_rmse_unattacked_std
            )
        )
        logger(
            "{:<14}  {:>10.4f}  {:>10.4f}".format(
                "", disparity_rmse_attacked_mean, disparity_rmse_attacked_std
            )
        )

        logger(
            "{:<14}  {:>10}  {:>10}".format(
                "",
                "D1-Error",
                "+/-",
            )
        )
        logger(
            "{:<14}  {:>10.4f}  {:>10.4f}".format(
                "", disparity_d1_unattacked_mean, disparity_d1_unattacked_std
            )
        )
        logger(
            "{:<14}  {:>10.4f}  {:>10.4f}".format(
                "", disparity_d1_attacked_mean, disparity_d1_attacked_std
            )
        )

        results_dict["disparity"] = {
            "Unattacked": {
                "MAE_mean": disparity_mae_unattacked_mean,
                "MAE_std": disparity_mae_unattacked_std,
                "RMSE_mean": disparity_rmse_unattacked_mean,
                "RMSE_std": disparity_rmse_unattacked_std,
                "D1_mean": disparity_d1_unattacked_mean,
                "D1_std": disparity_d1_unattacked_std,
            },
            "Attacked": {
                "MAE_mean": disparity_mae_attacked_mean,
                "MAE_std": disparity_mae_attacked_std,
                "RMSE_mean": disparity_rmse_attacked_mean,
                "RMSE_std": disparity_rmse_attacked_std,
                "D1_mean": disparity_d1_attacked_mean,
                "D1_std": disparity_d1_attacked_std,
            },
        }
    else:
        logger("{:<14}  {:>10}  {:>10}".format("Flow:", "EPE", "+/-"))
        logger(
            "{:<14}  {:>10.4f}  {:>10.4f}".format(
                "", flow_epe_origin_mean, flow_epe_origin_std
            )
        )
        logger("{:<14}  {:>10.4f}  {:>10.4f}".format("", flow_epe_mean, flow_epe_std))

        logger(
            "{:<14}  {:>10}  {:>10}".format(
                "",
                "L1",
                "+/-",
            )
        )
        logger(
            "{:<14}  {:>10.4f}  {:>10.4f}".format(
                "", flow_l1_unattacked_mean, flow_l1_unattacked_std
            )
        )
        logger(
            "{:<14}  {:>10.4f}  {:>10.4f}".format(
                "", flow_l1_attacked_mean, flow_l1_attacked_std
            )
        )

        logger(
            "{:<14}  {:>10}  {:>10}".format(
                "",
                "CosSim",
                "+/-",
            )
        )
        logger(
            "{:<14}  {:>10.4f}  {:>10.4f}".format(
                "", flow_cossim_unattacked_mean, flow_cossim_unattacked_std
            )
        )
        logger(
            "{:<14}  {:>10.4f}  {:>10.4f}".format(
                "", flow_cossim_attacked_mean, flow_cossim_attacked_std
            )
        )

        results_dict["flow"] = {
            "Unattacked": {
                "EPE_mean": flow_epe_origin_mean,
                "EPE_std": flow_epe_origin_std,
                "L1_mean": flow_l1_unattacked_mean,
                "L1_std": flow_l1_unattacked_std,
                "CosSim_mean": flow_cossim_unattacked_mean,
                "CosSim_std": flow_cossim_unattacked_std,
            },
            "Attacked": {
                "EPE_mean": flow_epe_mean,
                "EPE_std": flow_epe_std,
                "L1_mean": flow_l1_attacked_mean,
                "L1_std": flow_l1_attacked_std,
                "CosSim_mean": flow_cossim_attacked_mean,
                "CosSim_std": flow_cossim_attacked_std,
            },
        }

    with open(
        os.path.join(output_path, f"results{seed}.json"), "w", encoding="utf-8"
    ) as fp:
        json.dump(results_dict, fp, indent=4)
