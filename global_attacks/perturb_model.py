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
import os
import random
import sys
from typing import Union

import imagecorruptions
import imageio
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.util import random_noise
from torch.autograd import Variable

import global_attacks.global_constants as settings
import global_attacks.imagecorruptions_frost as imagecorruptions_frost
from models.utils_model import predict_flow


def compute_epe(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    u_gt, v_gt = gt[:, 0, :, :], gt[:, 1, :, :]
    pred = nn.functional.upsample(pred, size=(h_gt, w_gt), mode="bilinear")
    u_pred = pred[:, 0, :, :] * (w_gt / w_pred)
    v_pred = pred[:, 1, :, :] * (h_gt / h_pred)

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))

    if nc == 3:
        valid = gt[:, 2, :, :]
        epe = epe * valid
        avg_epe = epe.sum() / (valid.sum() + epsilon)
    else:
        avg_epe = epe.sum() / (bs * h_gt * w_gt)

    if isinstance(avg_epe, Variable):
        avg_epe = avg_epe.data

    return avg_epe.item()


def compute_cossim(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    pred = nn.functional.upsample(pred, size=(h_gt, w_gt), mode="bilinear")
    pred[:, 0, :, :] *= w_gt / w_pred
    pred[:, 1, :, :] *= h_gt / h_pred

    similarity = nn.functional.cosine_similarity(gt[:, :2], pred)
    if nc == 3:
        valid = gt[:, 2, :, :]
        similarity = similarity * valid
        avg_sim = similarity.sum() / (valid.sum() + epsilon)
    else:
        avg_sim = similarity.sum() / (bs * h_gt * w_gt)

    if isinstance(avg_sim, Variable):
        avg_sim = avg_sim.data

    return avg_sim.item()


def compute_l1(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8

    _, _, h_pred, w_pred = pred.size()
    _, _, h_gt, w_gt = gt.size()
    pred = nn.functional.upsample(pred, size=(h_gt, w_gt), mode="bilinear")
    pred[:, 0, :, :] *= w_gt / w_pred
    pred[:, 1, :, :] *= h_gt / h_pred
    i_loss = (pred - gt[:, :2, ...]).abs()
    l1_loss = torch.mean(i_loss[torch.logical_not(torch.isnan(i_loss))])
    if gt.shape[1] == 3:
        valid = gt[:, 2, ...]
        loss = l1_loss * valid
        return loss.sum() / (valid.sum() + epsilon)
    else:
        return l1_loss.mean()


def compute_flow_loss(
    flow_net: nn.Module,
    image0: torch.Tensor,
    image1: torch.Tensor,
    ground_truth: torch.Tensor,
    args,
) -> torch.Tensor:
    epsilon = 1e-8

    if "return_feat_maps" in args and args.return_feat_maps:
        flow_output, _ = predict_flow(
            flow_net,
            None,
            image0,
            image1,
            args,
            return_feat_maps=True,
        )
    else:
        flow_output = predict_flow(
            flow_net,
            None,
            image0,
            image1,
            args,
        )

    if args.flow_loss == "cossim":
        loss = 1 - nn.functional.cosine_similarity(flow_output, ground_truth[:, :2, ...])
    elif args.flow_loss == "l2":
        loss = (
            torch.sum((flow_output - ground_truth[:, :2, ...]) ** 2, dim=1) + 10e-8
        ).sqrt()
    elif args.flow_loss == "l1":
        loss = (flow_output - ground_truth[:, :2, ...]).abs()
    else:
        raise NotImplementedError

    if ground_truth.shape[1] == 3:
        valid = ground_truth[:, 2, ...]
        loss = loss * valid
        return loss.sum() / (valid.sum() + epsilon)
    else:
        return loss.mean()


class PerturbationsModel:
    """
    Adversarial perturbation model

    Args:
        perturb_method : str
            method to use to generate perturbations
        perturb_mode : str
            left, right, or both
        output_norm : float
            upper (infinity) norm of adversarial noise
        n_step : int
            number of steps to optimize perturbations
        learning_rate : float
            learning rate (alpha) to be used for optimizing perturbations
        momentum : float
            momemtum (mu) used for momentum iterative fast gradient sign method
        probability_diverse_input : float
            probability to use diverse input
        device : torch.device
            device to run optimization
    """

    def __init__(
        self,
        perturb_method: str = settings.PERTURB_METHOD,
        perturb_mode: str = settings.PERTURB_MODE,
        output_norm: Union[int, float] = settings.OUTPUT_NORM,
        n_step: int = settings.N_STEP,
        learning_rate: float = settings.LEARNING_RATE,
        momentum: float = settings.MOMENTUM,
        probability_diverse_input: float = settings.PROBABILITY_DIVERSE_INPUT,
        device=torch.device(settings.CUDA),
        disparity: bool = True,
        targeted: bool = False,
        show_perturbation_evolution=None,  # only sensible for iterative perturbation methods
        print_out: bool = True,
        args=None,
    ):

        self.__perturb_method = perturb_method
        self.__perturb_mode = perturb_mode
        if self.__perturb_method in imagecorruptions.get_corruption_names():
            self.__severity = int(output_norm)
        else:
            self.__output_norm = output_norm
        self.__n_step = n_step
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__probability_diverse_input = probability_diverse_input

        self.__device = device
        self.__disparity = disparity
        self.__targeted = targeted
        if show_perturbation_evolution:
            assert self.__perturb_method in ["ifgsm", "mifgsm"]
            self.__show_perturbation_evolution = show_perturbation_evolution
        else:
            self.__show_perturbation_evolution = None
        self.__print_out = print_out
        self.__args = args

    def forward(self, model, image0, image1, ground_truth):
        """
        Applies perturbations to image and clamp

        Args:
            model : object
                network
            image0 : tensor
                N x C x H x W RGB image
            image1 : tensor
                N x C x H x W RGB image
            ground_truth : tensor
                N x 1 x H x W ground truth disparity

        Returns:
            tensor : adversarial noise/perturbations for left image
            tensor : adversarial noise/perturbations for right image
            tensor : adversarially perturbed left image
            tensor : adversarially perturbed right image
        """

        if self.__perturb_method == "fgsm" or self.__perturb_method == "fgm":
            noise0, noise1 = self.__fgsm(model, image0, image1, ground_truth)

        elif self.__perturb_method == "ifgsm" or self.__perturb_method == "ifgm":
            noise0, noise1 = self.__ifgsm(model, image0, image1, ground_truth)

        elif self.__perturb_method == "mifgsm" or self.__perturb_method == "mifgm":
            noise0, noise1 = self.__mifgsm(model, image0, image1, ground_truth)

        elif self.__perturb_method == "gaussian":
            noise0, noise1 = self.__gaussian(image0, image1)

        elif self.__perturb_method == "uniform":
            noise0, noise1 = self.__uniform(image0, image1)

        elif self.__perturb_method in imagecorruptions.get_corruption_names():
            image0_output, image1_output = self.__image_corruptions(image0, image1)
            image0_output = torch.clamp(image0_output, 0.0, 1.0)
            image1_output = torch.clamp(image1_output, 0.0, 1.0)
            return (
                image0_output - image0,
                image1_output - image1,
                image0_output,
                image1_output,
            )

        elif self.__perturb_method == "none":
            noise0 = torch.zeros_like(image0)
            noise1 = torch.zeros_like(image1)

        else:
            raise ValueError("Invalid perturbation method: %s" % self.__perturb_method)

        # Add to image and clamp between supports of image intensity to get perturbed image
        image0_output = torch.clamp(image0 + noise0, 0.0, 1.0)
        image1_output = torch.clamp(image1 + noise1, 0.0, 1.0)

        # Output perturbations are the difference between original and perturbed image
        noise0_output = image0_output - image0
        noise1_output = image1_output - image1

        return noise0_output, noise1_output, image0_output, image1_output

    def __gaussian(self, image0, image1):
        """
        Computes gaussian noise as perturbations

        Args:
            image0 : tensor
                N x C x H x W RGB image
            image1 : tensor
                N x C x H x W RGB image

        Returns:
            tensor : gaussian noise/perturbations for left image
            tensor : gaussian noise/perturbations for right image
        """

        variance = (self.__output_norm / 4.0) ** 2

        # Apply gaussian noise to images
        image0_output = random_noise(
            image0.clone().detach().cpu().numpy(), mode="gaussian", var=variance
        )

        image1_output = random_noise(
            image1.clone().detach().cpu().numpy(), mode="gaussian", var=variance
        )

        image0_output = torch.from_numpy(image0_output).float()
        image1_output = torch.from_numpy(image1_output).float()

        if self.__device.type == settings.CUDA:
            image0 = image0.cuda()
            image1 = image1.cuda()
            image0_output = image0_output.cuda()
            image1_output = image1_output.cuda()
        else:
            image0 = image0.cpu()
            image1 = image1.cpu()
            image0_output = image0_output.cpu()
            image1_output = image1_output.cpu()

        # Subtract perturbed images from original images to get noise
        if self.__perturb_mode == "both":
            noise0_output = image0_output - image0
            noise1_output = image1_output - image1

        elif self.__perturb_mode == "left":
            noise0_output = image0_output - image0
            noise1_output = torch.zeros_like(image1)

        elif self.__perturb_mode == "right":
            noise0_output = torch.zeros_like(image0)
            noise1_output = image1_output - image1

        else:
            raise ValueError("Invalid perturbation mode: %s" % self.__perturb_mode)

        return noise0_output, noise1_output

    def __uniform(self, image0, image1):
        """
        Computes uniform noise as pertubations

        Args:
            image0 : tensor
                N x C x H x W RGB image
            image1 : tensor
                N x C x H x W RGB image

        Returns:
            tensor : uniform noise/perturbations for left image
            tensor : uniform noise/perturbations for right image
        """

        # Compute uniform noise to images
        noise0_output = np.random.uniform(
            size=image0.clone().detach().cpu().numpy().shape,
            low=-self.__output_norm,
            high=self.__output_norm,
        )

        noise1_output = np.random.uniform(
            size=image1.clone().detach().cpu().numpy().shape,
            low=-self.__output_norm,
            high=self.__output_norm,
        )

        if self.__perturb_mode == "both":
            noise0_output = torch.from_numpy(noise0_output).float()
            noise1_output = torch.from_numpy(noise1_output).float()

        elif self.__perturb_mode == "left":
            noise0_output = torch.from_numpy(noise0_output).float()
            noise1_output = torch.zeros_like(image1)

        elif self.__perturb_mode == "right":
            noise0_output = torch.zeros_like(image0)
            noise1_output = torch.from_numpy(noise1_output).float()

        else:
            raise ValueError("Invalid perturbation mode: %s" % self.__perturb_mode)

        if self.__device.type == settings.CUDA:
            noise0_output = noise0_output.cuda()
            noise1_output = noise1_output.cuda()
        else:
            noise0_output = noise0_output.cpu()
            noise1_output = noise1_output.cpu()

        return noise0_output, noise1_output

    def __image_corruptions(self, image0, image1):
        def float32touint8(img: np.ndarray) -> np.ndarray:
            return (img * 255).astype(np.uint8)

        def uint8tofloat32(img: np.ndarray) -> np.ndarray:
            return (img / 255).astype(np.float32)

        img0 = float32touint8(image0.cpu().numpy())[0, ...].transpose(1, 2, 0)
        if self.__perturb_mode == "same" and self.__perturb_method == "frost":
            idx = np.random.randint(5)
            img0 = imagecorruptions_frost.frost(
                Image.fromarray(img0), self.__severity, idx=idx
            ).astype(np.uint8)
        else:
            img0 = imagecorruptions.corrupt(
                img0, corruption_name=self.__perturb_method, severity=self.__severity
            )
        image0 = torch.from_numpy(
            uint8tofloat32(img0.transpose(2, 0, 1)[None, ...])
        ).cuda()

        img1 = float32touint8(image1.cpu().numpy())[0, ...].transpose(1, 2, 0)
        if self.__perturb_mode == "same" and self.__perturb_method == "frost":
            img1 = imagecorruptions_frost.frost(
                Image.fromarray(img1), self.__severity, idx=idx
            ).astype(np.uint8)
        else:
            img1 = imagecorruptions.corrupt(
                img1, corruption_name=self.__perturb_method, severity=self.__severity
            )
        image1 = torch.from_numpy(
            uint8tofloat32(img1.transpose(2, 0, 1)[None, ...])
        ).cuda()

        return image0, image1

    #################################################
    #              Gradient-based                   #
    #################################################
    def __fgsm(self, model, image0, image1, ground_truth):
        """
        Computes adversarial perturbations using fast gradient sign method

        Args:
            stereo_model : object
                stereo network
            image0 : tensor
                N x C x H x W RGB image
            image1 : tensor
                N x C x H x W RGB image
            ground_truth : tensor
                N x 1 x H x W ground truth disparity

        Returns:
            tensor : uniform noise/perturbations for left image
            tensor : uniform noise/perturbations for right image
        """

        # Set gradients for image to be true
        image0 = torch.autograd.Variable(image0, requires_grad=True)
        image1 = torch.autograd.Variable(image1, requires_grad=True)

        # Compute loss
        if self.__disparity:
            loss = model.compute_loss(image0, image1, ground_truth)
        else:
            loss = compute_flow_loss(model, image0, image1, ground_truth, self.__args)

        if self.__targeted:
            loss *= -1

        loss.backward(retain_graph=True)

        # Compute perturbations based on fast gradient sign method
        if self.__perturb_mode == "both":
            noise0_output = self.__output_norm * torch.sign(image0.grad.data)
            noise1_output = self.__output_norm * torch.sign(image1.grad.data)

        elif self.__perturb_mode == "left":
            noise0_output = self.__output_norm * torch.sign(image0.grad.data)
            noise1_output = torch.zeros_like(image1)

        elif self.__perturb_mode == "right":
            noise0_output = torch.zeros_like(image0)
            noise1_output = self.__output_norm * torch.sign(image1.grad.data)

        else:
            raise ValueError("Invalid perturbation mode: %s" % self.__perturb_mode)

        return noise0_output, noise1_output

    def __ifgsm(self, model, image0, image1, ground_truth):
        """
        Computes adversarial perturbations using iterative fast gradient sign method

        Args:
            stereo_model : object
                stereo network
            image0 : tensor
                N x C x H x W RGB image
            image1 : tensor
                N x C x H x W RGB image
            ground_truth : tensor
                N x 1 x H x W ground truth disparity

        Returns:
            tensor : uniform noise/perturbations for left image
            tensor : uniform noise/perturbations for right image
        """

        image0_output = image0.clone()
        image1_output = image1.clone()

        if self.__show_perturbation_evolution:
            noise0_outputs = []
            noise1_outputs = []

        for step in range(self.__n_step):

            # Set gradients for image to be true
            image0_output = torch.autograd.Variable(image0_output, requires_grad=True)
            image1_output = torch.autograd.Variable(image1_output, requires_grad=True)

            # Compute loss, input diversity is only used if probability greater than 0
            if self.__disparity:
                loss = model.compute_loss(
                    *self.__diverse_input(image0_output, image1_output, ground_truth)
                )
            else:
                img0_out, img1_out, gt = self.__diverse_input(
                    image0_output, image1_output, ground_truth
                )
                loss = compute_flow_loss(
                    model,
                    img0_out,
                    img1_out,
                    gt,
                    self.__args,
                )

            if self.__targeted:
                loss *= -1

            loss.backward(retain_graph=True)

            # Compute perturbations using fast gradient (sign) method
            if "ifgsm" in self.__perturb_method:
                image0_grad = torch.sign(image0_output.grad.data)
                image1_grad = (
                    torch.zeros_like(image1)
                    if image1_output.grad is None
                    else torch.sign(image1_output.grad.data)
                )
            elif self.__perturb_method == "ifgm":
                image0_grad = image0_output.grad.data
                image1_grad = image1_output.grad.data
            else:
                raise NotImplementedError

            if self.__perturb_mode == "both":
                noise0_output = self.__learning_rate * image0_grad
                noise1_output = self.__learning_rate * image1_grad

            elif self.__perturb_mode == "left":
                noise0_output = self.__learning_rate * image0_grad
                noise1_output = torch.zeros_like(image1)

            elif self.__perturb_mode == "right":
                noise0_output = torch.zeros_like(image0)
                noise1_output = self.__learning_rate * image1_grad

            else:
                raise ValueError("Invalid perturbation mode: %s" % self.__perturb_mode)

            # Add to image and clamp between supports of image intensity to get perturbed image
            image0_output = torch.clamp(image0_output + noise0_output, 0.0, 1.0)
            image1_output = torch.clamp(image1_output + noise1_output, 0.0, 1.0)

            # Output perturbations are the difference between original and perturbed image
            noise0_output = torch.clamp(
                image0_output - image0, -self.__output_norm, self.__output_norm
            )
            noise1_output = torch.clamp(
                image1_output - image1, -self.__output_norm, self.__output_norm
            )

            # Add perturbations to images
            image0_output = image0 + noise0_output
            image1_output = image1 + noise1_output

            if self.__print_out:
                sys.stdout.write(
                    "Step={:3}/{:3}  Model Loss={:.10f}\r".format(
                        step, self.__n_step, loss.item()
                    )
                )
                sys.stdout.flush()

            if self.__show_perturbation_evolution:
                cnoise0_output = (
                    noise0_output.detach().cpu().numpy()[0, ...].transpose((1, 2, 0))
                )
                noise0_outputs.append(
                    np.uint8(
                        (cnoise0_output - cnoise0_output.min())
                        / (cnoise0_output.max() - cnoise0_output.min())
                        * 255.0
                    )
                )
                cnoise1_output = (
                    noise1_output.detach().cpu().numpy()[0, ...].transpose((1, 2, 0))
                )
                noise1_outputs.append(
                    np.uint8(
                        (cnoise1_output - cnoise1_output.min())
                        / (cnoise1_output.max() - cnoise1_output.min())
                        * 255.0
                    )
                )

        if self.__show_perturbation_evolution:
            ctr = 0
            while os.path.isfile(self.__show_perturbation_evolution / f"viz{ctr}_0.gif"):
                ctr += 1
            imageio.mimsave(
                self.__show_perturbation_evolution / f"viz{ctr}_0.gif",
                noise0_outputs,
                duration=20 / self.__n_step,
            )
            imageio.mimsave(
                self.__show_perturbation_evolution / f"viz{ctr}_1.gif",
                noise1_outputs,
                duration=20 / self.__n_step,
            )

        return noise0_output, noise1_output

    def __mifgsm(self, model, image0, image1, ground_truth):
        """
        Computes adversarial perturbations using momentum iterative fast gradient sign method

        Args:
            stereo_model : object
                stereo network
            image0 : tensor
                N x C x H x W RGB image
            image1 : tensor
                N x C x H x W RGB image
            ground_truth : tensor
                N x 1 x H x W ground truth disparity

        Returns:
            tensor : uniform noise/perturbations for left image
            tensor : uniform noise/perturbations for right image
        """

        image0_output = image0.clone()
        image1_output = image1.clone()

        grad0 = torch.zeros_like(image0)
        grad1 = torch.zeros_like(image1)

        if self.__show_perturbation_evolution:
            noise0_outputs = []
            noise1_outputs = []

        for step in range(self.__n_step):

            # Set gradients for image to be true
            image0_output = torch.autograd.Variable(image0_output, requires_grad=True)
            image1_output = torch.autograd.Variable(image1_output, requires_grad=True)

            # Compute loss, input diversity is only used if probability greater than 0
            if self.__disparity:
                loss = model.compute_loss(
                    *self.__diverse_input(image0_output, image1_output, ground_truth)
                )
            else:
                img0_out, img1_out, gt = self.__diverse_input(
                    image0_output, image1_output, ground_truth
                )
                loss = compute_flow_loss(model, img0_out, img1_out, gt, self.__args)

            if self.__targeted:
                loss *= -1

            loss.backward(retain_graph=True)

            # Compute gradients with momentum
            grad0 = self.__momentum * grad0 + (
                1.0 - self.__momentum
            ) * image0_output.grad.data / torch.sum(torch.abs(image0_output.grad.data))
            grad1 = self.__momentum * grad1 + (
                1.0 - self.__momentum
            ) * image1_output.grad.data / torch.sum(torch.abs(image1_output.grad.data))

            # Compute perturbations using fast gradient sign method
            if self.__perturb_mode == "both":
                noise0_output = self.__learning_rate * torch.sign(grad0)
                noise1_output = self.__learning_rate * torch.sign(grad1)

            elif self.__perturb_mode == "left":
                noise0_output = self.__learning_rate * torch.sign(grad0)
                noise1_output = torch.zeros_like(image1)

            elif self.__perturb_mode == "right":
                noise0_output = torch.zeros_like(image0)
                noise1_output = self.__learning_rate * torch.sign(grad1)

            else:
                raise ValueError("Invalid perturbation mode: %s" % self.__perturb_mode)

            # Add to image and clamp between supports of image intensity to get perturbed image
            image0_output = torch.clamp(image0_output + noise0_output, 0.0, 1.0)
            image1_output = torch.clamp(image1_output + noise1_output, 0.0, 1.0)

            # Output perturbations are the difference between original and perturbed image
            noise0_output = torch.clamp(
                image0_output - image0, -self.__output_norm, self.__output_norm
            )
            noise1_output = torch.clamp(
                image1_output - image1, -self.__output_norm, self.__output_norm
            )

            # Add perturbations to images
            image0_output = image0 + noise0_output
            image1_output = image1 + noise1_output

            if self.__print_out:
                sys.stdout.write(
                    "Step={:3}/{:3}  Model Loss={:.10f}\r".format(
                        step, self.__n_step, loss.item()
                    )
                )
                sys.stdout.flush()

            if self.__show_perturbation_evolution:
                cnoise0_output = (
                    noise0_output.detach().cpu().numpy()[0, ...].transpose((1, 2, 0))
                )
                noise0_outputs.append(
                    np.uint8(
                        (cnoise0_output - cnoise0_output.min())
                        / (cnoise0_output.max() - cnoise0_output.min())
                        * 255.0
                    )
                )
                cnoise1_output = (
                    noise1_output.detach().cpu().numpy()[0, ...].transpose((1, 2, 0))
                )
                noise1_outputs.append(
                    np.uint8(
                        (cnoise1_output - cnoise1_output.min())
                        / (cnoise1_output.max() - cnoise1_output.min())
                        * 255.0
                    )
                )

        if self.__show_perturbation_evolution:
            ctr = 0
            while os.path.isfile(self.__show_perturbation_evolution / f"viz{ctr}_0.gif"):
                ctr += 1
            imageio.mimsave(
                self.__show_perturbation_evolution / f"viz{ctr}_0.gif",
                noise0_outputs,
                duration=20 / self.__n_step,
            )
            imageio.mimsave(
                self.__show_perturbation_evolution / f"viz{ctr}_1.gif",
                noise1_outputs,
                duration=20 / self.__n_step,
            )

        return noise0_output, noise1_output

    def __diverse_input(self, image0, image1, ground_truth):

        # If p greater than probability of input diversity
        if torch.rand(1) > self.__probability_diverse_input:
            return image0, image1, ground_truth

        assert image0.shape == image1.shape

        # Compute padding on each side
        _, _, o_height, o_width = image0.shape

        n_height = random.randint(int(o_height - o_height / 10.0), o_height)
        n_width = random.randint(int(o_width - o_width / 10.0), o_width)

        top_pad = random.randint(0, o_height - n_height)
        bottom_pad = o_height - n_height - top_pad

        left_pad = random.randint(0, o_width - n_width)
        right_pad = o_width - n_width - left_pad

        # Resize images to new size and pad with zeros to get original size
        image0_resized = torch.nn.functional.interpolate(
            image0, size=(n_height, n_width), mode="bilinear"
        )

        image0_output = torch.nn.functional.pad(
            image0_resized,
            pad=(left_pad, right_pad, top_pad, bottom_pad),
            mode="constant",
            value=0,
        )

        image1_resized = torch.nn.functional.interpolate(
            image1, size=(n_height, n_width), mode="bilinear"
        )

        image1_output = torch.nn.functional.pad(
            image1_resized,
            pad=(left_pad, right_pad, top_pad, bottom_pad),
            mode="constant",
            value=0,
        )

        # Resize ground truth and pad with zeros
        ground_truth_resized = torch.nn.functional.interpolate(
            ground_truth, size=(n_height, n_width), mode="nearest"
        )

        ground_truth_output = torch.nn.functional.pad(
            ground_truth_resized,
            pad=(left_pad, right_pad, top_pad, bottom_pad),
            mode="constant",
            value=0,
        )

        # Scale disparity to adjust for change in size
        ground_truth_output *= float(n_width) / float(o_width)

        assert image0.shape == image0_output.shape
        assert image1.shape == image1_output.shape
        assert ground_truth.shape == ground_truth_output.shape

        return image0_output, image1_output, ground_truth_output
