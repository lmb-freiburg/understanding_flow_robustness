# Understanding Adversarial Flow Robustness

PyTorch implementation of [*Towards Understanding Adversarial Robustness of Optical Flow Networks*](https://arxiv.org/abs/2103.16255).

If this work is useful to you, please consider citing our paper:

```
@inproceedings{schrodi2022towards,
  title={Towards Understanding Adversarial Robustness of Optical Flow Networks},
  author={Schrodi, Simon and Saikia, Tonmoy and Brox, Thomas},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Prerequisites

To setup your Python environment follow the steps below:

1. Clone this repository.

1. `cd` into this repository.

1. Create a conda environment:

```bash
conda create --name flow_rob --file conda_requirements.txt
```

or if your system uses GCC-7 as standard compiler, you can just run

```bash
conda create --name flow_rob python=3.8
```

4. Activate the conda environment `conda activate flow_rob`.

1. Run `conda develop "${pwd}"` (if you are in the repository) or `conda develop /path/to/project/understanding_flow_robustness`.

1. Install PyTorch 1.4.0 (Poetry cannot properly deal with that). Be aware that
   the CUDA version and GCC version need to match; see [here](https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version)
   for more information.

```bash
bash setup/install_torch.sh
```

8. Install poetry:

```bash
bash setup/install_poetry.sh
```

or follow the installation instructions [here](https://python-poetry.org/docs/).

9. Run `poetry install`.

1. Install custom cuda layers (requires CUDA runtime!):

```bash
bash setup/install_flownet2_deps.sh
```

11. Install Spatial correlation sampler:

```bash
pip install spatial-correlation-sampler
```

If you have problems installing it, please refer to [here](https://github.com/ClementPinard/Pytorch-Correlation-extension/issues). Note that we need to re-compile this package if run on different GPU_CCs. There might be soon a better solution, see [here](https://github.com/ClementPinard/Pytorch-Correlation-extension/issues/87). In the meantime, you need to run the following command everytime (unless you use the same GPU_CC):

```bash
pip uninstall spatial-correlation-sampler -y
cd models/Pytorch-Correlation-extension
rm -rf *_cuda.egg-info build dist __pycache__
python setup.py install
cd ../..
```

12. (Optional) Run `pre-commit install`.

### Preparing the data

0. By default, we assume the following folder structure:

```Shell
├── datasets
    ├── KITTI
        ├── 2012_prepared
            ├── 2011_09_26_drive_0001_sync_02
            ├── 2011_09_26_drive_0001_sync_03
            ├── ...
            ├── train.txt
            ├── val.txt
        ├── 2012
            ├── test
            ├── training
        ├── 2015
            ├── test
            ├── training
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
    ├── Sintel
        ├── test
        ├── training
```

1. Download the following datasets

   1. The raw [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) dataset with this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) from the official website,
   1. [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) (stereo flow),
   1. [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) (scene 4low),
   1. [Flying Chairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs),
   1. [Flying Things 3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), and
   1. [Sintel](http://sintel.is.tue.mpg.de/).

1. Run

```bash
python3 -m setup.prepare_train_data.py /path/to/KITTI/raw --dataset-format 'kitti' --dump-root /path/to/KITTI/2012_prepared --width 1280 --height 384 --num-threads 1 --with-gt
```

3. Create symbolic links (if not saved there already):

```
mkdir datasets
mkdir datasets/KITTI
ln -s /path/to/KITTI/2012_prepared datasets/KITTI/2012_prepared
ln -s /path/to/KITTI/2012/ datasets/KITTI/2012
ln -s /path/to/KITTI/2015/ datasets/KITTI/2015
ln -s /path/to/FlyingChairs_release/ datasets/FlyingChairs_release
ln -s /path/to/FlyingThings3D/ datasets/FlyingThings3D
ln -s /path/to/Sintel/ datasets/Sintel
```

4. Run

```bash
python3 setup/setup_dataset_kitti.py
```

### Pretrained models

0. We assume the following folder structure:

```Shell
├── pretrained_models
    ├── spynet_models
    ├── adv_kitti2012_pwcnet_ifgsm_l2_0.02.pth
    ├── adv_kitti2012_raft_ifgsm_l2_0.02.pth
    ├── adv_kitti2012_robustFlow_ifgsm_l2_0.02.pth
    ├── FlowNet2_checkpoint.pth.tar
    ├── FlowNet2-C_checkpoint.pth.tar
    ├── FlowNet2-S_checkpoint.pth.tar
    ├── larger_field_3x3_x0_l2.pth
    ├── pwc_net_chairs.pth.tar
    ├── raft_flowNetCEnc_noSeparateContext.pth
    ├── raft-things.pth
    ├── RobustFlowNetC.pth
```

1. Download the pretrained models for [FlowNetC](https://drive.google.com/file/d/1BFT6b7KgKJC8rA59RmOVAXRM_S7aSfKE/view), [FlowNetS](https://drive.google.com/file/d/1V61dZjFomwlynwlYklJHC-TLfdFom3Lg/view), [FlowNet2](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view), [PWC-Net](https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/pwc_net_chairs.pth.tar), [RAFT](https://github.com/princeton-vl/RAFT/blob/master/download_models.sh), [Robust FlowNetC](https://lmb.informatik.uni-freiburg.de/resources/binaries/cvpr22_adv_flow/RobustFlowNetC.pth), [FlowNetC trained with our training schedule](https://lmb.informatik.uni-freiburg.de/resources/binaries/cvpr22_adv_flow/larger_field_3x3_x0_l2.pth), [adv. trained Robust FlowNetC](https://lmb.informatik.uni-freiburg.de/resources/binaries/cvpr22_adv_flow/adv_kitti2012_robustFlow_ifgsm_l2_0.02.pth), [RAFT with FlowNetC encoder and no separate context encoder](https://lmb.informatik.uni-freiburg.de/resources/binaries/cvpr22_adv_flow/raft_flowNetCEnc_noSeparateContext.pth), [adv. trained PWC-Net](https://lmb.informatik.uni-freiburg.de/resources/binaries/cvpr22_adv_flow/adv_kitti2012_pwcnet_ifgsm_l2_0.02.pth), and [adv. trained RAFT](https://lmb.informatik.uni-freiburg.de/resources/binaries/cvpr22_adv_flow/adv_kitti2012_raft_ifgsm_l2_0.02.pth). Note that you have to transform the RAFT models to be compatible with PyTorch 1.4; see [here](https://github.com/pytorch/pytorch/issues/48915) for more information. For SPyNet copy the weights from [here](https://github.com/anuragranj/flowattack/tree/master/models/spynet_models) to `pretrained_models/spynet_models`.

For convenience, you can run `bash setup/download_weights` to download all model weights.

1. Create symbolic links (if not saved there already):

```
mkdir pretrained_models
ln -s /path/to/FlowNet2-C_checkpoint.pth.tar pretrained_models/FlowNet2-C_checkpoint.pth.tar
ln -s /path/to/FlowNet2-S_checkpoint.pth.tar pretrained_models/FlowNet2-S_checkpoint.pth.tar
ln -s /path/to/FlowNet2_checkpoint.pth.tar pretrained_models/FlowNet2_checkpoint.pth.tar
ln -s /path/to/pwc_net_chairs.pth.tar pretrained_models/pwc_net_chairs.pth.tar
ln -s /path/to/raft-things.pth pretrained_models/raft-things.pth
```

## Patch-based Adversarial Attacks on Optical Flow

### Generating Adversarial Patches

To generate adversarial patches, run

```bash
python3 patch_attacks/main.py \
    --name $path \
    --valset kitti2012 \
    --workers 1 \
    --flownet $flownet \
    --lr 1000 \
    --epochs 40 \
    --patch-size 0.1329 \
    --seed 42
```

### Analysis & Evaluation

To test the patch across various locations and find the worst one (Section 4.1), run

```bash
python3 patch_attacks/test_moving_patch.py \
    --name $path \
    --instance $instance \
    --patch_name $name \
    --flownet $flownet \
    --stride 25 \
    --norotate
```

To reproduce our experiments in Table 2 (requires previous generation of an adversarial patch), run

```bash
python3 patch_attacks/test_replace_features.py \
    --name $path \
    --feature_keys corr conv_redir \
    --flownet FlowNetC \
    --instance $instance \
    --patch_name $patch_name
```

```bash
python3 patch_attacks/test_replace_features.py \
    --name $path \
    --feature_keys corr \
    --flownet FlowNetC \
    --instance $instance \
    --patch_name $patch_name
```

```bash
python3 patch_attacks/test_replace_features.py \
    --name $path \
    --feature_keys conv_redir \
    --flownet FlowNetC \
    --instance $instance \
    --patch_name $patch_name
```

To create the t-SNE embeddings in Figure 3 and 9, run

```bash
python3 patch_attacks/test_patch_embeddings.py \
    --name $path \
    --instance $instance \
    --patch_name $patch_name \
    --flownet $flownet
```

### Handcrafted Patch Attacks

To attack without the need of any optimization (Section 5), run

```bash
python3 patch_attacks/test_moving_patch.py \
    --name $path \
    --self_correlated_patch vstripes \
    --patch_size $patch_size \
    --flownet $flownet \
    --stride 25 \
    --norotate
```

You can run the other scripts for analysis and evaluation accordingly.

## Global Adversarial Attacks on Optical Flow

### Attacking & Evaluation

To reproduce the experiments for untargeted global white-box attacks (Supplemental Section H), run

```bash
python3 global_attacks/run_perturb_model.py \
    --output_path $output_path \
    --dataset kitti2015 \
    --n_height 256 \
    --n_width 640 \
    --output_norm $output_norm \
    --learning_rate $lr \
    --n_step $n_step \
    --perturb_method $perturb_method \
    --flow_loss $flow_loss
--flownet $flownet \
    --write_out \
```

To reproduce the results for the targeted 42 attack (Figure 11), run

```bash
python3 global_attacks/run_perturb_model.py \
    --output_path $output_path \
    --dataset kitti2015 \
    --n_height 256 \
    --n_width 640 \
    --output_norm $output_norm \
    --learning_rate $lr \
    --perturb_method $perturb_method \
    --flownet $flownet \
    --write_out \
    --targeted \
    --arbitrary_gt_index fun
```

or to attack with an arbitrary GT from KITTI 2015 (Supplemental Section I), run

```bash
python3 global_attacks/run_perturb_model.py \
    --output_path /misc/lmbraid19/schrodi/stereopagnosia \
    --dataset kitti2015 \
    --n_height 256 \
    --n_width 640 \
    --output_norm $output_norm \
    --learning_rate $lr \
    --n_step $n_step \
    --perturb_method $perturb_method \
    --flownet $flownet \
    --seed 0 \
    --write_out \
    --flow_loss $flow_loss \
    --targeted
--arbitrary_gt_index $arbitrary_index \
```

where `arbitrary_index` can be between 0 and 199.

To generate universal adversarial perturbation (Section 7), run

```bash
python3 global_attacks/run_perturb_model.py \
    --output_path $output_path \
    --dataset kitti2015 \
    --n_height 256 \
    --n_width 640 \
    --perturb_method $perturb_method \
    --flownet $flownet \
    --perturb_mode same \
    --write_out
```

and then to attack, run

```bash
python3 $WORKDIR/global_attacks/run_perturb_model.py \
    --output_path $output_path \
    --dataset kitti2015 \
    --flownet $flownet \
    --universal_evaluation \
    --output_norm $output_norm \
    --perturb_method $perturb_method \
    --flow_loss $flow_loss \
    --folder_name $folder_name \
    --epoch_number $epoch_number \
    --n_height 256 \
    --n_width 640 \
    --write_out
```

To apply common image corruptions and test flow nets, run

```bash
python3 global_attacks/run_perturb_model.py \
    --output_path $output_path \
    --dataset kitti2015 \
    --n_height 384 \
    --n_width 1280 \
    --perturb_method $perturb_method \
    --flownet $flownet \
    --perturb_mode same \
    --write_out
```

### Adversarial Training

To train adversarially robust flow nets (Supplemental Section K), run:

```bash
python training/train.py --name $name \
    --restore_ckpt pretrained_models/raft-things.pth \
    --adv_train \
    --stage kitti2012 \
    --ckpt_dir $ckpt_dir \
    --gpus 0 1 \
    --num_steps $num_steps \
    --batch_size 1 \
    --lr 0.000125 \
    --image_size 256 640 \
    --wdecay 0.0001 \
    --flownet $flownet \
    --perturb_method $perturb_method \
    --output_norm $output_norm \
    --perturb_learning_rate $perturb_learning_rate \
    --perturb_n_step $perturb_n_step \
    --flow_loss $flow_loss
```

## Acknowledgements

We thank several GitHub users for their contributions which are used in this repository:

- Correlation module from [ClementPinard/Pytorch-Correlation-extension](https://github.com/ClementPinard/Pytorch-Correlation-extension).
- The basis for patch-based attacks is taken from [anuragranj/flowattack](https://github.com/anuragranj/flowattack/).
- The basis for global attacks is taken from [alexklwong/stereopagnosia](https://github.com/alexklwong/stereopagnosia).
- The basis for training of flow networks is taken from [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT).
