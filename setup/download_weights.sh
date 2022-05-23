#!/bin/bash

mkdir pretrained_models
cd pretrained_models

URL_BASE="https://lmb.informatik.uni-freiburg.de/resources/binaries/cvpr22_adv_flow"

download () {
	wget --no-check-certificate "$URL_BASE/$1.pth"
}

# Robust FlowNetC
download RobustFlowNetC

# FlowNetC using our training schedule
download larger_field_3x3_x0_l2

# RAFT with FlowNetC encoder and no context encoder
download raft_flowNetCEnc_noSeparateContext

# Robust FlowNetC adv. trained with I-FGSM, L2-loss and norm 0.02
download adv_kitti2012_robustFlow_ifgsm_l2_0.02

# PWC-Net adv. trained with I-FGSM, L2-loss and norm 0.02
download adv_kitti2012_pwcnet_ifgsm_l2_0.02

# RAFT adv. trained with I-FGSM, L2-loss and norm 0.02
download adv_kitti2012_raft_ifgsm_l2_0.02

# FlowNetC
gdown https://drive.google.com/u/0/uc?id=1BFT6b7KgKJC8rA59RmOVAXRM_S7aSfKE

# FlowNetS
gdown https://drive.google.com/u/0/uc?id=1V61dZjFomwlynwlYklJHC-TLfdFom3Lg

# FlowNet2
gdown https://drive.google.com/u/0/uc?id=1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da

# PWC-Net
wget https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/pwc_net_chairs.pth.tar?raw=true
mv pwc_net_chairs.pth.tar\?raw\=true pwc_net_chairs.pth.tar

# RAFT
wget --no-check-certificate https://raw.githubusercontent.com/princeton-vl/RAFT/master/download_models.sh
bash download_models.sh
mv models/* .
rmdir models
rm models.zip
rm download_models.sh

# SPyNet
git clone https://github.com/anuragranj/flowattack.git
mv flowattack/models/spynet_models/ .
rm -rf flowattack/

cd ..
