# Environment

Download the [model](https://drive.google.com/drive/folders/1Nq3VXclKsQNYKmQTuXEsjSh143Q9UvPT?usp=drive_link) folder and put it into `WildCap/model`. Similar to the [DoRA repo](https://github.com/yxuhan/DoRA), this model is trained on the [GauFace dataset](https://dafei-qin.github.io/TransGS.github.io/).

```
conda create -n wildcap python=3.10
conda activate wildcap

# install pytorch
pip install https://download.pytorch.org/whl/cu121/torch-2.3.1%2Bcu121-cp310-cp310-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu121/torchvision-0.18.1%2Bcu121-cp310-cp310-linux_x86_64.whl

# install other libs
pip install \
    kornia \
    tensorboard \
    trimesh \
    numpy==1.26.4 \
    torchdiffeq \
    torchsde \
    scipy \
    safetensors \
    jsonmerge \
    tqdm \
    scikit-image \
    dctorch \
    einops \
    clean-fid \
    clip-anytorch

# install pytorch3d
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu121_pyt231.tar.bz2

# install nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```
