### `gradio-client`

Copy file from `gradio-fix/utils.py` to location of `utils.py` in container.

### `nccl`

To install on `ubuntu` based images: 

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update && sudo apt install libnccl2 libnccl-dev -y
```

Refer [this](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)