See https://github.com/karpathy/nanochat/discussions/28

```
# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate
```

```
# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

```
python -m nanochat.dataset -n 240
```

```
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval
```

```
curl -L -o eval_bundle.zip https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
unzip -q eval_bundle.zip
rm eval_bundle.zip
mv eval_bundle "$HOME/.cache/nanochat"
```
Install CUDA 13.0.2

The next step in the nanochat instructions would be to now run pre-training, but that step will fail, because the default ptxas installed with Triton 3.5.0 is the CUDA 12.8 version and doesn't know about the sm_121a gpu-name of the Blackwell GB10.

At this time, you need to go to the nVIDIA Developer website and install CUDA 13.0.2 manually by following the steps here: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=arm64-sbsa&Compilation=Native&Distribution=Ubuntu&target_version=24.04&target_type=deb_local

In particular, this was the sequence that worked for me on the DGX Spark:

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_arm64.deb
sudo dpkg -i cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_arm64.deb
sudo cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
```

And now you need to tell Triton to use the new ptxas version you just installed with the CUDA 13.0.2 toolkit:

```
# assuming CUDA 13.0 is installed at /usr/local/cuda-13.0
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}
```

Run pre-training

Now you should be able to run pre-training on your GDX Spark with the usual command from the nanochat instructions:

```
torchrun --standalone --nproc_per_node=gpu -m scripts.base_train -- --depth=20
```


Is this with the default --device_batch_size=32?

