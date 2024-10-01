For MAC users
1. python -m venv VirtualEnv
2. pip install matplotlib numpy pylzma ipykernel jupyter
3. pip3 install torch torchvision torchaudio
4. Create the kernel: python -m ipykernel install --user --name=gpu_kernel --display-name "gpu kernel"

xcode-select --install if you get a ERROR: Failed building wheel for cffi while installing libraries

Inside the notebook:
device = 'mps' if torch.backends.mps.is_available() else 'cpu'