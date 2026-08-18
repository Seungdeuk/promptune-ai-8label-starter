$ErrorActionPreference = "Stop"

python --version
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "from app.core.labels import LABELS, NUM_LABELS; print('Labels:', LABELS); print('num_labels:', NUM_LABELS)"
