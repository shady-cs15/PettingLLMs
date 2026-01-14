# setup pyenv
sudo apt update
sudo apt install -y make build-essential \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev \
  libxmlsec1-dev libffi-dev liblzma-dev
curl https://pyenv.run | bash

# install Python 3.12
sleep 10
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
sleep 5
pyenv install 3.12.2
pyenv local 3.12.2

python --version
sudo ln -s "$(pyenv which python)" /usr/bin/python3.12

# install pip dependencies
python -m pip install -U "opentelemetry-sdk~=1.39.1"
python -m pip install -U "tensordict<=0.6.2"
python -m pip install datasets
python -m pip install hydra-core
python -m pip install "ray[default]"
python -m pip install cachetools
python -m pip install fastapi
python -m pip install uvicorn
python -m pip install openai
python -m pip install psutil
python -m pip install flash-attn --no-build-isolation

# install verl right version
ls -la verl
ls -la verl/verl/single_controller 2>/dev/null || true
python -m pip uninstall -y verl
python -m pip install -e ./verl

# setup wandb
export WANDB_USERNAME=satyaki
export WANDB_ENTITY=autonomy
export WANDB_API_KEY=local-75a521c35fa8d43fdc8abdad6f56814809f16f82
export WANDB_BASE_URL=https://wandb.agi.amazon.dev/