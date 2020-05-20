# Lambda Notes

### Install on Ubuntu (with Lambda Stack 18.04)

__Anaconda__
Download and install the latest Python 3 Anaconda from: https://www.anaconda.com/download/. I prefer not to use conda for default Python. Also you should disable the base environment from appearing in the terminal.


``
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
chmod +x Anaconda3-2020.02-Linux-x86_64.sh
./Anaconda3-2020.02-Linux-x86_64.sh
conda config --set auto_activate_base false
``


```
git clone https://github.com/chuanlilambda/faceswap.git
cd faceswap
conda create -n venv-faceswap python=3.6
conda activate venv-faceswap

# Run this and select N to amd, docker, select Y to CUDA
python setup.py
```