conda create -n doceb_py39 python=3.9 -y
conda activate doceb_py39
pip install -r requirements_py39.txt
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

source ~/.bashrc
conda activate base
conda activate doceb_py39