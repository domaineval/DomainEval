conda create -n doceb_py39 python=3.9 -y
conda activate doceb_py39
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements_py39.txt

source ~/.bashrc
conda activate base
conda activate doceb_py39