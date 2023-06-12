echo "PATH=$PATH:~/.local/bin" | tee -a ~/.bash_profile; . ~/.bash_profile
sudo apt install tmux fuse libfuse2 -y

git config --global user.name "reign12"
git config --global user.email "v-jingchu@microsoft.com"
git config --global core.editor "vim"


git config --global credential.helper store
pip install -U pip
pip install git+https://reign12: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@github.com/REIGN12/SimMIMScripts.git@main
mimserver download-setup; mimserver dev-setup

pip list
git config --global credential.helper store
cd
git clone https://github.com/REIGN12/rlhf_trlx.git trlx
cd trlx
pip install -e .
pip install -e .
pip install "transformers==4.28.1"
pip install "tokenizers>=0.13.3"
pip install sentencepiece
pip list