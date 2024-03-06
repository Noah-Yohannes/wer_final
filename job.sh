#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=test_job
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account= kuin0027
#SBATCH --output=WER.%j.out
#SBATCH --error=WER.%j.err 


module load miniconda/3 -q
module load cuda/11.7 -q
module load utilities/1.0

pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html -q
pip3 install transformers -q
pip3 install datasets -q
pip3 install evaluate -q
pip3 install jiwer -q
pip3 install librosa -q
pip3 install --upgrade tensorflow -q

conda create --wer_rate check python==3.8 -y -q

conda activate wer_rate
python  WER.py
conda deactivate
