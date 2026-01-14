#!/bin/bash -l
#SBATCH -J bbbp_ft_fl 
#SBATCH -o out_bbbp_ft_fl.txt
#SBATCH -e out_bbbp_ft_fl.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#SBATCH -A A100
#SBATCH -q a100_qos

module load cuda12.6/toolkit/12.6.2
python --version
nvidia-smi
nvcc --version

export MAMBA_EXE='/export/home/rmall/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/export/home/rmall/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from micromamba activate
fi
unset __mamba_setup
# <<< mamba initialize <<<

micromamba env list
micromamba activate effichem

python3 finetune_bbbp_fl.py
