#!/bin/bash -l
#SBATCH -J bace_ft_wl
#SBATCH -o out_bace_ft_wl.txt
#SBATCH -e out_bace_ft_wl.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --mem=140000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -x crirdchpxd005
#SBATCH -w crirdchpxd004


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

python3 finetune_bace_wl.py
