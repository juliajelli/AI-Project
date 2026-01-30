# How to finetune the LLM

## Build Apptainer image
```bash
apptainer build container.sif container.def
```

## create conda environment
```bash
conda create --name aibasllm --file ../requirements.txt
```

## Activate conda environment
```bash
conda activate aibasllm
```

## Login to Huggingface
```bash
hf auth login --token INSERT_TOKEN
```

## Download model
```bash
./download_model.sh
```

## Launch finetuning with SLURM
```bash
sbatch ./run_training.slurm
```

## Login to wandb
```bash
wandb login
```

## Sync offline wandb with online wandb for analysis
```bash
wandb sync --sync-all wandb/
```

## Infere LLM via interactive session (check activationBase for more details / use-cases)
```bash
srun --partition=gpu --gres=gpu:a100_40gb:1 --mem=64G --time=1:00:00 --pty \
apptainer exec --nv --bind data/ \
container.sif \
python activationBase/llm/inference.py
```
