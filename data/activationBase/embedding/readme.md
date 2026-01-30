# Embedding how-to use
Take note that the .sif image (apptainer image) must be available as it is tailored to the runtime of the embedding inference and training

## How to infer from the embedding
```bash
srun --partition=gpu --gres=gpu:a100_40gb:1 --mem=64G --time=1:00:00 --pty \
  apptainer exec --nv --bind $PWD:$PWD $PWD/embedding_finetune.sif \
  python $PWD/classify.py --model_dir ./finetuned_model \
  --text "INSERT TEXT HERE IN DOUBLE QUOTES"
```
Interactive SLURM terminal session. Returns the top 5 most likely fitting ICD codes

## How to run a benchmark for the embedding
```bash
srun --partition=gpu --gres=gpu:a100_40gb:1 --mem=64G --time=1:00:00 --pty \
  apptainer exec --nv --bind $PWD:$PWD $PWD/embedding_finetune.sif \
  python $PWD/classify.py --model_dir ./finetuned_model \
  --input ./validation_finetuning_embedding.json --evaluate \
  --output embedding_benchmark_result.json
´´´
