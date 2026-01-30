# How to infer from the LLM
Take note that the .sif image (apptainer image) must be available as it is tailored to the runtime of the LLM inference and training

## Run inference
```bash
srun --partition=gpu --gres=gpu:a100_40gb:1 --mem=64G --time=1:00:00 --pty \
apptainer exec --nv --bind $PWD$:$PWD \
$PWD/container.sif \
python $PWD/inference.py
```
Note that the user will run an interactive terminal session that will ask them for text input. Submit the dialoge as a one-liner. Result will be printed out in the terminal.

## Run benchmark
```bash
sbatch ./run_benchmark_bf16.slurm
```
