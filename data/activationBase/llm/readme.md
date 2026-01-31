# ActivationBase – AI Project Jelli

## Ownership
Author: Julia Jellimek and Keno Hanken
This Docker image is owned and maintained by the author named above.

## Course Context
This Docker image was created as part of the course  
**“M. Grum: Advanced AI-based Application Systems”**  
offered by the Junior Chair for Business Information Science,  
especially AI-based Application Systems,  
at the University of Potsdam.

## AI Model Characterization
This image provides the activation and inference environment for a fine-tuned
Large Language Model (LLM). It includes scripts for model inference as well as
benchmarking functionality to evaluate model performance on domain-specific
classification tasks.

The container is designed to execute inference workflows by default and supports
optional benchmarking executions.

## License
This project is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
By using this image, you agree to comply with the terms and conditions of this license.


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
