# Antigravity Developer Environment for Loki

## Project Overview
Loki is a novel sparse attention method that ranks and selects tokens in the KV-cache based on attention scores computed in a low-dimensional space. It reduces compute and memory costs for large language models (LLMs) with long sequence lengths. 
**Paper Reference**: [Loki: Low-Rank Keys for Efficient Sparse Attention (2406.02542)](https://arxiv.org/abs/2406.02542)

## Environment & Dependencies
The environment has been adapted to run on a newer `compass` conda environment.
- **Python Framework**: PyTorch
- **Dependencies** (Adapted versions):
  - `torch==2.9.1+cu128`
  - `transformers==4.56.0`
  - `accelerate==1.12.0`
  - `datasets==4.4.2`
  - `wandb==0.23.1`

**Environment Configuration**:
To set up and configure the `compass` environment with all necessary dependencies, use the provided setup script:
```bash
bash setup_env.sh
```

To activate the environment after initial setup:
```bash
conda activate compass
```

## Repository Structure
- `evaluate_tasks.py`: Script to generate keys/queries/values tensors and run evaluations on downstream tasks (e.g., LM Harness).
- `evaluate_compute.py`: Script to run attention benchmarks (Loki vs Vanilla attention).
- `generate_samples.py`: Script to generate samples from the model.
- `infer.py`: General inference script.
- `pca_analysis/`: Contains `pca.py` to compute the PCA of the generated key tensors.
- `methods/`: Contains the core implementation of the Loki sparse attention mechanisms and baselines.
- `helper/`: Helper functions.

## Main Workflows

### 1. Compute PCA of Keys
**Step 1:** Run perplexity evaluation to save tensors:
```bash
python -u evaluate_tasks.py --sequence-length 4096 --model-id meta-llama/Llama-2-7b-hf --model-type llama --dataset wikitext-valid --save-tensors --tensor-dir <TENSOR_DIR> --use-topk --top-k 1
```
**Step 2:** Compute the PCA in the `pca_analysis` directory:
```bash
cd pca_analysis
python pca.py key <NUM_LAYERS> <TENSOR_DIR> <PCA_OUTPUT_DIR>
```
*Note: Ensure the subdirectory structure in `<PCA_OUTPUT_DIR>` remains unchanged as downstream tasks depend on it.*

### 2. Run ML Evaluations (Downstream Tasks)
Evaluate the performance on downstream tasks using the computed PCA transforms:
```bash
python -u evaluate_tasks.py \
  --sequence-length 4096 \
  --model-id meta-llama/Llama-2-7b-hf \
  --model-type llama \
  --use-pca-topk \
  --top-r 16 \
  --top-k 0.25 \
  --rotary-type prerotary \
  --dataset wikitext-test \
  --transform-dataset wikitext
```

### 3. Run Compute Benchmarks
Benchmark attention computation time/memory (Loki vs Vanilla):
```bash
python evaluate_compute.py
```

## Agent Development Guidelines
1. **Dependency Adaption**: The codebase has been modified to support `transformers==4.56.0` and `torch==2.9.1`. Be careful with forward passes of Attention mechanisms like `LlamaAttention` which may utilize new caching objects.
2. **Context Path**: Default working directory should be the repository root `/home/zijie/Code/loki/`. Always use absolute paths in terminal interactions.
3. **Execution & Hardware Issues**: 
   - If encountering CUDA OOM out-of-memory errors, you may use the `--use-axonn` flag for tensor parallelism, which must be launched via `mpirun` (e.g., `mpirun -np 2 python -u evaluate_tasks.py ...`).
   - Monitor GPU usage to ensure the environment is optimal for running heavy LLM tasks.
4. **Code Modification**: When modifying attention kernels or injection points inside `methods/`, maintain backward compatibility with shape dimensions, particularly respecting the top-k selection logic over the KV sequence length dimension.
