# DesignAsCode

**DesignAsCode** is a framework that generates **fully editable** graphic designs from natural-language prompts. Unlike image generation models that produce flat raster images, or layout generation methods that output abstract bounding boxes, DesignAsCode represents designs as **HTML/CSS code** — preserving both high visual fidelity and fine-grained structural editability. This code-native representation also unlocks advanced capabilities such as automatic layout retargeting, complex document generation, and CSS-based animation.

The framework uses a **Plan → Implement → Reflection** pipeline: a fine-tuned Semantic Planner constructs dynamic element hierarchies, an Implementation module translates the plan into executable HTML/CSS with generated image assets, and a Visual-Aware Reflection mechanism iteratively refines the output to fix rendering artifacts.

<p align="center">
  <a href="https://liuziyuan1109.github.io/design-as-code/"><img src="https://img.shields.io/badge/🌐-Project%20Page-green?style=for-the-badge" alt="Project Page"></a>
  <a href="https://liuziyuan1109.github.io/design-as-code/"><img src="https://img.shields.io/badge/📄-Paper-blue?style=for-the-badge" alt="Paper"></a>
  <a href="https://huggingface.co/Tony1109/DesignAsCode-planner"><img src="https://img.shields.io/badge/🤗-Model-orange?style=for-the-badge" alt="Model"></a>
  <a href="https://huggingface.co/datasets/Tony1109/DesignAsCode-training-data"><img src="https://img.shields.io/badge/🤗-Dataset-orange?style=for-the-badge" alt="Dataset"></a>
</p>

---

## Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/liuziyuan1109/design-as-code.git
cd design-as-code

conda create -n designascode python=3.10 -y
conda activate designascode
pip install -r requirements.txt
playwright install chromium
```

> **Note:** GPU inference requires CUDA 11.8+ and ≥24 GB VRAM.

### 2. Download Model & Data

```bash
# Planner model (~16 GB)
conda install git-lfs -c conda-forge -y
git lfs install
git clone https://huggingface.co/Tony1109/DesignAsCode-planner models/planner

# Image retrieval library + FAISS index (~19 GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('Tony1109/crello-image-library', repo_type='dataset', local_dir='retrieval_assets')"
cd retrieval_assets
tar -xzf crello_pngs.tar.gz   # may take a relatively long time (many small files)
mv crello_pngs ../data/image_library
mv elements_local.index ../data/
mv id_mapping_local.json ../data/
cd .. && rm -rf retrieval_assets
```

### 3. Set OpenAI API Key

The pipeline calls three OpenAI models — your key must have access to all of them:

| Model | Purpose |
|---|---|
| `gpt-5` | HTML/CSS generation and layout refinement |
| `gpt-4o` | Image quality analysis |
| `gpt-image-1` | Image generation and editing |

```bash
export OPENAI_API_KEY='sk-your-api-key-here'
```

### 4. Run Inference

```bash
python infer.py \
  --prompt "A promotional poster for International Day of Forests" \
  --output output/forest_day
```

Output is saved to the specified directory:

```
output/forest_day/
├── layout_prompt.txt         # Design plan + retrieved image URLs
├── generated_images/         # Retrieved images for each layer
├── init_result.html          # Initial HTML design
├── after_image_refine.html   # After image refinement
└── refine_*.html             # Final result after layout refinement
```

---

## Batch Evaluation

After completing the Quick Start setup above, you can run the full pipeline on the test set:

```bash
export OPENAI_API_KEY='sk-...'

cd code
python inference_pipeline.py \
  --test-data ../data/test.jsonl \
  --output-dir ../output/test_results
```

This processes all 546 samples in `test.jsonl` through the complete pipeline. For the Broad test set (500 samples), use `--test-data ../data/broad_test.jsonl`.

### Distributed / Sharded Runs

Split the workload across multiple GPUs or machines:

```bash
# Machine 0 of 4
python inference_pipeline.py --test-data ../data/test.jsonl --output-dir ../output/test_results --shard-index 0 --num-shards 4

# Machine 1 of 4
python inference_pipeline.py --test-data ../data/test.jsonl --output-dir ../output/test_results --shard-index 1 --num-shards 4
```

### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--model-path` | `../models/planner` | Path to planner model |
| `--index-path` | `../data/elements_local.index` | Path to FAISS index |
| `--id-mapping-path` | `../data/id_mapping_local.json` | Path to ID mapping |
| `--test-data` | `../data/test.jsonl` | Path to test JSONL |
| `--output-dir` | `batch_outputs` | Output directory |
| `--shard-index` | `0` | Current shard index |
| `--num-shards` | `1` | Total number of shards |

### Required API Access

The pipeline calls three OpenAI models:

| Model | Purpose |
|---|---|
| `gpt-5` | HTML/CSS generation and layout refinement |
| `gpt-4o` | Image quality analysis |
| `gpt-image-1` | Image generation and editing |

---

## Training the Planner

Download the training data (~19k distilled samples from Crello):

```bash
huggingface-cli download Tony1109/DesignAsCode-training-data --repo-type dataset --local-dir training_data
```

Train on a single GPU:

```bash
cd code
python train_planner.py \
  --model Qwen/Qwen3-8B \
  --data ../training_data/dataset.jsonl \
  --output-dir ../models/planner_ckpt
```

Multi-GPU with DeepSpeed (optional — requires `pip install deepspeed`):

```bash
deepspeed --num_gpus=4 train_planner.py \
  --model Qwen/Qwen3-8B \
  --data ../training_data/dataset.jsonl \
  --output-dir ../models/planner_ckpt \
  --deepspeed ds_config.json
```

The final model is saved to `<output-dir>/final/`.

### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen3-8B` | Base model name or path |
| `--data` | *(required)* | Path to training JSONL |
| `--output-dir` | *(required)* | Checkpoint output directory |
| `--epochs` | `2` | Number of training epochs |
| `--batch-size` | `1` | Per-device batch size |
| `--gradient-accumulation-steps` | `2` | Gradient accumulation steps |
| `--learning-rate` | `5e-5` | Learning rate |
| `--max-length` | `6144` | Maximum token length |
| `--save-steps` | `500` | Save checkpoint every N steps |
| `--deepspeed` | `None` | Path to DeepSpeed config (optional) |

### Resources

- **Base model:** [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- **Our fine-tuned model:** [Tony1109/DesignAsCode-planner](https://huggingface.co/Tony1109/DesignAsCode-planner)
- **Training data:** [Tony1109/DesignAsCode-training-data](https://huggingface.co/datasets/Tony1109/DesignAsCode-training-data)
- **Training format:** **Input** = `prompt` → **Output** = `layout_thought` + `grouping` + `image_generator` + `generate_text`

---

## Citation

```bibtex
@article{liu2025designascode,
  title     = {DesignAsCode: Bridging Structural Editability and 
               Visual Fidelity in Graphic Design Generation},
  author    = {Liu, Ziyuan and Sun, Shizhao and Huang, Danqing 
               and Shi, Yingdong and Zhang, Meisheng and Li, Ji 
               and Yu, Jingsong and Bian, Jiang},
  journal   = {arXiv preprint},
  year      = {2025}
}
```
