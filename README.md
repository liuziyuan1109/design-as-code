# DesignAsCode

**DesignAsCode** generates editable graphic designs (HTML/CSS) from natural-language prompts via a Plan → Implement → Reflection pipeline.

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
git lfs install
git clone https://huggingface.co/Tony1109/DesignAsCode-planner models/planner

# Image retrieval library + FAISS index (~19 GB)
pip install huggingface_hub
huggingface-cli download Tony1109/crello-image-library --repo-type dataset --local-dir retrieval_assets
cd retrieval_assets
tar -xzf crello_pngs.tar.gz
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
  --prompt "A modern promotional poster for a coffee shop grand opening" \
  --output output/coffee_shop
```

Output is saved to the specified directory:

```
output/coffee_shop/
├── planner_output.txt        # Design plan from planner model
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

To train your own Semantic Planner:

- **Base model:** [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- **Our fine-tuned model:** [Tony1109/DesignAsCode-planner](https://huggingface.co/Tony1109/DesignAsCode-planner)
- **Training data:** [Tony1109/DesignAsCode-training-data](https://huggingface.co/datasets/Tony1109/DesignAsCode-training-data) (19,479 samples distilled from Crello; we used ~10k for training)
- **Image retrieval library:** [Tony1109/crello-image-library](https://huggingface.co/datasets/Tony1109/crello-image-library) (FAISS index + 228k element images)

Training format: **Input** = `prompt` → **Output** = `layout_thought` + `grouping` + `image_generator` + `generate_text`

See the [training data repo](https://huggingface.co/datasets/Tony1109/DesignAsCode-training-data) for field details.

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
