# Quick Start Guide

## 1. Clone Repository

```bash
git clone https://github.com/liuziyuan1109/design-as-code.git
cd design-as-code
```

## 2. Set Up Environment

```bash
conda create -n designascode python=3.10 -y
conda activate designascode
pip install -r requirements.txt

# Install Playwright browser (required for HTML rendering)
playwright install chromium
```

> **Note:** GPU inference requires CUDA 11.8+ and ≥24 GB VRAM.

## 3. Download Model & Data

### Planner Model

Download the fine-tuned Semantic Planner (~16 GB) from Hugging Face:

```bash
# Make sure git-lfs is installed
git lfs install

git clone https://huggingface.co/Tony1109/DesignAsCode-planner models/planner
```

### Image Retrieval Library

Download the image library, FAISS index, and ID mapping used for element retrieval:

```bash
pip install huggingface_hub

# Download all retrieval assets (~19 GB total)
huggingface-cli download Tony1109/crello-image-library --repo-type dataset --local-dir retrieval_assets

# Place files into data/
cd retrieval_assets
tar -xzf crello_pngs.tar.gz
mv crello_pngs ../data/image_library
mv elements_local.index ../data/
mv id_mapping_local.json ../data/
cd .. && rm -rf retrieval_assets
```

## 4. Set OpenAI API Key

The pipeline calls three OpenAI models, so your API key must have access to all of them:

| Model | Purpose |
|---|---|
| `gpt-5` | HTML/CSS generation and layout refinement |
| `gpt-4o` | Image quality analysis |
| `gpt-image-1` | Image generation and editing |

```bash
export OPENAI_API_KEY='sk-your-api-key-here'
```

> Get your API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

## 5. Run Inference

Generate a complete graphic design (HTML + rendered image) from a text prompt:

```bash
python infer.py \
  --prompt "A modern promotional poster for a coffee shop grand opening" \
  --output output/coffee_shop
```

### Arguments

| Argument       | Description                           | Default                        |
|----------------|---------------------------------------|--------------------------------|
| `--prompt`     | Design description (required)         | —                              |
| `--model`      | Path to planner model                 | `models/planner`               |
| `--index`      | Path to FAISS index file              | `data/elements_local.index`    |
| `--id-mapping` | Path to ID mapping JSON               | `data/id_mapping_local.json`   |
| `--device`     | `cuda` or `cpu`                       | auto-detect                    |
| `--output`     | Output directory                      | `output`                       |

### Pipeline Steps

The script runs the full DesignAsCode pipeline:

1. **Plan** — Planner model generates layout thought, grouping, image prompts, and text specs
2. **Retrieve** — Semantic search retrieves matching images from the image library via FAISS
3. **Implement** — LLM generates HTML/CSS design from the plan and retrieved images
4. **Refine (images)** — LLM-guided image element refinement
5. **Refine (layout)** — LLM-guided overall layout refinement

### Output

Results are saved to the output directory:

```
output/coffee_shop/
├── planner_output.txt        # Raw design plan from planner model
├── generated_images/         # Retrieved images for each layer
├── layout_prompt.txt         # Combined prompt sent to LLM
├── init_result.html          # Initial HTML design
├── after_image_refine.html   # After image refinement
└── refine_*.html             # After layout refinement (final result)
```
