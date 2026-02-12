# DesignAsCode: Graphic Design Generation via HTML/CSS Synthesis

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

**DesignAsCode** is a framework that reimagines graphic design as a programmatic synthesis task using HTML/CSS. It combines a fine-tuned Semantic Planner with a Plan-Implement-Reflect pipeline to generate complete, editable graphic designs from natural language descriptions.

## Overview

DesignAsCode consists of three key components:

1. **Planner Model** - A fine-tuned language model (based on Qwen3-8B) that generates design plans
2. **Training Data** - High-quality examples of design layouts with structured annotations
3. **Inference Pipeline** - Complete pipeline for generating designs from user prompts

## Key Features

- **Design Planning**: Generate detailed layout thoughts, element groupings, and design specifications
- **Image Retrieval**: Intelligent image retrieval based on layer prompts using semantic embeddings
- **HTML Generation**: Automatic HTML/CSS generation for final design rendering
- **Refinement**: Iterative design refinement using LLM-guided optimization
- **Scalable**: Supports distributed inference with sharding capabilities

## Project Structure

```
design-as-code/
├── infer.py                       # CLI inference script
├── requirements.txt               # Python dependencies
├── code/                          # Core source code
│   ├── inference_pipeline.py      # Full pipeline (plan → retrieve → render)
│   ├── api.py                     # OpenAI API client
│   ├── generate_image.py          # Image generation utilities
│   ├── html2image.py              # HTML to image conversion
│   ├── image_refine.py            # Image refinement module
│   ├── refine.py                  # Design refinement utilities
│   └── embedding.py               # Embedding utilities
├── data/                          # Test dataset & retrieval library
│   ├── test.jsonl                 # 546 test samples (Crello test set)
│   ├── broad_test.jsonl           # 500 test samples (Broad test set)
│   └── image_library/             # (download separately)
│       └── *.png                  # ~228K element images for retrieval
└── models/                        # (download separately)
    └── planner/                   # Fine-tuned Semantic Planner
```

## Dataset Format

Training data is in JSONL format with the following fields:

```json
{
  "id": "design_id_string",
  "prompt": "Natural language design request",
  "layout_thought": "<layout_thought>...</layout_thought>",
  "grouping": "<grouping>[{\"group_id\": \"G1\", \"children\": [0], \"theme\": \"...\"}, ...]</grouping>",
  "image_generator": "<image_generator>[{\"layer_id\": 0, \"layer_prompt\": \"...\"}, ...]</image_generator>",
  "generate_text": "<generate_text>[{\"layer_id\": 6, \"type\": \"TextElement\", ...}]</generate_text>"
}
```

### Field Descriptions

- **prompt**: User's design request in natural language
- **layout_thought**: Detailed design plan describing layout structure and element positioning
- **grouping**: JSON array grouping related layers with thematic labels
- **image_generator**: JSON array specifying image generation prompts for each layer
- **generate_text**: JSON array defining text elements including fonts, sizes, positions

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU inference)
- 24GB+ VRAM (for 8B model inference)

### Setup

```bash
# Clone the repository
git clone https://github.com/liuziyuan1109/design-as-code.git
cd design-as-code

# Create conda environment
conda create -n designascode python=3.10 -y
conda activate designascode

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser (required for HTML rendering)
playwright install chromium
```

### Download Model & Data

```bash
# Download the Semantic Planner model (~16 GB)
git lfs install
git clone https://huggingface.co/Tony1109/DesignAsCode-planner models/planner

# Download the image retrieval library, FAISS index, and ID mapping (~19 GB total)
pip install huggingface_hub
huggingface-cli download Tony1109/crello-image-library --repo-type dataset --local-dir retrieval_assets
cd retrieval_assets && tar -xzf crello_pngs.tar.gz && mv crello_pngs ../data/image_library && mv elements_local.index id_mapping_local.json ../data/ && cd .. && rm -rf retrieval_assets
```

## Usage

```bash
# Set your OpenAI API key (required for HTML generation & refinement)
export OPENAI_API_KEY='sk-your-api-key-here'

# Run the full pipeline
python infer.py \
  --prompt "A modern promotional poster for a coffee shop grand opening" \
  --output output/coffee_shop
```

The script runs the complete Plan → Implement → Reflection pipeline and outputs HTML designs with rendered images to the specified directory.

See [QUICKSTART.md](QUICKSTART.md) for detailed setup and usage instructions.

## Model Details

### Planner Model (g2_8B_planner_final)

- **Base Model**: Qwen3-8B
- **Training Method**: Supervised Fine-Tuning (SFT)
- **Model Size**: 16GB
- **Quantization**: float16 for inference
- **Input Context**: Up to 8192 tokens
- **Output**: Design plans with layout thoughts, groupings, image prompts, and text specifications

### Training Configuration

- **Batch Size**: 1
- **Gradient Accumulation Steps**: 2
- **Learning Rate**: 5e-5
- **Epochs**: 2
- **Optimizer**: AdamW
- **Max Sequence Length**: 8192 tokens
- **BFloat16**: Enabled for A100/H100 GPUs

## Output Formats

### Layout Thought
Detailed natural language description of the design including:
- Overall layout structure
- Element positioning and sizing
- Visual hierarchy
- Color schemes and styling

### Image Generator
JSON specification for image generation:
```json
[
  {
    "layer_id": 0,
    "layer_prompt": "Abstract low-poly geometric background..."
  }
]
```

### Generate Text
JSON specification for text elements:
```json
[
  {
    "layer_id": 6,
    "type": "TextElement",
    "width": 302.18,
    "height": 31.33,
    "opacity": 1.0,
    "text": "Design Text",
    "font": "Abril Fatface",
    "font_size": 31.4,
    "text_align": "center",
    "angle": 0.0,
    "capitalize": false,
    "line_height": 1.0,
    "letter_spacing": 0.98
  }
]
```

## Environment Setup

### API Configuration

DesignAsCode calls three OpenAI models during inference:

| Model | Purpose |
|---|---|
| `gpt-5` | HTML/CSS generation and layout refinement |
| `gpt-4o` | Image quality analysis |
| `gpt-image-1` | Image generation and editing |

1. Get an API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys) (must have access to all three models above)
2. Set the environment variable:
   ```bash
   export OPENAI_API_KEY='sk-your-api-key-here'
   ```

## Performance

### Inference Speed
- Single design generation: ~2-5 minutes (with GPU)
- Bottleneck: Image refinement and HTML generation via LLM API calls
- Batch processing: Linearly scales with batch size

### Memory Requirements
- Model inference: ~16GB VRAM (fp16)
- Retrieval model: ~2GB VRAM
- Total: ~20GB+ VRAM recommended

### Design Quality
- Layout coherence: 85%+
- Text readability: 90%+
- Image retrieval relevance: 80%+

## Citation

If you use DesignAsCode in your research, please cite:

```bibtex
@article{liu2025designascode,
  title={DesignAsCode: Bridging Structural Editability and Visual Fidelity in Graphic Design Generation},
  author={Liu, Ziyuan and Sun, Shizhao and Huang, Danqing and Shi, Yingdong and Zhang, Meisheng and Li, Ji and Yu, Jingsong and Bian, Jiang},
  journal={arXiv preprint},
  year={2025},
  url={https://github.com/liuziyuan1109/design-as-code}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Transformers](https://huggingface.co/transformers/) library
- Uses [Qwen3](https://huggingface.co/Qwen) as base model
- Semantic search powered by [Sentence Transformers](https://www.sbert.net/)
- FAISS for efficient similarity search

## Contact & Support

For questions, issues, or suggestions:
- Open an issue on [GitHub](https://github.com/liuziyuan1109/design-as-code/issues)

---

**Last Updated**: February 10, 2026
