# DesignAsCode

**DesignAsCode** generates editable graphic designs (HTML/CSS) from natural-language prompts via a Plan → Implement → Reflection pipeline.

> **First time?** Follow the [Quick Start Guide](QUICKSTART.md) to set up the environment, download the model & data, and run your first design.

---

## Batch Evaluation

After completing the [Quick Start](QUICKSTART.md) setup, you can run the full pipeline on the test set:

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

## License

Apache License 2.0. See [LICENSE](LICENSE).
