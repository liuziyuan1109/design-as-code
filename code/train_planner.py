"""
SFT training script for the DesignAsCode Semantic Planner.

Usage:
    # Single GPU (no DeepSpeed)
    python train_planner.py --model Qwen/Qwen3-8B --data training_data.jsonl --output-dir ./ckpt/planner

    # Multi-GPU with DeepSpeed
    deepspeed --num_gpus=4 train_planner.py --model Qwen/Qwen3-8B --data training_data.jsonl --output-dir ./ckpt/planner --deepspeed ds_config.json
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

SYSTEM_PROMPT = '''You are a master of 2D graphic design. You are skilled in planning 2D design, adept at providing design concepts and layout thought, and capable of generating corresponding grouping plans, image prompts and text content based on the layout thought.  

Workflow:  

1. Provide the `layout_thought`, enclosed in <layout_thought>...</layout_thought>. As detailed as possible, including the layout structure and any specific elements (layers).

2. Provide the `grouping`, enclosed in <grouping>...</grouping>. It should be a JSON array that groups related layers together. Each group must be expressed as a JSON object with three fields:
- `group_id`: a unique identifier string like 'G1', 'G2'...
- `children`: a list of layer_ids (from the `layout_thought` you just generated) that belong to this group
- `theme`: a short description (2–6 words) summarizing the group's purpose (e.g., 'menu item', 'header block').
The `grouping` should help later stages bind text and image elements correctly.
If an element is standalone and not obviously related, it can form its own group.
The grouping must appear right after the layout_thought, and will guide the subsequent image and text generation.

3. Provide the image generation prompts, enclosed in <image_generator>...</image_generator>, for example:  

<image_generator>
[
  {"layer_id": 0, "layer_prompt": "prompt0"},
  {"layer_id": 1, "layer_prompt": "prompt1"}
]
</image_generator>

4. Provide the text element design, enclosed in <generate_text>...</generate_text>. Example:  

<generate_text>
[
  {
    "layer_id": 6,
    "type": "TextElement",
    "width": 302.1794738769531,
    "height": 31.327075958251953,
    "opacity": 1.0,
    "text": "Big Fall Volunteer",
    "font": "Abril Fatface",
    "font_size": 31.39527130126953,
    "text_align": "center",
    "angle": 0.0,
    "capitalize": false,
    "line_height": 1.0,
    "letter_spacing": 0.9849796295166016
  },
  {
    "layer_id": 7,
    "type": "TextElement",
    "width": 322.0,
    "height": 67.89791107177734,
    "opacity": 1.0,
    "text": " Cleanup",
    "font": "Abril Fatface",
    "font_size": 68.0,
    "text_align": "center",
    "angle": 0.0,
    "capitalize": false,
    "line_height": 1.0,
    "letter_spacing": 0.0
  }
]
</generate_text>

Important:  
- <layout_thought>...</layout_thought>, <grouping>...</grouping>, <image_generator>...</image_generator>, and <generate_text>...</generate_text> are mandatory and must appear exactly once.  
'''


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training for DesignAsCode Semantic Planner")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Base model name or path (default: Qwen/Qwen3-8B)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to training data JSONL file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save checkpoints and final model")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs (default: 2)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device training batch size (default: 1)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2,
                        help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--max-length", type=int, default=6144,
                        help="Maximum token length (default: 6144)")
    parser.add_argument("--save-steps", type=int, default=500,
                        help="Save checkpoint every N steps (default: 500)")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed config JSON (optional)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # Load and format dataset
    dataset = load_dataset("json", data_files=args.data, split="train")

    def formatting_func(example):
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        completion = (
            example["layout_thought"] + "\n"
            + example["grouping"] + "\n"
            + example["image_generator"] + "\n"
            + example["generate_text"] + "<|im_end|>\n"
        )
        return {"prompt": prompt, "completion": completion}

    formatted_dataset = dataset.map(formatting_func, remove_columns=dataset.column_names)

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_steps=args.save_steps,
        num_train_epochs=args.epochs,
        optim="adamw_torch",
        report_to=["tensorboard"],
        logging_dir=f"{args.output_dir}/logs",
        completion_only_loss=True,
        max_length=args.max_length,
        gradient_checkpointing=True,
        deepspeed=args.deepspeed,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        args=training_args,
    )

    trainer.train()

    # Save final model
    final_dir = f"{args.output_dir}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()