#!/usr/bin/env python3
"""
DesignAsCode - Single prompt full-pipeline inference script.
Generates a complete graphic design (HTML + rendered image) from a text prompt.
"""
import torch
import json
import re
import os
import sys
import time
import gc
import shutil
import argparse
import faiss
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from openai import APIStatusError, APIError

# Add code/ to path so we can import local modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
from api import get_api_client
from html2image import html_to_image
from refine import refine
from image_refine import image_refine


# LLM model used for HTML generation and refinement
LLM_MODEL = "gpt-5"


def single_inference(client, user_prompt, tokenizer, model, retrieve_model, index, id_mapping, output_dir):
    """
    Run the full DesignAsCode pipeline for a single prompt:
      1. Planner model generates design plan (layout_thought, grouping, image prompts, text specs)
      2. Retrieve images from the image library via FAISS
      3. LLM generates HTML/CSS design
      4. Image refinement
      5. Layout refinement

    Args:
        client: OpenAI API client
        user_prompt: Design description text
        tokenizer: Planner model tokenizer
        model: Planner model
        retrieve_model: SentenceTransformer retrieval model
        index: FAISS index
        id_mapping: FAISS index → image ID mapping
        output_dir: Output directory
    """
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Step 1: Generate design plan with planner model ──────────────────
    system_prompt = '''You are a master of 2D graphic design. You are skilled in planning 2D design, adept at providing design concepts and layout thought, and capable of generating corresponding grouping plans, image prompts and text content based on the layout thought.  

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

    print("=" * 60)
    print(f"📝 Prompt: {user_prompt}")
    print("=" * 60)

    print("\n⏳ Step 1/5: Generating design plan with planner model...")
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    generated_text = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

    print("✅ Design plan generated")

    del inputs, output_ids
    torch.cuda.empty_cache()
    gc.collect()

    # Parse <image_generator> tag
    match = re.search(r"<image_generator>(.*?)</image_generator>", generated_text, re.DOTALL)
    if not match:
        print("❌ Failed: <image_generator> tag not found in planner output.")
        return False

    try:
        image_generator_list = json.loads(match.group(1).strip())
    except Exception as e:
        print(f"❌ Failed to parse <image_generator> content: {e}")
        return False

    # ── Step 2: Retrieve images from library ─────────────────────────────
    print(f"\n⏳ Step 2/5: Retrieving images for {len(image_generator_list)} layers...")
    image_generator_results = []
    os.makedirs(os.path.join(output_dir, "generated_images"), exist_ok=True)
    IMG_ROOT = "data/image_library"

    for item in tqdm(image_generator_list, desc="Retrieving images"):
        query = item["layer_prompt"]
        query_emb = retrieve_model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        k = 1
        distances, indices = index.search(query_emb, k)
        for idx, dist in zip(indices[0], distances[0]):
            image_url = f"generated_images/layer{item['layer_id']}.png"
            image_path = os.path.join(output_dir, image_url)

            elem_id = id_mapping[idx]
            origin_img_path = os.path.join(IMG_ROOT, elem_id + ".png")
            if os.path.exists(origin_img_path):
                shutil.copy(origin_img_path, image_path)
                print(f"  ✅ Layer {item['layer_id']}: {origin_img_path} (similarity={dist:.4f})")
            else:
                print(f"  ❌ Image not found: {origin_img_path}")
                return False

        image_generator_results.append({
            "layer_id": item["layer_id"],
            "url": image_url
        })

    image_generator_results_str = "<image_generator_result>\n" + json.dumps(image_generator_results, ensure_ascii=False, separators=(", ", ": ")) + "\n</image_generator_result>"
    layout_prompt = "<user_input>\n" + user_prompt + "\n</user_input>\n" + generated_text + "\n" + image_generator_results_str

    with open(os.path.join(output_dir, "layout_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(layout_prompt)
    print("✅ Image retrieval complete")

    # ── Step 3: Generate HTML design via LLM ─────────────────────────────
    print(f"\n⏳ Step 3/5: Generating HTML design via {LLM_MODEL}...")
    layout_system_prompt = '''
    You are a **2D graphic design master**, highly skilled in generating a wide range of visual designs, especially **posters, advertisements, and promotional pages in HTML format**.  

    When creating the final output, you must follow and consider these key elements:  
    - The **user's input**  
    - The **intermediate reasoning process (layout_thought)**  
    - Content generated by various tools, including **text** and **images** (insert image URLs directly into the HTML)  

    **Strict requirement:**
    - The final HTML **must include a main container with the class** .poster.
    - All design content (background, images, text, decorations) must be placed **inside** .poster.

    Your output should be a 2D graphic design in the form of HTML. Just return the HTML. Do not return anything else except this. Do not place the HTML in the code block.
    '''

    messages = [
        {"role": "system", "content": layout_system_prompt},
        {"role": "user", "content": layout_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
        )
        html_content = response.choices[0].message.content
    except (APIStatusError, APIError) as e:
        print(f"❌ API error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    html_path = os.path.join(output_dir, "init_result.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ Initial HTML saved to init_result.html")

    html_image_path = html_to_image(html_path)
    print(f"✅ Rendered to {os.path.basename(html_image_path)}")

    # ── Step 4: Image refinement ─────────────────────────────────────────
    print("\n⏳ Step 4/5: Refining images in design...")
    successful_image_refine = image_refine(
        client=client,
        layout_prompt=layout_prompt,
        html_content=html_content,
        html_image_path=html_image_path,
        folder=output_dir
    )
    if not successful_image_refine:
        print("⚠️ Image refinement failed, continuing with initial result...")

    after_image_refine_html_path = os.path.join(output_dir, "after_image_refine.html")
    shutil.copy(os.path.join(output_dir, "init_result.html"), after_image_refine_html_path)
    html_image_path = html_to_image(after_image_refine_html_path)
    html_path = after_image_refine_html_path
    print("✅ Image refinement complete")

    # ── Step 5: Layout refinement ────────────────────────────────────────
    print("\n⏳ Step 5/5: Refining overall layout...")
    successful_refine = refine(
        client=client,
        html_image_path=html_image_path,
        html_path=html_path,
        folder=output_dir,
        times=1
    )
    if not successful_refine:
        print("⚠️ Layout refinement failed, using previous result.")

    elapsed = (time.time() - start_time) / 60
    print("\n" + "=" * 60)
    print(f"✅ Pipeline complete! ({elapsed:.1f} min)")
    print(f"📂 Output directory: {output_dir}")
    print("   Key files:")
    print(f"   - layout_prompt.txt       (design plan + retrieved image URLs)")
    print(f"   - init_result.html        (initial design)")
    print(f"   - after_image_refine.html (after image refinement)")
    for f in sorted(os.listdir(output_dir)):
        if f.startswith("refine_") and f.endswith(".html"):
            print(f"   - {f}  (refined design)")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="DesignAsCode: Generate a complete graphic design from a text prompt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer.py --prompt "A modern poster for a coffee shop grand opening"
  python infer.py --prompt "A minimalist business card" --output results/my_design
  python infer.py --prompt "Summer sale banner" --device cpu
        """
    )
    parser.add_argument("--prompt", type=str, required=True, help="Design description")
    parser.add_argument("--model", type=str, default="models/planner", help="Path to planner model (default: models/planner)")
    parser.add_argument("--index", type=str, default="data/elements_local.index", help="Path to FAISS index file")
    parser.add_argument("--id-mapping", type=str, default="data/id_mapping_local.json", help="Path to ID mapping JSON file")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None, help="Device (default: auto-detect)")
    parser.add_argument("--output", type=str, default="output", help="Output directory (default: output)")

    args = parser.parse_args()

    # Check OPENAI_API_KEY
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is not set.")
        print("   Please set it before running:")
        print("   export OPENAI_API_KEY='sk-your-api-key-here'")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Initialize OpenAI client
    print(f"🔑 Initializing OpenAI client...")
    client = get_api_client()

    # Load planner model
    print(f"🧠 Loading planner model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    planner_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        planner_model = planner_model.to(device)
    planner_model.eval()

    # Load retrieval model and FAISS index
    print("🔍 Loading retrieval model and FAISS index...")
    retrieve_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    index = faiss.read_index(args.index)
    with open(args.id_mapping, "r", encoding="utf-8") as f:
        id_mapping = json.load(f)
    print(f"   Index: {args.index} ({index.ntotal} vectors)")
    print(f"   ID mapping: {len(id_mapping)} entries")

    # Run full pipeline
    success = single_inference(
        client=client,
        user_prompt=args.prompt,
        tokenizer=tokenizer,
        model=planner_model,
        retrieve_model=retrieve_model,
        index=index,
        id_mapping=id_mapping,
        output_dir=args.output
    )

    if not success:
        print("\n❌ Pipeline failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
