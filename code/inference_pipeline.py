from transformers import AutoTokenizer, AutoModelForCausalLM
from api import get_api_client
from generate_image import generate_image
from html2image import html_to_image
from openai import APIStatusError, APIError
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
from refine import refine
from image_refine import image_refine
import torch
import re
import json
import os
import time
import faiss
import shutil
import gc
import argparse

def g_mix_test(client, user_prompt, example_id, tokenizer, model, retrieve_model, index, id_mapping, fail_log, success_log, output_dir):
    
    start_time = time.time()

    folder = os.path.join(output_dir, example_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Input text
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

    print(f"Starting to generate a design plan: {user_prompt}")
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():  # Prevent memory leaks
        # Call generate to produce text
        output_ids = model.generate(
            **inputs,
            max_new_tokens=8192,  # Generation length
            do_sample=True,      # Use sampling
            temperature=0.7,     # Sampling temperature
            top_p=0.9            # Nucleus sampling parameter
        )

    # Decode output
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
    # print(full_text)

    # Clean up CUDA memory tensors
    del inputs, output_ids
    torch.cuda.empty_cache()
    gc.collect()

    match = re.search(r"<image_generator>(.*?)</image_generator>", generated_text, re.DOTALL)

    if not match:
        print("<image_generator> tag content not found")
        with open(fail_log, "a", encoding="utf-8") as f:
            f.write(f"{example_id}\n")
        return
        
    try:
        json_str = match.group(1).strip()
        image_generator_list = json.loads(json_str)
    except Exception as e:
        print(f"Failed to parse <image_generator> content: {e}")
        with open(fail_log, "a", encoding="utf-8") as f:
            f.write(f"{example_id}\n")
        return
    print("Successfully found <image_generator> tag content")

    # Retrieve images
    image_generator_results = []
    # retrieve_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    os.makedirs(os.path.join(folder, "generated_images"), exist_ok=True)
    IMG_ROOT = "data/image_library"
    # Load FAISS index and ID mapping
    # index = faiss.read_index("elements_local.index")
    # with open("id_mapping_local.json", "r", encoding="utf-8") as f:
    #     id_mapping = json.load(f)
    for item in tqdm(image_generator_list, desc="Retrieving images"):
        # User query
        query = item["layer_prompt"]
        query_emb = retrieve_model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        # Retrieve from FAISS index
        k = 1
        distances, indices = index.search(query_emb, k)
        for idx, dist in zip(indices[0], distances[0]):
            image_url = f"generated_images/layer{item['layer_id']}.png"
            image_path = os.path.join(folder, image_url)
            
            elem_id = id_mapping[idx] 
            origin_img_path = os.path.join(IMG_ROOT, elem_id + ".png")
            if os.path.exists(origin_img_path):
                shutil.copy(origin_img_path, image_path)
                print(f"✅ Copied {origin_img_path} → {image_path} (similarity={dist:.4f})")
            else:
                print(f"⚠️ Image not found: {origin_img_path}")
                with open(fail_log, "a", encoding="utf-8") as f:
                    f.write(f"{example_id}\n")
                return
        
        image_generator_results.append({
                "layer_id": item["layer_id"],
                "url": image_url
            })
        
    image_generator_results = "<image_generator_result>\n" + json.dumps(image_generator_results, ensure_ascii=False, separators=(", ", ": ")) + "\n</image_generator_result>"
    layout_prompt = "<user_input>\n" + user_prompt + "\n</user_input>\n" + generated_text + "\n" + image_generator_results
    # print(layout_prompt)
    with open(f"{folder}/layout_prompt.txt", "w", encoding="utf-8") as f:
        f.write(layout_prompt)


    layout_system_prompt = '''
    You are a **2D graphic design master**, highly skilled in generating a wide range of visual designs, especially **posters, advertisements, and promotional pages in HTML format**.  

    When creating the final output, you must follow and consider these key elements:  
    - The **user’s input**  
    - The **intermediate reasoning process (layout_thought)**  
    - Content generated by various tools, including **text** and **images** (insert image URLs directly into the HTML)  

    **Strict requirement:**
    - The final HTML **must include a main container with the class** .poster.
    - All design content (background, images, text, decorations) must be placed **inside** .poster.

    Your output should be a 2D graphic design in the form of HTML. Just return the HTML. Do not return anything else except this. Do not place the HTML in the code block.
    '''
    # compose messages
    messages = [
        {
            "role": "system",
            "content": layout_system_prompt,
        },
        {
            "role": "user",
            "content": layout_prompt,
        }
    ]

    print("\nStarting initial overall design generation...")
    deployment_name = "gpt-5"
    successful_completion = False
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
        )
        response_content = response.choices[0].message.content
        # print(response_content)
        successful_completion = True
    except APIStatusError as e:
        if e.status_code == 410:  # ModelDeprecated
            print(f"Model {deployment_name} is deprecated (Error {e.status_code}): {getattr(e, 'message', str(e))}")
        else:
            print(f"API Status Error with model {deployment_name} (Error {e.status_code}): {getattr(e, 'message', str(e))}")
    except APIError as e:  # Catch other OpenAI API errors
        print(f"API Error with model {deployment_name}: {e}")
    except Exception as e:  # Catch any other unexpected errors
        print(f"An unexpected error occurred with model {deployment_name}: {e}")
        
    if successful_completion:
        html_path = f"{folder}/init_result.html"
        html_content = response_content
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(response_content)
        print(f"\nInitial design result saved to {html_path}")
        html_image_path = html_to_image(html_path)
    else:
        print("\nFailed to generate the initial design result.")
        with open(fail_log, "a", encoding="utf-8") as f:
            f.write(f"{example_id}\n")
        return
        
    successful_image_refine = image_refine(
            client=client,
            layout_prompt=layout_prompt,
            html_content=html_content,
            html_image_path=html_image_path,
            folder=folder
        )
    if not successful_image_refine:
        with open(fail_log, "a", encoding="utf-8") as f:
            f.write(f"{example_id}\n")
        return
    after_image_refine_html_path = f"{folder}/after_image_refine.html"
    shutil.copy(f"{folder}/init_result.html", after_image_refine_html_path)
    html_image_path = html_to_image(after_image_refine_html_path)
    html_path = after_image_refine_html_path

    successful_refine = refine(client=client, html_image_path=html_image_path, html_path=html_path, folder=folder, times=1)
    if not successful_refine:
        with open(fail_log, "a", encoding="utf-8") as f:
            f.write(f"{example_id}\n")
        return

    end_time = time.time()

    print(f"Execution time: {(end_time - start_time) / 60:.4f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference pipeline")
    parser.add_argument("--model-path", type=str, default="../models/planner", help="Path to planner model")
    parser.add_argument("--index-path", type=str, default="../data/elements_local.index", help="Path to FAISS index")
    parser.add_argument("--id-mapping-path", type=str, default="../data/id_mapping_local.json", help="Path to ID mapping JSON")
    parser.add_argument("--test-data", type=str, default="../data/test.jsonl", help="Path to test dataset JSONL")
    parser.add_argument("--output-dir", type=str, default="batch_outputs", help="Output directory")
    parser.add_argument("--shard-index", type=int, default=0, help="Current shard index (for distributed runs)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    client = get_api_client(model="gpt-5")

    model_path = args.model_path
    print(f"Loading planner model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
    model.eval()

    print("Loading retrieval model and FAISS index...")
    retrieve_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    index = faiss.read_index(args.index_path)
    with open(args.id_mapping_path, "r", encoding="utf-8") as f:
        id_mapping = json.load(f)

    prompts = []
    ids = []
    with open(args.test_data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if "prompt" in data and "id" in data:
                    prompts.append(data["prompt"])
                    ids.append(data["id"])

    shard_index = args.shard_index
    num_shards = args.num_shards
    success_log = f"batch_success_{shard_index}.txt"
    fail_log = f"batch_failed_{shard_index}.txt"
    output_dir = args.output_dir
    if not os.path.exists(success_log):
        open(success_log, "w", encoding="utf-8").close()
    open(fail_log, "w", encoding="utf-8").close()
    os.makedirs(output_dir, exist_ok=True)

    all_examples = sorted(zip(prompts, ids), key=lambda x: x[1])
    examples = all_examples[shard_index::num_shards]
    print(f"Total examples: {len(all_examples)}, shard {shard_index}/{num_shards} has {len(examples)} examples")

    for prompt, id in tqdm(examples, desc="Processing"):
        print(f"Processing example ID: {id}")
        with open(success_log, "r", encoding="utf-8") as f:
            successful_ids = set(line.strip() for line in f if line.strip())
        if id not in successful_ids:
            g_mix_test(client, prompt, id, tokenizer, model, retrieve_model, index, id_mapping, fail_log=fail_log, success_log=success_log, output_dir=output_dir)
            torch.cuda.empty_cache()
            gc.collect()
            with open(fail_log, "r", encoding="utf-8") as f:
                failed_ids = set(line.strip() for line in f if line.strip())
            if id not in failed_ids:
                with open(success_log, "a", encoding="utf-8") as f:
                    f.write(f"{id}\n")
        print(f"Finished example ID: {id}\n{'-'*50}\n")