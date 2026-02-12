import re
import os
import json
import base64
import time
import shutil
from generate_image import generate_image
from api import get_api_client
from openai import APIStatusError, APIError

def image_refine(client, layout_prompt, html_content, html_image_path, folder):

    start_time = time.time()

    deployment_name = "gpt-4o"

    # === 1. Parse generated information from layout_prompt ===
    def extract_json(tag):
        match = re.search(rf"<{tag}>(.*?)</{tag}>", layout_prompt, re.DOTALL)
        if not match:
            print(f"<{tag}> tag content not found")
            return None
        try:
            return json.loads(match.group(1).strip())
        except Exception as e:
            print(f"<{tag}> content parsing failed: {e}")
            return None

    image_generator_list = extract_json("image_generator")
    if image_generator_list is None:
        return False

    # === 2. Call GPT to analyze image quality ===
    print("Analyzing which generated images have poor quality...")

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Getting the Base64 string
    base64_image = encode_image(html_image_path)

    loose_prompt = f"""
You are a visual design quality reviewer.

Please review the following design description and its generated images.

Only report layers that are **significantly wrong** — for example:
- The image clearly mismatches the prompt (wrong subject, color, style, or transparency)
- The image quality is very poor (e.g., blurry, broken, unreadable)

Do NOT report minor or acceptable differences. Only evaluate the image layers; there is no need to evaluate the text layers.

If everything looks acceptable, return only:
{{ "bad_layers": [] }}

Return your result strictly in JSON format as follows (only JSON, no extra text):
{{
  "bad_layers": [layer_id, ...],
  "reason": {{
    "layer_id": "short English description of what is wrong"
  }}
}}

Inputs:
- layout_prompt: {layout_prompt}
- html_content: {html_content}
"""

    strict_prompt = f"""
You are a professional visual design quality inspector.

Compare the following materials and identify which generated image layers do NOT match the design intent (e.g., wrong shape, color, opacity, or style).

Only evaluate the image layers; there is no need to evaluate the text layers.

Return a clean JSON only if there are issues; if all images look acceptable, return {{"bad_layers": []}}.

Inputs:
- layout_prompt: {layout_prompt}
- html_content: {html_content}

Required output format (JSON only, no extra text):
{{
  "bad_layers": [layer_id, ...],
  "reason": {{
    "layer_id": "short English description of what is wrong"
  }}
}}
"""

    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional visual design critic. Your job is to inspect image rendering quality against layout instructions.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": loose_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                        },
                    ],
                },
            ]
        )
        response_content = response.choices[0].message.content
        print(f"Successfully used model: {deployment_name}")
    except APIStatusError as e:
        if e.status_code == 410:  # ModelDeprecated
            print(f"Model {deployment_name} is deprecated (Error {e.status_code}): {getattr(e, 'message', str(e))}")
        else:
            print(f"API Status Error with model {deployment_name} (Error {e.status_code}): {getattr(e, 'message', str(e))}")
        return False
    except APIError as e:  # Catch other OpenAI API errors
        print(f"API Error with model {deployment_name}: {e}")
        return False
    except Exception as e:  # Catch any other unexpected errors
        print(f"An unexpected error occurred with model {deployment_name}: {e}")
        return False


    try:
        refine_info = json.loads(re.search(r"\{.*\}", response_content, re.DOTALL).group(0))
    except:
        print("Failed to parse GPT analysis output:", response_content)
        return False

    bad_layers = refine_info.get("bad_layers", [])
    if not bad_layers:
        print("All image quality is good, no regeneration needed.")
        return True

    print("Detected layers that need regeneration:", bad_layers)

    # === 3. Regenerate poor quality images ===
    layer_prompts = {}
    for layer in bad_layers:
        layer_prompt = next((x["layer_prompt"] for x in image_generator_list if x["layer_id"] == layer), None)
        if not layer_prompt:
            print(f"Generation prompt not found for layer {layer}, skipping.")
            continue
        layer_prompts[layer] = layer_prompt

        print(f"Regenerating layer {layer}...")
        success_layer = generate_image(client, layer_prompt, layer, folder)
        if not success_layer:
            print(f"Layer {layer} regeneration failed, terminating optimization process.")
            return False

    # === 4. Update layout_prompt or write result file ===
    refined_info = {
        "refine_rate": len(bad_layers) / len(image_generator_list),
        "refined_layers": bad_layers,
        "reason": refine_info.get("reason", {}),
        "layer_prompt": layer_prompts
    }

    output_path = os.path.join(folder, "image_refine_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(refined_info, f, ensure_ascii=False, indent=2)

    print(f"Image optimization complete, results saved to {output_path}")
    end_time = time.time()
    print(f"Image refine took: {(end_time - start_time) / 60:.4f} minutes")
    return True

if __name__ == "__main__":
    client = get_api_client()
    print(f"Using model: {client.default_model}")
    print("Image refinement module loaded successfully.")
