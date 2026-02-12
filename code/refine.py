import base64
import os
from html2image import html_to_image
from api import get_api_client


def refine(client, html_image_path, html_path, folder, times=1, deployment_chat="gpt-5"):
    """
    Optimize HTML design iteratively:
    1. Optimize the rendered image using an image model.
    2. Feed original HTML, original image, and refined image to GPT-5.
    3. Get improved HTML, save, and render again.
    4. Repeat for given `times`.
    """
    deployment_image = "gpt-image-1"

    # ensure output folder exists
    os.makedirs(folder, exist_ok=True)

    def encode_image_to_data_url(image_path):
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"

    current_html_path = html_path
    current_image_path = html_image_path

    is_all_refine_successful = True

    for i in range(1, times + 1):
        print(f"\n=== Refinement round {i}/{times} ===")

        # 1. Refine image
        print("Refining the rendered image...")
        image_refine_prompt = (
            "Enhance this design image by improving clarity, balance, "
            "color harmony, typography sharpness, and overall aesthetics "
            "while preserving the layout structure."
        )

        refined_image_path = os.path.join(folder, f"reference_{i}.png")
        try:
            result = client.images.edit(
                model=deployment_image,
                image=[open(current_image_path, "rb")],
                prompt=image_refine_prompt,
                background="auto",
                output_format="png",
            )
            image_base64 = result.data[0].b64_json
            with open(refined_image_path, "wb") as f:
                f.write(base64.b64decode(image_base64))
            print(f"Refined image saved to {refined_image_path}")
        except Exception as e:
            print(f"[Round {i}] Failed to refine image: {e}")
            is_all_refine_successful = False
            break

        # 2. Refine HTML using GPT-5
        print("Refining HTML with GPT-5...")
        with open(current_html_path, "r", encoding="utf-8", errors='replace') as f:
            html_content = f.read()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional graphic and web designer.\n"
                    "Based on the original HTML, its rendered image, and the refined reference image, "
                    "improve the HTML code:\n"
                    "- Enhance visual aesthetics (colors, typography, spacing, alignment).\n"
                    "- Ensure layout balance and professional design.\n"
                    "- Preserve semantic correctness and responsiveness.\n"
                    "Return only the improved HTML. Do not place the HTML in the code block.\n\n"
                    "**Strict requirement:\n**"
                    "- The HTML **must include a main container with the class** .poster.\n"
                    "- All design content (background, images, text, decorations) must be placed **inside** .poster."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": (
                        "Here is the original HTML code, along with two images:\n"
                        "- The first image is the Original Rendered HTML Image.\n"
                        "- The second image is the Refined Reference Image.\n\n"
                        "Please use them to improve the HTML."
                    ),
                    },
                    {"type": "text", "text": html_content},
                    {"type": "image_url", "image_url": {"url": encode_image_to_data_url(current_image_path)}},
                    {"type": "image_url", "image_url": {"url": encode_image_to_data_url(refined_image_path)}},
                ],
            },
        ]

        try:
            response = client.chat.completions.create(
                model=deployment_chat,
                messages=messages,
            )
            refined_html_content = response.choices[0].message.content

            refined_html_path = os.path.join(folder, f"refined_{i}.html")
            with open(refined_html_path, "w", encoding="utf-8") as f:
                f.write(refined_html_content)

            print(f"Refined HTML saved to {refined_html_path}")

            # 3. Render new HTML to image
            refined_html_image_path = html_to_image(refined_html_path)
            if refined_html_image_path is None:
                print(f"[Round {i}] Failed to render refined HTML to image.")
                is_all_refine_successful = False
                break
            print(f"Rendered refined HTML for round {i}")

            # update for next round
            current_html_path = refined_html_path
            current_image_path = refined_html_image_path

        except Exception as e:
            print(f"[Round {i}] Failed to refine HTML: {e}")
            is_all_refine_successful = False
            break

    print("\n=== Refinement process completed ===")
    return is_all_refine_successful

if __name__ == "__main__":
    client = get_api_client(model="gpt-4o")
    refine(      
        client=client,
        html_image_path="pipelines/g_retrieve_A flyer advertising fresh seafood with prices and images._2025-09-26_03-12-19/init_result.png",
        html_path="pipelines/g_retrieve_A flyer advertising fresh seafood with prices and images._2025-09-26_03-12-19/init_result.html",
        folder="pipelines/g_retrieve_A flyer advertising fresh seafood with prices and images._2025-09-26_03-12-19",
        times=1
        )
