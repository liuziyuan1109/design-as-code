from openai import APIStatusError, APIError
import base64
import os

def generate_image(client, user_prompt, layer_id, save_folder):
    image_url = f"generated_images/layer{layer_id}.png"
    temp_image_url = f"generated_images/temp_layer{layer_id}.png"
    folder = save_folder
    os.makedirs(os.path.join(folder, "generated_images"), exist_ok=True)
    
    system_prompt = '''system: You are a master of image generation. When the user requests something to be transparent, set the "transparent" fully to rgb(85, 107, 47) rather than Mosaic grids.'''

    prompt = f"{system_prompt}\nuser: {user_prompt}"

    deployment_name = "gpt-image-1"
    successful_image = False
    try:
        response = client.images.generate(
        model=deployment_name,
        prompt=prompt,
        background="auto",
        output_format="png",
    )   
        image_base64 = response.data[0].b64_json
        image_data = base64.b64decode(image_base64)      
        
        image_path = os.path.join(folder, temp_image_url)
        with open(image_path, "wb") as f:
            f.write(image_data)

        successful_image = True
    except APIStatusError as e:
        if e.status_code == 410:  # ModelDeprecated
            print(f"Model {deployment_name} is deprecated (Error {e.status_code}): {getattr(e, 'message', str(e))}")
        else:
            print(f"API Status Error with model {deployment_name} (Error {e.status_code}): {getattr(e, 'message', str(e))}")
    except APIError as e:  # Catch other OpenAI API errors
        print(f"API Error with model {deployment_name}: {e}")
    except Exception as e:  # Catch any other unexpected errors
        print(f"An unexpected error occurred with model {deployment_name}: {e}")

    if not successful_image:
        print(f"\nFailed to generate image with model {deployment_name}.")
        return False
    else:
        print(f"\nImage successfully generated and saved to {image_path}")
        
        
        

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    # Getting the Base64 string
    base64_image = encode_image(image_path)
    deployment_name = "gpt-4o"
    successful_completion = False
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "If the image includes the color rgb(85, 107, 47), return True, otherwise return False. Do not include any other text in your response."
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
        print(f"Need remove background: {response_content}")
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

    if not successful_completion:
        print(f"\nFailed to determine whether the background needs to be changed with model {deployment_name}.")
        os.rename(os.path.join(folder, temp_image_url), os.path.join(folder, image_url))
        return False
    else:
        print(f"\nGeneration successfully completed with model {deployment_name}.")
    
    

    
    
        
    if response_content.strip().lower() == "true":
        deployment_name = "gpt-image-1"
        successful_image = False
        try:
            remove_background_prompt = """
            Remove the area of rgb(85, 107, 47) from the image and make its alpha channel to be 0. Note: The main object may be solid-colored geometric shape or contain solid-colored geometric shape. Do not change it and do not make it transparent.
            """

            result = client.images.edit(
                model="gpt-image-1",
                image=[
                    open(image_path, "rb"),
                ],
                prompt=remove_background_prompt,
                background="transparent",
                output_format="png",
            )

            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)

            # Save the image to a file
            image_path = os.path.join(folder, image_url)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
                
            successful_image = True
            
        except APIStatusError as e:
            if e.status_code == 410:  # ModelDeprecated
                print(f"Model {deployment_name} is deprecated (Error {e.status_code}): {getattr(e, 'message', str(e))}")
            else:
                print(f"API Status Error with model {deployment_name} (Error {e.status_code}): {getattr(e, 'message', str(e))}")
        except APIError as e:  # Catch other OpenAI API errors
            print(f"API Error with model {deployment_name}: {e}")
        except Exception as e:  # Catch any other unexpected errors
            print(f"An unexpected error occurred with model {deployment_name}: {e}")

        if not successful_image:
            print(f"\nFailed to generate image with model {deployment_name}.")
            os.rename(os.path.join(folder, temp_image_url), os.path.join(folder, image_url))
            return False
        else:
            print(f"\nImage successfully generated and saved to {image_path}")
            
    else:
        os.rename(os.path.join(folder, temp_image_url), os.path.join(folder, image_url))
        print("No areas to remove. Image remains unchanged.")
    
    return image_url
