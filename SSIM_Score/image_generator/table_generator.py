from io import BytesIO
from diffusers import StableDiffusionXLPipeline
import torch
import base64
import gc
import os


def generate_image(prompt, timesteps, model):
    try:
        model_id = os.path.join(os.path.join(os.getcwd(), "models"), model)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        pipeline = StableDiffusionXLPipeline.from_pretrained(model_id,
                                                             torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        pipeline = pipeline.to(device)

        output = pipeline(prompt, num_inference_steps=timesteps, num_images_per_prompt=1)

        if output and hasattr(output, "images") and output.images:
            return output.images
        else:
            print("Error: No images returned by the pipeline.")
            return None

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

    finally:
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()


def pil_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str
