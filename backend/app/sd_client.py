import os
import io
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROVIDER = "fal-ai"

def generate_image_from_prompt(prompt: str) -> bytes:
    """
    Calls Stable Diffusion via Hugging Face InferenceClient (fal-ai provider).
    Returns raw PNG image bytes.
    """
    if not HF_API_KEY:
        raise ValueError("HF_API_KEY environment variable is not set in .env file.")

    client = InferenceClient(
        provider=PROVIDER,
        api_key=HF_API_KEY,
    )

    # Returns a PIL Image (requires Pillow, already installed)
    pil_image = client.text_to_image(prompt, model=MODEL_ID)

    # Convert PIL Image → PNG bytes for OpenCV in main.py
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()
