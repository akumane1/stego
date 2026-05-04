import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64
from fastapi.responses import Response, JSONResponse
from .steganography import embed_text_dct, extract_text_dct, calculate_metrics, run_robustness_tests
from .sd_client import generate_image_from_prompt

app = FastAPI(title="AI Steganography API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    """Health check endpoint for frontend connection status indicator."""
    return {"status": "ok", "message": "Stable Diffusion Steganography API is running."}

@app.post("/api/generate-and-hide")
async def generate_and_hide(prompt: str = Form(...), secret_text: str = Form(...)):
    """
    1. Generate an image from prompt using Stable Diffusion.
    2. Embed the secret text into the generated image using DCT steganography.
    3. Return the encoded image as a PNG.
    """
    try:
        # Generate image via Hugging Face Inference API
        image_bytes = generate_image_from_prompt(prompt)

        # Load into OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_np is None:
            raise HTTPException(status_code=500, detail="Не вдалося декодувати згенероване зображення.")

        # Embed secret text using DCT steganography
        encoded_img = embed_text_dct(img_np, secret_text)

        # Calculate statistics
        metrics = calculate_metrics(img_np, encoded_img)

        # Encode as lossless PNG to preserve the embedded data
        is_success, buffer = cv2.imencode(".png", encoded_img)
        if not is_success:
            raise HTTPException(status_code=500, detail="Не вдалося закодувати результуюче зображення.")

        # Convert to Base64 to return alongside JSON metrics
        encoded_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{encoded_base64}"

        return JSONResponse(content={
            "image": img_data_url,
            "metrics": metrics
        })

    except HTTPException:
        raise
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    """
    Extract hidden text from an uploaded image using DCT steganography.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_np is None:
            raise HTTPException(status_code=400, detail="Не вдалося декодувати завантажене зображення.")

        text = extract_text_dct(img_np)
        return {"text": text}

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/test-robustness")
async def test_robustness(file: UploadFile = File(...), secret_text: str = Form(...)):
    """
    Apply attacks to the uploaded stego-image, extract text, and return results.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_np is None:
            raise HTTPException(status_code=400, detail="Не вдалося декодувати завантажене зображення.")

        results = run_robustness_tests(img_np, secret_text)
        return {"results": results}

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def read_root():
    return {"message": "Stable Diffusion Steganography API is running."}
