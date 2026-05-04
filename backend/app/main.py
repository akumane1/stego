import io
import json
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, StreamingResponse
from pydantic import BaseModel
from .steganography import embed_text_dct, extract_text_dct
from .sd_client import generate_image_from_prompt

class GenerateRequest(BaseModel):
    prompt: str
    secret_text: str

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
async def generate_and_hide(request: GenerateRequest):
    try:
        # Fast path: just generate and embed
        image_bytes = generate_image_from_prompt(request.prompt)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_np is None:
            raise HTTPException(status_code=500, detail="Не вдалося декодувати зображення")
            
        stego_img = embed_text_dct(img_np, request.secret_text)
        
        _, buffer = cv2.imencode('.png', stego_img)
        return StreamingResponse(io.BytesIO(buffer), media_type="image/png")
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/test-robustness")
async def test_robustness(file: UploadFile = File(...), secret_text: str = Form(...)):
    """
    Perform on-demand robustness tests on a provided image.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        stego_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if stego_img is None:
            raise HTTPException(status_code=400, detail="Не вдалося прочитати зображення")

        stats = {
            "baseline": False,
            "blur_resilience": False,
            "jpeg_resilience": False
        }
        
        # Test 1: Baseline
        if extract_text_dct(stego_img) == secret_text:
            stats["baseline"] = True
            
        # Test 2: Blur
        blurred = cv2.GaussianBlur(stego_img, (5, 5), 0)
        if extract_text_dct(blurred) == secret_text:
            stats["blur_resilience"] = True
            
        # Test 3: JPEG 80%
        _, encimg = cv2.imencode('.jpg', stego_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        decoded_jpg = cv2.imdecode(encimg, 1)
        if extract_text_dct(decoded_jpg) == secret_text:
            stats["jpeg_resilience"] = True
            
        return stats
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

@app.get("/")
def read_root():
    return {"message": "Stable Diffusion Steganography API is running."}
