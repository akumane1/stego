import cv2
import numpy as np
import os
import sys

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.steganography import embed_text_dct, extract_text_dct

def test_steganography_cycle():
    # Create a random image
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    secret_text = "Hello, this is a secret message survive JPEG!"
    
    # Embed
    encoded_img = embed_text_dct(img.copy(), secret_text)
    
    # Extract immediately
    extracted_text = extract_text_dct(encoded_img)
    print(f"Original: {secret_text}")
    print(f"Extracted (Lossless): {extracted_text}")
    assert secret_text == extracted_text

def test_steganography_jpeg_robustness():
    # Create a random image
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    secret_text = "Robust secret message!"
    
    # Embed
    encoded_img = embed_text_dct(img.copy(), secret_text)
    
    # Save as JPEG with quality 90
    jpeg_path = "test_encoded.jpg"
    cv2.imwrite(jpeg_path, encoded_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
    # Load back
    decoded_img = cv2.imread(jpeg_path)
    
    # Extract
    extracted_text = extract_text_dct(decoded_img)
    print(f"Original: {secret_text}")
    print(f"Extracted (JPEG 90): {extracted_text}")
    
    os.remove(jpeg_path)
    assert secret_text == extracted_text

if __name__ == "__main__":
    print("Running tests...")
    try:
        test_steganography_cycle()
        print("Test 1 (Lossless) passed!")
        test_steganography_jpeg_robustness()
        print("Test 2 (JPEG Robustness) passed!")
    except Exception as e:
        print(f"Tests failed: {e}")
        sys.exit(1)
