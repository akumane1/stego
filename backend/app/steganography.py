import cv2
import numpy as np
from scipy.fftpack import dct, idct
from collections import Counter
import base64
import difflib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

# --- EXTREME ROBUSTNESS CONFIG ---
# We use 5x redundancy and a very large step to survive heavy blur
QUANT_STEP = 50      
REDUNDANCY = 5       
# Low frequencies [1,2] and [2,1] are the most robust against filters/blur
COEFFS = [(1, 2), (2, 1)] 

def text_to_bits(text):
    bits = []
    for char in text:
        bin_char = bin(ord(char))[2:].zfill(8)
        bits.extend([int(b) for b in bin_char])
    bits.extend([0] * 8)
    return bits

def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8: break
        char_code = int("".join(map(str, byte)), 2)
        if char_code == 0: break
        chars.append(chr(char_code))
    return "".join(chars)

def embed_text_dct(image_np, text):
    """
    Ultra-robust embedding using low frequencies and high redundancy.
    """
    img_ycrcb = cv2.cvtColor(image_np, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)
    
    base_bits = text_to_bits(text)
    redundant_bits = []
    for bit in base_bits:
        redundant_bits.extend([bit] * REDUNDANCY)
    
    h, w = y.shape
    block_size = 8
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    
    # We use two coefficients per block to spread the energy
    if len(redundant_bits) > num_blocks_h * num_blocks_w:
        raise ValueError(f"Занадто довгий текст. Максимум: {int((num_blocks_h * num_blocks_w) / REDUNDANCY / 8)} симв.")

    y_float = y.astype(float)
    bit_idx = 0

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            if bit_idx >= len(redundant_bits): break
                
            row, col = i * block_size, j * block_size
            block = y_float[row:row+block_size, col:col+block_size]
            block_dct = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            bit = redundant_bits[bit_idx]
            
            # Apply QIM to two coefficients for double protection
            for cy, cx in COEFFS:
                coeff = block_dct[cy, cx]
                if bit == 1:
                    block_dct[cy, cx] = (np.floor(coeff / QUANT_STEP) + 0.5) * QUANT_STEP
                else:
                    block_dct[cy, cx] = np.round(coeff / QUANT_STEP) * QUANT_STEP

            block_idct = idct(idct(block_dct.T, norm='ortho').T, norm='ortho')
            y_float[row:row+block_size, col:col+block_size] = block_idct
            bit_idx += 1
            
    y_final = np.clip(y_float, 0, 255).astype(np.uint8)
    merged = cv2.merge([y_final, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

def extract_text_dct(image_np):
    """
    Extraction with multi-coefficient voting and high redundancy.
    """
    img_ycrcb = cv2.cvtColor(image_np, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)
    y_float = y.astype(float)
    
    h, w = y.shape
    block_size = 8
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    
    all_raw_bits = []
    
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            row, col = i * block_size, j * block_size
            block = y_float[row:row+block_size, col:col+block_size]
            block_dct = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            # Extract from both coefficients and take the best guess
            votes_for_this_block = []
            for cy, cx in COEFFS:
                coeff = block_dct[cy, cx]
                val = (coeff / QUANT_STEP) % 1
                votes_for_this_block.append(1 if (0.25 <= val < 0.75) else 0)
            
            # Majority vote within the block itself
            all_raw_bits.append(Counter(votes_for_this_block).most_common(1)[0][0])

    # Final majority vote across redundant copies
    final_bits = []
    for i in range(0, len(all_raw_bits), REDUNDANCY):
        votes = all_raw_bits[i : i + REDUNDANCY]
        if len(votes) < REDUNDANCY: break
        
        final_bits.append(Counter(votes).most_common(1)[0][0])
        
        if len(final_bits) >= 8 and len(final_bits) % 8 == 0:
            if all(b == 0 for b in final_bits[-8:]):
                break

    return bits_to_text(final_bits)

def calculate_metrics(original_img, stego_img):
    """
    Calculate statistical metrics (MSE, PSNR, SSIM, Capacity) 
    between the original image and the steganographic image.
    """
    # Convert BGR to Grayscale for SSIM (standard practice)
    orig_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    stego_gray = cv2.cvtColor(stego_img, cv2.COLOR_BGR2GRAY)

    # Calculate MSE
    err_mse = mse(orig_gray, stego_gray)
    
    # Calculate PSNR
    val_psnr = psnr(orig_gray, stego_gray, data_range=255)
    
    # Calculate SSIM
    val_ssim = ssim(orig_gray, stego_gray, data_range=255)
    
    # Calculate Capacity (in characters/bytes)
    h, w = orig_gray.shape
    num_blocks_h = h // 8
    num_blocks_w = w // 8
    capacity = (num_blocks_h * num_blocks_w) // REDUNDANCY // 8
    
    return {
        "mse": round(err_mse, 4),
        "psnr": round(val_psnr, 2),
        "ssim": round(val_ssim, 4),
        "capacity": capacity
    }

def run_robustness_tests(stego_img, original_text):
    """
    Applies several attacks to the stego image, attempts extraction,
    and returns visual and statistical results.
    """
    results = []

    # 1. Gaussian Blur (5x5)
    img_blur = cv2.GaussianBlur(stego_img, (5, 5), 0)
    results.append(_evaluate_attack("Розмиття (Gaussian Blur 5x5)", img_blur, original_text))

    # 2. Gaussian Noise
    noise = np.random.normal(0, 15, stego_img.shape).astype(np.float32)
    img_noise = cv2.add(stego_img.astype(np.float32), noise)
    img_noise = np.clip(img_noise, 0, 255).astype(np.uint8)
    results.append(_evaluate_attack("Шум (Gaussian Noise)", img_noise, original_text))

    # 3. JPEG Compression (Quality 50%)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    _, encoded_jpg = cv2.imencode('.jpg', stego_img, encode_param)
    img_jpeg = cv2.imdecode(encoded_jpg, cv2.IMREAD_COLOR)
    results.append(_evaluate_attack("Стиснення (JPEG 50%)", img_jpeg, original_text))

    # 4. Brightness (+30)
    img_bright = cv2.convertScaleAbs(stego_img, alpha=1.0, beta=30)
    results.append(_evaluate_attack("Зміна яскравості (+30)", img_bright, original_text))

    return results

def _evaluate_attack(attack_name, attacked_img, original_text):
    """Helper to extract text, calculate survival, and encode thumbnail."""
    try:
        extracted_text = extract_text_dct(attacked_img)
    except Exception:
        extracted_text = ""

    # Calculate match percentage
    if original_text and extracted_text:
        match_ratio = difflib.SequenceMatcher(None, original_text, extracted_text).ratio()
    else:
        match_ratio = 0.0

    # Create thumbnail for UI (resize to save bandwidth)
    thumb = cv2.resize(attacked_img, (200, 200))
    _, buffer = cv2.imencode(".jpg", thumb)
    b64_thumb = base64.b64encode(buffer.tobytes()).decode('utf-8')

    return {
        "name": attack_name,
        "image": f"data:image/jpeg;base64,{b64_thumb}",
        "extractedText": extracted_text,
        "survivalRate": round(match_ratio * 100, 1)
    }


