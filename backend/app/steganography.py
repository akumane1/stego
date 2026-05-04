import cv2
import numpy as np
from scipy.fftpack import dct, idct
from collections import Counter
import base64
import difflib
import hashlib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

# We use very high redundancy and a massive step to survive heavy blur and noise
QUANT_STEP = 60      
REDUNDANCY = 11       
# Lowest AC frequencies [0,1], [1,0], [1,1] are virtually immune to standard blur
COEFFS = [(2, 1)] 

def text_to_bits(text):
    bits = []
    bytes_data = text.encode('utf-8')
    for b in bytes_data:
        bin_b = bin(b)[2:].zfill(8)
        bits.extend([int(bit) for bit in bin_b])
    bits.extend([0] * 8)
    return bits

def bits_to_text(bits):
    byte_vals = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8: break
        val = int("".join(map(str, byte_bits)), 2)
        if val == 0: break
        byte_vals.append(val)
    
    try:
        return bytes(byte_vals).decode('utf-8')
    except UnicodeDecodeError:
        return bytes(byte_vals).decode('utf-8', errors='replace')

def embed_text_dct(img_np, text, password=""):
    """
    Embed text into image using robust DCT with majority voting, double-coeff storage,
    and optional cryptographic block shuffling.
    """
    img_ycc = cv2.cvtColor(img_np, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img_ycc)

    # 1. Convert text to bits
    bits = text_to_bits(text)

    # 2. Add terminator and redundancy
    terminator = [0]*8
    msg_bits = bits + terminator
    redundant_bits = []
    for b in msg_bits:
        redundant_bits.extend([b] * REDUNDANCY)

    h, w = Y.shape
    num_blocks_h = h // 8
    num_blocks_w = w // 8
    total_blocks = num_blocks_h * num_blocks_w

    if len(redundant_bits) > total_blocks:
        raise ValueError("Text is too long for this image size with current redundancy.")

    # 3. Generate Block Indices (Shuffled if password provided)
    indices = np.arange(total_blocks)
    if password:
        seed = int(hashlib.sha256(password.encode()).hexdigest(), 16) % (2**32)
        np.random.RandomState(seed).shuffle(indices)

    # 4. Embed bits
    for bit_idx, bit in enumerate(redundant_bits):
        block_idx = indices[bit_idx]
        i = block_idx // num_blocks_w
        j = block_idx % num_blocks_w

        block = Y[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32)
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
        
        for cy, cx in COEFFS:
            coeff = dct_block[cy, cx]
            if bit == 1:
                dct_block[cy, cx] = (np.floor(coeff / QUANT_STEP) + 0.5) * QUANT_STEP
            else:
                dct_block[cy, cx] = np.round(coeff / QUANT_STEP) * QUANT_STEP
        
        inv_dct = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
        Y[i*8:(i+1)*8, j*8:(j+1)*8] = np.clip(inv_dct, 0, 255)

    img_ycc_encoded = cv2.merge([Y, Cr, Cb])
    return cv2.cvtColor(img_ycc_encoded, cv2.COLOR_YCrCb2BGR)

def extract_text_dct(img_np, password=""):
    """
    Extract text using majority voting, double-coeff logic, and optional block unshuffling.
    """
    img_ycc = cv2.cvtColor(img_np, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img_ycc)
    
    h, w = Y.shape
    num_blocks_h = h // 8
    num_blocks_w = w // 8
    total_blocks = num_blocks_h * num_blocks_w

    # Generate Block Indices
    indices = np.arange(total_blocks)
    if password:
        seed = int(hashlib.sha256(password.encode()).hexdigest(), 16) % (2**32)
        np.random.RandomState(seed).shuffle(indices)

    all_raw_bits = []
    # Read blocks in the exact same sequence they were written
    for block_idx in indices:
        i = block_idx // num_blocks_w
        j = block_idx % num_blocks_w

        block = Y[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32)
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
        
        # Extract from both coefficients and take the best guess
        votes_for_this_block = []
        for cy, cx in COEFFS:
            coeff = dct_block[cy, cx]
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

def run_robustness_tests(stego_img, original_text, password=""):
    """
    Applies several attacks to the stego image, attempts extraction,
    and returns visual and statistical results.
    """
    results = []

    # 1. Gaussian Blur (5x5)
    img_blur = cv2.GaussianBlur(stego_img, (5, 5), 0)
    results.append(_evaluate_attack("Розмиття (Gaussian Blur 5x5)", img_blur, original_text, password))

    # 2. Gaussian Noise
    noise = np.random.normal(0, 15, stego_img.shape).astype(np.float32)
    img_noise = cv2.add(stego_img.astype(np.float32), noise)
    img_noise = np.clip(img_noise, 0, 255).astype(np.uint8)
    results.append(_evaluate_attack("Шум (Gaussian Noise)", img_noise, original_text, password))

    # 3. JPEG Compression (Quality 50%)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    _, encoded_jpg = cv2.imencode('.jpg', stego_img, encode_param)
    img_jpeg = cv2.imdecode(encoded_jpg, cv2.IMREAD_COLOR)
    results.append(_evaluate_attack("Стиснення (JPEG 50%)", img_jpeg, original_text, password))

    # 4. Brightness (+30)
    img_bright = cv2.convertScaleAbs(stego_img, alpha=1.0, beta=30)
    results.append(_evaluate_attack("Зміна яскравості (+30)", img_bright, original_text, password))

    return results

def _evaluate_attack(attack_name, attacked_img, original_text, password):
    """Helper to extract text, calculate survival, and encode thumbnail."""
    try:
        extracted_text = extract_text_dct(attacked_img, password)
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


