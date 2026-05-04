import cv2
import numpy as np
from scipy.fftpack import dct, idct
from collections import Counter

# --- ULTRA-ROBUST SPATIAL CONFIG ---
QUANT_STEP = 50      
REDUNDANCY = 5       
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
    img_ycrcb = cv2.cvtColor(image_np, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)
    
    base_bits = text_to_bits(text)
    h, w = y.shape
    block_size = 8
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    total_blocks = num_blocks_h * num_blocks_w
    
    sector_size = total_blocks // REDUNDANCY
    
    if len(base_bits) > sector_size:
        raise ValueError(f"Текст занадто довгий.")

    y_float = y.astype(float)

    for r in range(REDUNDANCY):
        sector_start = r * sector_size
        for bit_idx, bit in enumerate(base_bits):
            global_idx = sector_start + bit_idx
            i = global_idx // num_blocks_w
            j = global_idx % num_blocks_w
            if i >= num_blocks_h: break
            
            row, col = i * block_size, j * block_size
            block = y_float[row:row+block_size, col:col+block_size]
            
            # Using cv2.dct is much faster than scipy for blocks
            block_dct = cv2.dct(block)
            
            for cy, cx in COEFFS:
                coeff = block_dct[cy, cx]
                if bit == 1:
                    block_dct[cy, cx] = (np.floor(coeff / QUANT_STEP) + 0.5) * QUANT_STEP
                else:
                    block_dct[cy, cx] = np.round(coeff / QUANT_STEP) * QUANT_STEP

            y_float[row:row+block_size, col:col+block_size] = cv2.idct(block_dct)
            
    y_final = np.clip(y_float, 0, 255).astype(np.uint8)
    merged = cv2.merge([y_final, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

def extract_text_dct(image_np):
    """
    Optimized extraction: stops early and uses faster cv2.dct.
    """
    img_ycrcb = cv2.cvtColor(image_np, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)
    y_float = y.astype(float)
    
    h, w = y.shape
    block_size = 8
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    total_blocks = num_blocks_h * num_blocks_w
    sector_size = total_blocks // REDUNDANCY
    
    # We will store bits byte-by-byte to allow early exit
    extracted_bits_by_sector = [[] for _ in range(REDUNDANCY)]
    finished_sectors = [False] * REDUNDANCY
    
    # Iterate through blocks but checking for null-terminators frequently
    for bit_idx in range(sector_size):
        if all(finished_sectors): break
        
        for r in range(REDUNDANCY):
            if finished_sectors[r]: continue
            
            global_idx = r * sector_size + bit_idx
            i = global_idx // num_blocks_w
            j = global_idx % num_blocks_w
            if i >= num_blocks_h: 
                finished_sectors[r] = True
                continue
            
            row, col = i * block_size, j * block_size
            block = y_float[row:row+block_size, col:col+block_size]
            
            # cv2.dct is highly optimized C++
            block_dct = cv2.dct(block)
            
            votes = []
            for cy, cx in COEFFS:
                coeff = block_dct[cy, cx]
                val = (coeff / QUANT_STEP) % 1
                votes.append(1 if (0.25 <= val < 0.75) else 0)
            
            bit = 1 if sum(votes) > (len(COEFFS) / 2) else 0
            extracted_bits_by_sector[r].append(bit)
            
            # Early exit check for this sector (null terminator)
            if len(extracted_bits_by_sector[r]) >= 8 and len(extracted_bits_by_sector[r]) % 8 == 0:
                if all(b == 0 for b in extracted_bits_by_sector[r][-8:]):
                    finished_sectors[r] = True

    # Majority vote across sectors for the extracted bits
    max_len = max(len(s) for s in extracted_bits_by_sector)
    final_bits = []
    for i in range(max_len):
        votes = [s[i] for s in extracted_bits_by_sector if i < len(s)]
        if votes:
            final_bits.append(Counter(votes).most_common(1)[0][0])
            
    return bits_to_text(final_bits)
