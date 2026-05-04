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
    """
    Embed text by distributing full copies across the entire image area.
    This allows survival even if large parts of the image are cropped.
    """
    img_ycrcb = cv2.cvtColor(image_np, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)
    
    base_bits = text_to_bits(text)
    
    h, w = y.shape
    block_size = 8
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    total_blocks = num_blocks_h * num_blocks_w
    
    # Calculate how many blocks each copy (sector) gets
    sector_size = total_blocks // REDUNDANCY
    
    if len(base_bits) > sector_size:
        raise ValueError(f"Текст занадто довгий. Для {REDUNDANCY}-кратного копіювання ліміт: {int(sector_size / 8)} симв.")

    y_float = y.astype(float)

    # Embed each full copy in its own sector
    for r in range(REDUNDANCY):
        sector_start = r * sector_size
        for bit_idx, bit in enumerate(base_bits):
            # Calculate global block index
            global_idx = sector_start + bit_idx
            
            # Convert global index to 2D block coordinates
            i = global_idx // num_blocks_w
            j = global_idx % num_blocks_w
            
            if i >= num_blocks_h: break
            
            row, col = i * block_size, j * block_size
            block = y_float[row:row+block_size, col:col+block_size]
            block_dct = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            for cy, cx in COEFFS:
                coeff = block_dct[cy, cx]
                if bit == 1:
                    block_dct[cy, cx] = (np.floor(coeff / QUANT_STEP) + 0.5) * QUANT_STEP
                else:
                    block_dct[cy, cx] = np.round(coeff / QUANT_STEP) * QUANT_STEP

            block_idct = idct(idct(block_dct.T, norm='ortho').T, norm='ortho')
            y_float[row:row+block_size, col:col+block_size] = block_idct
            
    y_final = np.clip(y_float, 0, 255).astype(np.uint8)
    merged = cv2.merge([y_final, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

def extract_text_dct(image_np):
    """
    Extract text by voting across geographically different sectors of the image.
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
    
    # 1. Collect raw bits from all sectors
    # Each row in 'sectors_bits' will be one full copy of the message area
    sectors_bits = [[] for _ in range(REDUNDANCY)]
    
    for r in range(REDUNDANCY):
        sector_start = r * sector_size
        # We don't know the message length, so we read the whole sector
        for bit_idx in range(sector_size):
            global_idx = sector_start + bit_idx
            i = global_idx // num_blocks_w
            j = global_idx % num_blocks_w
            if i >= num_blocks_h: break
            
            row, col = i * block_size, j * block_size
            block = y_float[row:row+block_size, col:col+block_size]
            block_dct = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            # Intra-block vote
            block_votes = []
            for cy, cx in COEFFS:
                coeff = block_dct[cy, cx]
                val = (coeff / QUANT_STEP) % 1
                block_votes.append(1 if (0.25 <= val < 0.75) else 0)
            
            sectors_bits[r].append(Counter(block_votes).most_common(1)[0][0])

    # 2. Vote across sectors
    final_bits = []
    for b_idx in range(sector_size):
        votes = []
        for r in range(REDUNDANCY):
            if b_idx < len(sectors_bits[r]):
                votes.append(sectors_bits[r][b_idx])
        
        if votes:
            final_bits.append(Counter(votes).most_common(1)[0][0])
        
        # Stop at null terminator
        if len(final_bits) >= 8 and len(final_bits) % 8 == 0:
            if all(b == 0 for b in final_bits[-8:]):
                break

    return bits_to_text(final_bits)
