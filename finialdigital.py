import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import f1_score
from skimage import data

# 1. Anisotropic Diffusion
def anisotropic_diffusion(img, niter=15, kappa=20, gamma=0.1):
    img = img.astype('float32')
    img_out = img.copy()
    for _ in range(niter):
        deltaN = np.roll(img_out, -1, axis=0) - img_out
        deltaS = np.roll(img_out, 1, axis=0) - img_out
        deltaE = np.roll(img_out, -1, axis=1) - img_out
        deltaW = np.roll(img_out, 1, axis=1) - img_out
        
        cN = np.exp(-(deltaN/kappa)**2)
        cS = np.exp(-(deltaS/kappa)**2)
        cE = np.exp(-(deltaE/kappa)**2)
        cW = np.exp(-(deltaW/kappa)**2)
        
        img_out += gamma * (cN*deltaN + cS*deltaS + cE*deltaE + cW*deltaW)
    return img_out.astype('uint8')

# 2. Noise-Calibrated Canny
def noise_calibrated_canny(image, kappa_val=30):
    median = np.median(image)
    mad = np.median(np.abs(image - median))
    sigma_est = mad / 0.6745
    
    iters = int(np.clip(sigma_est / 2, 5, 20))
    smoothed = anisotropic_diffusion(image, niter=iters, kappa=kappa_val, gamma=0.15)
    
    gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_normalized = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
    # DÜZELTME: OpenCV threshold dönüş isimlendirmesi
    ret_high_thresh, _ = cv2.threshold(mag_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * ret_high_thresh
    
    edges = cv2.Canny(smoothed, low_thresh, ret_high_thresh)
    return edges, ret_high_thresh, low_thresh, sigma_est

# --- ABLATION STUDY DÖNGÜSÜ (Tablo 1'i Üretmek İçin) ---
img = data.camera()
clean_edges = cv2.Canny(cv2.GaussianBlur(img, (5,5), 1.0), 50, 150)
y_true = (clean_edges > 0).flatten()

noise_levels = [10, 25, 50, 75]
kappa_values = [15, 30, 45]

print("--- ABLATION STUDY SONUÇLARI ---")
for sigma in noise_levels:
    noise = np.random.normal(0, sigma, img.shape).astype('float32')
    noisy_img = np.clip(cv2.add(img.astype('float32'), noise), 0, 255).astype('uint8')
    
    blurred_noisy = cv2.GaussianBlur(noisy_img, (5,5), 1.0)
    base_edges = cv2.Canny(blurred_noisy, 50, 150)
    f1_base = f1_score(y_true, (base_edges > 0).flatten())
    
    print(f"\nNoise Sigma: {sigma} | Canny (Fixed) F1: {f1_base:.3f}")
    
    for k in kappa_values:
        ncc_edges, _, _, _ = noise_calibrated_canny(noisy_img, kappa_val=k)
        f1_ncc = f1_score(y_true, (ncc_edges > 0).flatten())
        print(f"  NCC (kappa={k}) F1: {f1_ncc:.3f}")
