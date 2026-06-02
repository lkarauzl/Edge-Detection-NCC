
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import f1_score
from skimage import data


img = data.camera()

# 2. Anisotropic Diffusion (Perona-Malik) Fonksiyonu
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

# 3. Noise-Calibrated Canny 
def noise_calibrated_canny(image):
    # a. Gürültü Tahmini (Median Absolute Deviation)
    median = np.median(image)
    mad = np.median(np.abs(image - median))
    sigma_est = mad / 0.6745
    
    # b. Ön işleme (Anisotropic Diffusion ile kenar koruyarak gürültü temizleme)
    # Gürültü seviyesine göre iterasyon sayısını ayarla
    iters = int(np.clip(sigma_est / 2, 5, 20))
    smoothed = anisotropic_diffusion(image, niter=iters, kappa=30, gamma=0.15)
    
    # c. Gradyan hesaplama
    gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_normalized = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
    # d. Otsu ile otomatik eşik belirleme
    high_thresh, _ = cv2.threshold(mag_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    
    # e. Canny uygulama
    edges = cv2.Canny(smoothed, low_thresh, high_thresh)
    return edges, high_thresh, low_thresh, sigma_est

# --- DENEY KURULUMU ---
print("Deney başlatılıyor...")

# Orijinal "Temiz" Kenar Haritası (Ground Truth niyetine)
clean_edges = cv2.Canny(cv2.GaussianBlur(img, (5,5), 1.0), 50, 150)

# Görüntüye Sentetik Gürültü Ekleme
noise = np.random.normal(0, 25, img.shape).astype('float32')
noisy_img = cv2.add(img.astype('float32'), noise)
noisy_img = np.clip(noisy_img, 0, 255).astype('uint8')

# 1. Klasik Canny Testi
start_time = time.time()
blurred_noisy = cv2.GaussianBlur(noisy_img, (5,5), 1.0)
baseline_edges = cv2.Canny(blurred_noisy, 50, 150)
baseline_time = time.time() - start_time

# 2. Noise-Calibrated Canny (NCC) Testi
start_time = time.time()
ncc_edges, th, tl, noise_est = noise_calibrated_canny(noisy_img)
ncc_time = time.time() - start_time

# --- PERFORMANS DEĞERLENDİRMESİ (F1-Score) ---
y_true = (clean_edges > 0).flatten()
y_base = (baseline_edges > 0).flatten()
y_ncc = (ncc_edges > 0).flatten()

f1_base = f1_score(y_true, y_base)
f1_ncc = f1_score(y_true, y_ncc)

print("-" * 30)
print(f"Tahmin Edilen Gürültü Seviyesi (MAD): {noise_est:.2f}")
print(f"NCC Otomatik Eşik Değerleri: Low={tl:.1f}, High={th:.1f}")
print("-" * 30)
print(f"Klasik Canny Çalışma Süresi: {baseline_time:.4f} sn")
print(f"Klasik Canny F1-Skoru: {f1_base:.4f}")
print("-" * 30)
print(f"NCC Çalışma Süresi: {ncc_time:.4f} sn")
print(f"NCC F1-Skoru: {f1_ncc:.4f}")
print("-" * 30)

# --- GÖRSELLEŞTİRME VE KAYDETME ---
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.title("Orijinal Görüntü (Gürültülü)")
plt.imshow(noisy_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Ground Truth (Temiz Kenarlar)")
plt.imshow(clean_edges, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title(f"Klasik Canny\nF1: {f1_base:.3f}")
plt.imshow(baseline_edges, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title(f"Noise-Calibrated Canny (Önerilen)\nF1: {f1_ncc:.3f}")
plt.imshow(ncc_edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig("edge_comparison_results.png", dpi=300)
plt.show()
print("Sonuç grafiği 'edge_comparison_results.png' olarak kaydedildi.")
