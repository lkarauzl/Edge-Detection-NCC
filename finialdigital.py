import cv2
import numpy as np
from skimage import data, color
from sklearn.metrics import f1_score

def anisotropic_diffusion(img, niter=15, kappa=30, gamma=0.15):
    img = img.astype('float32')
    img_out = img.copy()
    for _ in range(niter):
        deltaN = np.roll(img_out, -1, axis=0) - img_out
        deltaS = np.roll(img_out, 1, axis=0) - img_out
        deltaE = np.roll(img_out, -1, axis=1) - img_out
        deltaW = np.roll(img_out, 1, axis=1) - img_out
        cN = np.exp(-(deltaN/kappa)**2); cS = np.exp(-(deltaS/kappa)**2)
        cE = np.exp(-(deltaE/kappa)**2); cW = np.exp(-(deltaW/kappa)**2)
        img_out += gamma * (cN*deltaN + cS*deltaS + cE*deltaE + cW*deltaW)
    return img_out.astype('uint8')

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
    ret_high_thresh, _ = cv2.threshold(mag_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * ret_high_thresh
    edges = cv2.Canny(smoothed, low_thresh, ret_high_thresh)
    return edges

# 3 Farklı Test Resmi
images = {
    'Camera': data.camera(),
    'Astronaut': (color.rgb2gray(data.astronaut()) * 255).astype(np.uint8),
    'Chelsea': (color.rgb2gray(data.chelsea()) * 255).astype(np.uint8)
}

noise_levels = [10, 25, 50, 75]
num_runs = 5
kappa_fixed = 30

print("--- ORTALAMA VE STANDART SAPMA SONUÇLARI ---")
for sigma in noise_levels:
    canny_f1s = []
    ncc_f1s = []

    for img_name, img in images.items():
        clean_edges = cv2.Canny(cv2.GaussianBlur(img, (5,5), 1.0), 50, 150)
        y_true = (clean_edges > 0).flatten()

        for _ in range(num_runs):
            noise = np.random.normal(0, sigma, img.shape).astype('float32')
            noisy_img = np.clip(cv2.add(img.astype('float32'), noise), 0, 255).astype('uint8')

            # Baseline Canny
            blurred_noisy = cv2.GaussianBlur(noisy_img, (5,5), 1.0)
            base_edges = cv2.Canny(blurred_noisy, 50, 150)
            canny_f1s.append(f1_score(y_true, (base_edges > 0).flatten()))

            # NCC
            ncc_edges = noise_calibrated_canny(noisy_img, kappa_val=kappa_fixed)
            ncc_f1s.append(f1_score(y_true, (ncc_edges > 0).flatten()))

    print(f"Noise Sigma: {sigma}")
    print(f"  Canny F1: {np.mean(canny_f1s):.3f} ± {np.std(canny_f1s):.3f}")
    print(f"  NCC F1:   {np.mean(ncc_f1s):.3f} ± {np.std(ncc_f1s):.3f}")
