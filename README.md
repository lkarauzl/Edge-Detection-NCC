# Edge Detection with Scale-Space Denoising and Automatic Thresholding

This repository contains the Python implementation of the **Noise-Calibrated Canny (NCC)** edge detection pipeline. This project was developed as a term project for the Computer Engineering department at Abdullah Gül University (AGÜ).

## Project Overview
The traditional Canny edge detector is highly sensitive to manual parameter tuning (Gaussian blur variance and hysteresis thresholds). The NCC pipeline aims to automate this process to produce stable edge maps under varying noise and contrast conditions.

The proposed pipeline consists of four main automated steps:
1. **Noise Estimation:** Blindly estimates the image noise level using Median Absolute Deviation (MAD).
2. **Adaptive Denoising:** Applies Perona-Malik Anisotropic Diffusion to smooth the image while preserving structural boundaries. The number of diffusion iterations scales dynamically with the estimated noise.
3. **Gradient Computation:** Calculates spatial gradients using Sobel operators and applies Non-Maximum Suppression (NMS).
4. **Automatic Thresholding:** Utilizes Otsu's method on the gradient magnitude histogram to dynamically determine the optimal high and low hysteresis thresholds.

## Requirements
To run the code, you need Python 3.x and the following libraries installed:
* `opencv-python` (cv2)
* `numpy`
* `matplotlib`
* `scikit-learn`
* `scikit-image`

You can install the dependencies using pip:
```bash
pip install opencv-python numpy matplotlib scikit-learn scikit-image
