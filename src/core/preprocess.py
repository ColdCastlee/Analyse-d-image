import cv2

def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def denoise(gray, ksize=(9, 9), sigma=0):
    return cv2.GaussianBlur(gray, ksize, sigma)

def enhance_contrast(gray_blur, clip=3.0, grid=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(gray_blur)