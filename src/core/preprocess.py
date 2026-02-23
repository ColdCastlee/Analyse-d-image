import cv2

def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def denoise(gray, ksize=(7,7), sigma=0):
    return cv2.bilateralFilter(gray, 9, 75, 75) 

def enhance_contrast(gray_blur, clip=2.0, grid=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(gray_blur)