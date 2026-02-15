import numpy as np
from scipy import ndimage
from PIL import Image

# Deskewing 
def deskew(image):
    c, r = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    m = image.sum()
    if m < 1e-5: return image.copy()
    
    mu_01 = (r * image).sum() / m
    mu_10 = (c * image).sum() / m
    mu_11 = (r * c * image).sum() / m - mu_01 * mu_10
    mu_02 = (r**2 * image).sum() / m - mu_01**2
    
    skew = mu_11 / mu_02
    M = np.array([[1, -skew], [0, 1]])
    offset = np.array([mu_01, mu_10]) - M.dot([mu_01, mu_10])
    
    # Använder affine_transform precis som i notebooken
    return ndimage.affine_transform(image, M, offset=offset)

# Hjälpfunktioner för topologi
def get_digit_bounds(img_array):
    rows = np.any(img_array > 0, axis=1)
    cols = np.any(img_array > 0, axis=0)
    if not np.any(rows) or not np.any(cols): return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def center_digit(img_28x28):
    cy, cx = ndimage.center_of_mass(img_28x28)
    return ndimage.shift(img_28x28, [14 - cy, 14 - cx], mode='constant', cval=0)

def preprocess_image(img_input):
    # Gör om PIL Image till gråskala och numpy array
    img = img_input.convert('L')
    img_array = np.array(img)

    # Invertera om bakgrunden är ljus (MNIST är vit siffra på svart)
    if img_array[0, 0] > 127:
        img_array = 255 - img_array

    # Enkel tröskelvärdeshantering utan OpenCV
    img_thresh = (img_array > 50).astype(np.uint8) * 255

    labeled_array, num_blobs = ndimage.label(img_thresh)
    bounds = get_digit_bounds(img_thresh)

    if bounds is None:
        return np.zeros((1, 784)), np.zeros((28, 28)), 0, 0

    rmin, rmax, cmin, cmax = bounds
    digit = img_thresh[rmin:rmax+1, cmin:cmax+1]
    aspect_ratio = (cmax - cmin) / (rmax - rmin) if (rmax - rmin) > 0 else 0

    # Skala om till 20x20 med Pillow istället för OpenCV
    digit_img = Image.fromarray(digit)
    h, w = digit.shape
    scale = 20 / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    digit_rescaled = np.array(digit_img.resize(new_size, Image.Resampling.LANCZOS))

    # Centrera i 28x28
    h_res, w_res = digit_rescaled.shape
    img_28x28 = np.zeros((28, 28), dtype=np.uint8)
    img_28x28[(28-h_res)//2 : (28-h_res)//2+h_res, (28-w_res)//2 : (28-w_res)//2+w_res] = digit_rescaled
    
    img_final = center_digit(img_28x28).astype('float32') / 255.0
    
    # Upprätningen
    img_deskewed = deskew(img_final)
    features = img_deskewed.reshape(1, -1)

    return features, img_deskewed, num_blobs, aspect_ratio