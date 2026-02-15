import numpy as np
from scipy import ndimage
from PIL import Image
from skimage.morphology import skeletonize

def deskew(image):
    # Exakt logik från din notebook cell 11
    c, r = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    m = image.sum()
    if m < 1e-5: return image.copy()
    
    mu_01 = (r * image).sum() / m
    mu_10 = (c * image).sum() / m
    mu_11 = (r * c * image).sum() / m - mu_01 * mu_10
    mu_02 = (r**2 * image).sum() / m - mu_01**2
    
    skew = mu_11 / mu_02
    # Vi dämpar deskew något (0.7) för att inte deformera sexans nacke för mycket
    M = np.array([[1, -(skew * 0.7)], [0, 1]])
    offset = np.array([mu_01, mu_10]) - M.dot([mu_01, mu_10])
    
    return ndimage.affine_transform(image, M, offset=offset)

def center_digit(img_28x28):
    # Stabil centrering baserat på masscentrum
    cy, cx = ndimage.center_of_mass(img_28x28)
    if np.isnan(cy) or np.isnan(cx): return img_28x28
    return ndimage.shift(img_28x28, [14 - cy, 14 - cx], mode='constant', cval=0)

def preprocess_image(img_input):
    # 1. Gråskala och Invertering
    img = img_input.convert('L')
    img_array = np.array(img)
    if img_array[0, 0] > 127: img_array = 255 - img_array
    
    # 2. Smart Cropping - Vi isolerar siffran oavsett var den ritats
    binary_mask = (img_array > 45).astype(np.uint8)
    rows = np.any(binary_mask > 0, axis=1)
    cols = np.any(binary_mask > 0, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return np.zeros((1, 784)), np.zeros((28, 28)), 0, 0, 0

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Klipp ut siffran med en liten marginal (padding)
    digit = img_array[rmin:rmax+1, cmin:cmax+1]
    
    # 3. Räkna hål (Topologi) - En 6:a har 1 hål, en 5:a har 0
    # Vi gör detta på den utklippta siffran för bäst precision
    digit_bin = (digit > 120).astype(np.uint8)
    skeleton = skeletonize(digit_bin)
    # Fyller hål och jämför för att hitta slutna cirklar
    filled = ndimage.binary_fill_holes(digit_bin)
    holes_mask = filled.astype(int) - digit_bin.astype(int)
    _, num_holes = ndimage.label(holes_mask > 0)

    # 4. Skalning till MNIST-storlek (20x20 inuti 28x28)
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    digit_rescaled = np.array(Image.fromarray(digit).resize(new_size, Image.Resampling.LANCZOS))
    
    # Placera i mitten av en 28x28 matris
    img_28 = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - digit_rescaled.shape[0]) // 2
    x_off = (28 - digit_rescaled.shape[1]) // 2
    img_28[y_off:y_off+digit_rescaled.shape[0], x_off:x_off+digit_rescaled.shape[1]] = digit_rescaled
    
    # 5. Sudda ut bilden (Gaussian Blur)
    # Detta gör att din skarpa digitala ritning liknar MNIST-datats "luddiga" stil
    img_blurred = ndimage.gaussian_filter(img_28.astype('float32'), sigma=0.4)
    img_final = center_digit(img_blurred) / 255.0
    
    # 6. Deskew och Normalisering
    img_deskewed = deskew(img_final)
    img_deskewed = np.clip(img_deskewed, 0, 1)

    aspect_ratio = (cmax - cmin) / (rmax - rmin) if (rmax - rmin) > 0 else 0
    _, num_blobs = ndimage.label(binary_mask)

    return img_deskewed.reshape(1, 784), img_deskewed, num_blobs, aspect_ratio, num_holes