import numpy as np
from scipy import ndimage
from PIL import Image
from skimage.morphology import skeletonize

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
    
    # 2. Smart Cropping - Isolerar siffran oavsett var den ritats
    binary_mask = (img_array > 45).astype(np.uint8)
    rows = np.any(binary_mask > 0, axis=1)
    cols = np.any(binary_mask > 0, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return np.zeros((1, 784)), np.zeros((28, 28)), 0, 0, 0

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Klipp ut siffran med marginal
    digit = img_array[rmin:rmax+1, cmin:cmax+1]
    
    # 3. Räkna hål (Topologi) - Används för logisk korrigering i predict.py
    digit_bin = (digit > 120).astype(np.uint8)
    filled = ndimage.binary_fill_holes(digit_bin)
    holes_mask = filled.astype(int) - digit_bin.astype(int)
    _, num_holes = ndimage.label(holes_mask > 0)

    # 4. Skalning till MNIST-storlek (20x20 inuti 28x28)
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    digit_rescaled = np.array(Image.fromarray(digit).resize(new_size, Image.Resampling.LANCZOS))
    
    # Placera i mitten
    img_28 = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - digit_rescaled.shape[0]) // 2
    x_off = (28 - digit_rescaled.shape[1]) // 2
    img_28[y_off:y_off+digit_rescaled.shape[0], x_off:x_off+digit_rescaled.shape[1]] = digit_rescaled
    
    # 5. Sudda ut bilden något för att likna MNIST-stilen
    img_blurred = ndimage.gaussian_filter(img_28.astype('float32'), sigma=0.4)
    img_final = center_digit(img_blurred) / 255.0
    img_final = np.clip(img_final, 0, 1)

    # Vi skickar nu den centrerade bilden direkt till modellen utan deskew!
    aspect_ratio = (cmax - cmin) / (rmax - rmin) if (rmax - rmin) > 0 else 0
    _, num_blobs = ndimage.label(binary_mask)

    return img_final.reshape(1, 784), img_final, num_blobs, aspect_ratio, num_holes