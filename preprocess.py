import numpy as np
from scipy import ndimage
from PIL import Image

def center_digit(img_28x28):
    """Flyttar siffran till mitten av 28x28-rutan baserat på dess masscentrum."""
    cy, cx = ndimage.center_of_mass(img_28x28)
    if np.isnan(cy) or np.isnan(cx): return img_28x28
    return ndimage.shift(img_28x28, [14 - cy, 14 - cx], mode='constant', cval=0)

def preprocess_image(img_input, is_upload=False):
    # Gråskala
    img = img_input.convert('L')
    img_array = np.array(img).astype(np.float32)

    # Identifiera bakgrunden innan vi börjar modifiera pixlarna
    h, w = img_array.shape
    b = 4
    edges_orig = np.concatenate([img_array[:b,:].flatten(), img_array[-b:,:].flatten(),
                                 img_array[:,:b].flatten(), img_array[:, -b:].flatten()])
    
    # Avgör om bakgrunden är ljus baserat på medianen av kanterna
    is_light_bg = np.median(edges_orig) > 127

    # Ljusa upp och städa av ramen
    if is_light_bg:
        # Foto/Ljus bakgrund: Boosta vid behov och gör ramen vit
        if is_upload and np.median(edges_orig) < 250:
            img_array = img_array + 70
            img_array = np.clip(img_array, 0, 255)
        img_array[:b, :] = 255; img_array[-b:, :] = 255
        img_array[:, :b] = 255; img_array[:, -b:] = 255
    else:
        # Redan inverterad/mörk bakgrund: Gör ramen svart
        img_array[:b, :] = 0; img_array[-b:, :] = 0
        img_array[:, :b] = 0; img_array[:, -b:] = 0

    # Enhetlig invertering så att siffran alltid blir vit på svart bakgrund
    if is_light_bg:
        img_array = 255 - img_array

    # Binär mask (Tröskel 100 för att inte tappa tunna ettor)
    threshold = 100 if is_upload else 45
    binary_mask = (img_array > threshold).astype(np.uint8)

    # Cropping
    rows = np.any(binary_mask > 0, axis=1)
    cols = np.any(binary_mask > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return np.zeros((1, 784)), np.zeros((28, 28)), 0, 0, 0

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    digit = img_array[rmin:rmax+1, cmin:cmax+1]

    # Förtjockning & Hålräkning
    digit_bin = (digit > 110).astype(np.uint8)
    digit_bin = ndimage.binary_dilation(digit_bin, structure=np.ones((2,2))).astype(np.uint8)

    labeled_blobs, num_found_blobs = ndimage.label(digit_bin)
    num_holes = 0
    
    # Loopar igenom figurer som hittats
    for label_idx in range(1, num_found_blobs + 1):
        single_blob_mask = (labeled_blobs == label_idx).astype(np.uint8)
        
        # Fyll hål för just denna figur
        filled = ndimage.binary_fill_holes(single_blob_mask)
        holes_mask = filled.astype(int) - single_blob_mask.astype(int)
        labeled_holes, n_found_holes = ndimage.label(holes_mask > 0)
        
        # Räkna giltiga hål i denna figur
        for i in range(1, n_found_holes + 1):
            if np.sum(labeled_holes == i) > 5: # Filter för småbrus
                num_holes += 1

    # Säkerhetsspärr för extremt kladd
    if num_holes > 2:
        num_holes = 0

    # Skalning & Finalisering
    h_d, w_d = digit.shape
    scale = 20.0 / max(h_d, w_d)
    new_size = (int(w_d * scale), int(h_d * scale))
    digit_rescaled = np.array(Image.fromarray((digit_bin * 255).astype(np.uint8)).resize(new_size, Image.Resampling.LANCZOS))
    
    img_28 = np.zeros((28, 28), dtype=np.uint8)
    y_off, x_off = (28 - digit_rescaled.shape[0]) // 2, (28 - digit_rescaled.shape[1]) // 2
    img_28[y_off:y_off+digit_rescaled.shape[0], x_off:x_off+digit_rescaled.shape[1]] = digit_rescaled
    
    img_final = center_digit(img_28.astype('float32')) / 255.0
    img_final = np.clip(img_final, 0.0, 1.0)

    aspect_ratio = (cmax - cmin) / (rmax - rmin) if (rmax - rmin) > 0 else 0
    
    return img_final.reshape(1, 784), img_final, num_found_blobs, aspect_ratio, num_holes