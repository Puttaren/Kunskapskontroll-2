import numpy as np
from scipy import ndimage
import cv2

# Hjälpfunktion för att räta upp en siffra som lutar för mycket
# Detta hjälper modellen att fokusera på siffrans form.
def deskew(image):
    m = cv2.moments(image)
    if abs(m['mu02']) < 1e-2:
        return image.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, -skew, 0.5 * 28 * skew], [0, 1, 0]])
    img_deskewed = cv2.warpAffine(image, M, (28, 28), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img_deskewed

def get_digit_bounds(img_array):
    rows = np.any(img_array > 0, axis=1)
    cols = np.any(img_array > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def center_digit(img_28x28):
    cy, cx = ndimage.center_of_mass(img_28x28)
    shift_y = 14 - cy
    shift_x = 14 - cx
    return ndimage.shift(img_28x28, [shift_y, shift_x], mode='constant', cval=0)

def preprocess_image(img_input):
    img_array = np.array(img_input)

    # Om bilden har 3 eller 4 kanaler (RGBA/RGB), gör om till gråskala
    if len(img_array.shape) > 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Kolla hörn-pixeln för att avgöra om vi behöver invertera (MNIST vill ha svart bakgrund)
    if img_array[0, 0] > 127:
        img_array = cv2.bitwise_not(img_array)

    # Tröskelvärde för att få en ren svartvit bild
    _, img_thresh = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Analysera topologi (antal figurer) och form
    labeled_array, num_blobs = ndimage.label(img_thresh)
    bounds = get_digit_bounds(img_thresh)

    if bounds is None:
        return np.zeros((1, 784)), np.zeros((28, 28)), 0, 0

    rmin, rmax, cmin, cmax = bounds
    digit = img_thresh[rmin:rmax+1, cmin:cmax+1]
    
    # Beräkna aspektförhållande (bredd/höjd) för feedback
    aspect_ratio = (cmax - cmin) / (rmax - rmin) if (rmax - rmin) > 0 else 0

    # Skala om till 20x20 (MNIST-standard bevarar proportioner inom denna box)
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    digit_rescaled = cv2.resize(digit, (None, None), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Placera i en 28x28-ram
    h_res, w_res = digit_rescaled.shape
    y_off = (28 - h_res) // 2
    x_off = (28 - w_res) // 2
    img_28x28 = np.zeros((28, 28), dtype=np.uint8)
    img_28x28[y_off:y_off+h_res, x_off:x_off+w_res] = digit_rescaled

    # Centrera baserat på tyngdpunkt (Center of Mass)
    img_final_28x28 = center_digit(img_28x28)

    # Normalisera pixelvärden (0-1)
    img_final_28x28 = img_final_28x28.astype('float32') / 255.0

    # Eftersom den nya modellen är tränad på "raka" siffror rätar vi upp även ritade/uppladdade bilder.
    img_deskewed = deskew(img_final_28x28)

    # Vi plattar till den upprätade bilden för modellen (784 features)
    features = img_deskewed.reshape(1, -1)

    # Nu returneras den upprätade bilden så att "Maskinens vy" 
    # i Streamlit visar exakt vad modellen faktiskt analyserar.
    return features, img_deskewed, num_blobs, aspect_ratio