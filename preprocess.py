import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from scipy import ndimage

def preprocess_image(path):
    # 1. Öppna och ljusa upp
    im = Image.open(path).convert("L")
    bright = ImageEnhance.Brightness(im)
    im = bright.enhance(2.0) 

    # 2. Invertera och öka kontrast
    im = ImageOps.invert(im)
    im = ImageOps.autocontrast(im, cutoff=2)

    arr = np.array(im)
    h, w = arr.shape

    # Begränsa sökytan till mitten för att undvika brus vid kanterna
    x_lo, x_hi = int(0.2 * w), int(0.8 * w)
    y_lo, y_hi = int(0.2 * h), int(0.8 * h)

    # Skapa en mask som bara är aktiv i mitten av bilden
    mask_center = np.zeros_like(arr)
    mask_center[y_lo:y_hi, x_lo:x_hi] = arr[y_lo:y_hi, x_lo:x_hi]
    
    # Hitta siffran baserat på det rena mittpartiet
    ys, xs = np.where(mask_center > 100)
    # -----------------------------------------------------------------------

    if len(xs) == 0:
        im28 = im.resize((28, 28), Image.Resampling.LANCZOS)
    else:
        # 3. Beskär tajtare runt siffran
        digit = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        
        # 4. Skala till 20x20 (MNIST-standard)
        w_d, h_d = digit.size
        ratio = 20.0 / max(w_d, h_d)
        new_size = (int(round(w_d * ratio)), int(round(h_d * ratio)))
        digit = digit.resize(new_size, Image.Resampling.LANCZOS)

        # 5. Geometrisk centrering
        temp_im = Image.new("L", (28, 28), 0)
        left = (28 - new_size[0]) // 2
        top = (28 - new_size[1]) // 2
        temp_im.paste(digit, (left, top))
        
        # 6. Finjustering med tyngdpunkt
        arr_temp = np.array(temp_im)
        cy, cx = ndimage.center_of_mass(arr_temp)
        
        shift_y = 14 - cy
        shift_x = 14 - cx
        
        arr_centered = ndimage.shift(arr_temp, shift=[shift_y, shift_x])
        im28 = Image.fromarray(arr_centered.astype('uint8'))

    # 7. Suddighet för att efterlikna MNIST-estetik
    im28 = im28.filter(ImageFilter.GaussianBlur(radius=0.4))

    # 8. Returnera som array (1, 784),
    # normaliserad genom delning med 255.0 till en float64 mellan 0 och 1. 
    X = np.array(im28).astype("float64")
    X = X/255.0
    X = X.reshape(1, 784)

    return X, im28