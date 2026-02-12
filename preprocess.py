import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from scipy import ndimage


def preprocess_image(path_or_img):

    # Läs in filen/den ritade bilden

    if isinstance(path_or_img, str):
        im = Image.open(path_or_img).convert("L")
    elif isinstance(path_or_img, np.ndarray):
        im = Image.fromarray(path_or_img.astype('uint8')).convert("L")
    else:
        im = path_or_img.convert("L")
    
    arr_original = np.array(im)
    h, w = arr_original.shape
    
    # Bakgrundsanalys - detta vara "nyckeln" till att hantera "dåliga" bilder.
    # Analysera yttre 30 % av bilden för att se om det finns skuggor/brus
    border_size = int(min(h, w) * 0.3)
    
    # Ta pixlar från kanterna (top, bottom, left, right)
    top_border = arr_original[:border_size, :]
    bottom_border = arr_original[-border_size:, :]
    left_border = arr_original[:, :border_size]
    right_border = arr_original[:, -border_size:]
    
    # Beräkna medelvärdet av kanterna
    border_mean = np.mean([
        top_border.mean(),
        bottom_border.mean(),
        left_border.mean(),
        right_border.mean()
    ])
    
    # Om bakgrunden är ljus (> 200) = ren bakgrund, SKIPPA brightness
    # Om bakgrunden är mörk (< 200) = skuggor finns, KÖR brightness
    needs_brightness = border_mean < 200
    
    if needs_brightness:
        # Bilden har skuggor/mörk bakgrund - ljusa upp
        im = ImageEnhance.Brightness(im).enhance(1.75)
        im = ImageEnhance.Contrast(im).enhance(1.2)

    # Invertera och ta bort skuggor
    im = ImageOps.invert(im)
    
    # Anpassa threshold baserat på om vi körde brightness
    threshold_value = 130 if needs_brightness else 110
    im = im.point(lambda p: p if p > threshold_value else 0)
    
    arr = np.array(im)

    # Filtrera brus och hitta siffran i bilden
    im_clean = im.filter(ImageFilter.MedianFilter(size=3))
    arr_clean = np.array(im_clean)
    ys, xs = np.where(arr_clean > 100)

    if len(xs) == 0:
        # Om rutan är tom, skicka bara en tom 28x28
        im28 = Image.new("L", (28, 28), 0)
    else:
        # Lägg till marginal för att få hela siffran
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        
        # Lägg till 5 % marginal
        margin_x = max(1, int((x_max - x_min) * 0.05))
        margin_y = max(1, int((y_max - y_min) * 0.05))
        
        x_min = max(0, x_min - margin_x)
        x_max = min(w - 1, x_max + margin_x)
        y_min = max(0, y_min - margin_y)
        y_max = min(h - 1, y_max + margin_y)
        
        # Skapa en tajt box runt det faktiska innehållet
        digit = im_clean.crop((x_min, y_min, x_max + 1, y_max + 1))
   
        # Skala till 20x20 (MNIST-standard)
        w_d, h_d = digit.size
        ratio = 20.0 / max(w_d, h_d)
        new_size = (int(round(w_d * ratio)), int(round(h_d * ratio)))
        digit = digit.resize(new_size, Image.Resampling.LANCZOS)

        # Geometrisk centrering
        temp_im = Image.new("L", (28, 28), 0)
        left = (28 - new_size[0]) // 2
        top = (28 - new_size[1]) // 2
        temp_im.paste(digit, (left, top))
        
        # Finjustering med tyngdpunkt
        arr_temp = np.array(temp_im)
        
        # Kontrollera att det finns innehåll
        if arr_temp.sum() > 0:
            cy, cx = ndimage.center_of_mass(arr_temp)
            
            shift_y = 14.0 - cy
            shift_x = 14.0 - cx
            
            # Begränsa shift - lite mer generöst för vertikal justering
            shift_y = np.clip(shift_y, -4, 4)  # Ökat från ±3 till ±4
            shift_x = np.clip(shift_x, -3, 3)
            
            arr_centered = ndimage.shift(arr_temp, shift=[shift_y, shift_x], mode='constant', cval=0)
        else:
            arr_centered = arr_temp
            
        im28 = Image.fromarray(arr_centered.astype('uint8'))

    # Bättre svärta
    im28 = ImageOps.autocontrast(im28)

    # Suddighet som i MNIST-filerna
    im28 = im28.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Returnera som array (1, 784)
    X = np.array(im28).astype("float64") / 255.0
    return X.reshape(1, 784), im28
