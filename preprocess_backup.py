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

    # Öka kontrast, ljusa upp bilden för att städa bort skuggor
    im = ImageEnhance.Brightness(im).enhance(1.75)
    im = ImageEnhance.Contrast(im).enhance(1)

    # Invertera och ta bort skuggor
    im = ImageOps.invert(im)
    # im = ImageOps.autocontrast(im, cutoff=20)
    im = im.point(lambda p: p if p > 150 else 0) 
    
    arr = np.array(im)
    h, w = arr.shape

    # Filtrera brus och hitta siffran i bilden
    im_clean = im.filter(ImageFilter.MedianFilter(size=3))
    arr_clean = np.array(im_clean)
    ys, xs = np.where(arr > 100) 

    if len(xs) == 0:
        # Om rutan är tom, skicka bara en tom 28x28
        im28 = im.resize((28, 28), Image.Resampling.LANCZOS)
    else:
        # Skapa en tajt box runt det faktiska innehållet
        # Detta är den "föranalys" du efterfrågade.
        digit = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        # digit = digit.filter(ImageFilter.MaxFilter(3))
   
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
        
        shift_y = 14.5 - cy
        shift_x = 14.5 - cx
        
        arr_centered = ndimage.shift(arr_temp, shift=[shift_y, shift_x])
        im28 = Image.fromarray(arr_centered.astype('uint8'))

    # Bättre svärta
    im28 = ImageOps.autocontrast(im28)

    # Suddighet som i MNIST-filerna
    # im28 = im28.filter(ImageFilter.GaussianBlur(radius=0.3))

    # Returnera som array (1, 784)
    X = np.array(im28).astype("float64") / 255.0
    return X.reshape(1, 784), im28
