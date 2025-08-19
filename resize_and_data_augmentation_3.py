"""
Este programa está hecho para redimensionar y aplicar técnicas de data augmentation a un dataset de
imágenes locales, hecho para entrenar una red neuronal de visión artificial.
"""

import os
import random
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, UnidentifiedImageError
import shutil
import math

# --- CONFIG ---
INPUT_FOLDER = r"C:\Users\Jose_Enrique\Documents\VSCode Projects\Python\redimensionar_imagenes\Fotos aumentar"
OUTPUT_FOLDER = r"C:\Users\Jose_Enrique\Documents\VSCode Projects\Python\redimensionar_imagenes\Exportadas con data augmentation"

# Nuevo: procesar todo o solo un subconjunto
PROCESS_SUBSET = False
SUBSET_SIZE = 10

# --- CONFIG EXTRA ---
DELAY_MIN = 0.2  # segundos
DELAY_MAX = 1.0
NUM_VUELTAS = 1

# Crear carpetas
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Generador global de números random con NumPy
_rng = np.random.default_rng()


# --- FUNCIONES ---

def make_unique_filename(filename, existing_names):
    name, ext = os.path.splitext(filename)
    if not ext:
        ext = ".jpg"
    counter = 1
    new_name = name + ext
    while new_name in existing_names:
        new_name = f"{name}_{counter}{ext}"
        counter += 1
    existing_names.add(new_name)
    return new_name


def weighted_angle():
    return float(_rng.uniform(0, 360))


def make_square_crop(img):
    """Recorta la imagen a un cuadrado centrado"""
    w, h = img.size
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    right = left + min_side
    bottom = top + min_side
    return img.crop((left, top, right, bottom))


def apply_augmentation(img):
    """Aplica varios cambios de color, brillo, contraste, etc."""
    # Brillo
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1))
    # Contraste
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    # Saturación/Color
    img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
    # Nitidez
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.7, 1.5))

    # Aleatoriamente aplicar blur ligero o detalle extra
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    elif random.random() < 0.3:
        img = img.filter(ImageFilter.DETAIL)

    return img


def preprocess_size(obj_img):
    """Redimensionamiento intermedio antes del paso final"""
    w, h = obj_img.size
    resize_factor = random.uniform(1, 1.5)
    obj_img = obj_img.resize((int(w * resize_factor), int(h * resize_factor)), Image.LANCZOS)
    return obj_img


def process_image(img_path):
    """Procesa la imagen aplicando augmentations y redimensionando"""
    try:
        img = Image.open(img_path).convert("RGBA")
    except (UnidentifiedImageError, OSError):
        print(f"⏩ Descartado (no es imagen o está corrupto): {img_path}")
        return []

    results = []
    
    # --- 1) Recortar a cuadrado ---
    img = make_square_crop(img)

    # --- 2) Rotación + augmentations + resize intermedio ---
    angle = weighted_angle()
    img = img.rotate(angle, expand=True)
    img = apply_augmentation(img)
    img = preprocess_size(img)

    # --- 3) Pegar en fondo negro y redimensionar a 640x640 ---
    final_canvas = Image.new("RGB", (640, 640), (0, 0, 0))  # fondo negro
    # Ajustar imagen para que quepa dentro de 640x640 sin deformar
    img.thumbnail((640, 640), Image.LANCZOS)
    # Centrar en el canvas
    x = (640 - img.size[0]) // 2
    y = (640 - img.size[1]) // 2
    final_canvas.paste(img, (x, y), img if img.mode == "RGBA" else None)

    results.append(final_canvas)

    return results


# Lista inicial de archivos
all_files = [f for f in os.listdir(INPUT_FOLDER) if os.path.isfile(os.path.join(INPUT_FOLDER, f))]
if PROCESS_SUBSET:
    all_files = random.sample(all_files, min(SUBSET_SIZE, len(all_files)))

existing_names = set()
total_imgs = len(all_files) * NUM_VUELTAS
procesadas = 0

start_time = time.time()

# Múltiples vueltas
for vuelta in range(NUM_VUELTAS):
    for filename in all_files:
        file_path = os.path.join(INPUT_FOLDER, filename)

        processed_images = process_image(file_path)
        if processed_images:
            for img in processed_images:
                unique_name = make_unique_filename(Path(filename).stem + f"_v{vuelta+1}.png", existing_names)
                img.save(os.path.join(OUTPUT_FOLDER, unique_name), quality=95)
        
        procesadas += 1
        restantes = total_imgs - procesadas
        porcentaje = (procesadas / total_imgs) * 100

        # Calcular tiempo estimado restante
        elapsed = time.time() - start_time
        avg_time_per_img = elapsed / procesadas
        eta_seconds = avg_time_per_img * restantes
        eta_min = int(eta_seconds // 60)
        eta_sec = int(eta_seconds % 60)

        print(f"[{procesadas}/{total_imgs}] Restantes: {restantes} ({porcentaje:.1f}%) | "
              f"ETA: {eta_min}m {eta_sec}s")

        # Delay aleatorio entre cada imagen
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

print(f"\n✅ Proceso finalizado en {int((time.time() - start_time)//60)}m {int((time.time() - start_time)%60)}s. "
      f"Imágenes generadas en: {OUTPUT_FOLDER}")
