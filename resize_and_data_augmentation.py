"""Este programa está hecho para redimensionar y aplicar técnicas de data augmentation a un dataset de
imágenes locales, hecho para entrenar una red neuronal de visión artificial."""


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
INPUT_FOLDER = r"C:\Users\Jose_Enrique\Documents\VSCode Projects\Python\redimensionar_imagenes\pesas de mano"
TEXTURES_FOLDER = r"C:\Users\Jose_Enrique\Documents\VSCode Projects\Python\redimensionar_imagenes\FondosTextura"
OUTPUT_FOLDER = r"C:\Users\Jose_Enrique\Documents\VSCode Projects\Python\redimensionar_imagenes\Fotos exportadas para probar"
DISCARDED_FOLDER = r"C:\Users\Jose_Enrique\Documents\VSCode Projects\Python\redimensionar_imagenes\Descartadas"

TARGET_SIZE = 640
MIN_SIZE = 200
ENLARGE_THRESHOLD = 320

# Nuevo: procesar todo o solo un subconjunto
PROCESS_SUBSET = True
SUBSET_SIZE = 10

# --- CONFIG EXTRA ---
DELAY_MIN = 0.2  # segundos
DELAY_MAX = 1.0
NUM_VUELTAS = 1

# Crear carpetas
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DISCARDED_FOLDER, exist_ok=True)

# Cargar texturas de fondo
TEXTURE_FILES = list(Path(TEXTURES_FOLDER).glob("*"))
if not TEXTURE_FILES:
    raise FileNotFoundError(f"No se encontraron texturas en {TEXTURES_FOLDER}")

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

def get_random_background():
    bg = Image.open(random.choice(TEXTURE_FILES)).convert("RGB")
    return bg.resize((TARGET_SIZE, TARGET_SIZE))

def weighted_angle():
    return float(_rng.uniform(0, 360))

def apply_augmentation(img):
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    return img


def preprocess_size(obj_img):
    w, h = obj_img.size
    if w < MIN_SIZE or h < MIN_SIZE:
        return None
    if w < ENLARGE_THRESHOLD or h < ENLARGE_THRESHOLD:
        resize_factor = random.uniform(2.0, 3.0)
        obj_img = obj_img.resize((int(w * resize_factor), int(h * resize_factor)), Image.LANCZOS)
    return obj_img

def process_image(img_path):
    # Verificar que el archivo sea una imagen válida
    try:
        img = Image.open(img_path).convert("RGBA")
    except (UnidentifiedImageError, OSError):
        print(f"⏩ Descartado (no es imagen o está corrupto): {img_path}")
        return []
    
    w, h = img.size

    if w < MIN_SIZE and h < MIN_SIZE:
        shutil.move(img_path, os.path.join(DISCARDED_FOLDER, os.path.basename(img_path)))
        return []

    results = []
    
    bg = get_random_background().convert("RGBA")

    if w > TARGET_SIZE or h > TARGET_SIZE:
        scale_factor = TARGET_SIZE / max(w, h)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        obj = img.resize((new_w, new_h), Image.LANCZOS)
    else:
        obj = preprocess_size(img.copy())

    if obj is None:
        return []

    # Rotar solo el objeto
    angle = weighted_angle()
    obj = obj.rotate(angle, expand=True)

    # Calcular bounding box de píxeles no transparentes
    alpha = np.array(obj.split()[3])
    ys, xs = np.where(alpha > 0)
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    obj = obj.crop((min_x, min_y, max_x + 1, max_y + 1))

    # Pegar en posición aleatoria sin recortar píxeles visibles
    max_x_offset = TARGET_SIZE - obj.width
    max_y_offset = TARGET_SIZE - obj.height
    pos_x = random.randint(0, max_x_offset) if max_x_offset > 0 else 0
    pos_y = random.randint(0, max_y_offset) if max_y_offset > 0 else 0

    composite = bg.copy()
    composite.paste(obj, (pos_x, pos_y), obj)
    composite = composite.convert("RGB")
    composite = apply_augmentation(composite)

    results.append(composite)

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
                unique_name = make_unique_filename(Path(filename).stem + f"_v{vuelta+1}.jpg", existing_names)
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
