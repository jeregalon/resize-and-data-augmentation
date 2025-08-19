"""Programa para quitar fondo blanco a una serie de imágenes locales"""

import os
from pathlib import Path
import numpy as np
import cv2
import random

# --- CONFIG ---
INPUT_FOLDERS = [
    r"C:\Users\Jose_Enrique\Downloads\Datasets\Imágenes\Imágenes Google original",
    r"C:\Users\Jose_Enrique\Downloads\Datasets\Imágenes\Imágenes Google 2 original"
]
OUTPUT_FOLDER = r"C:\Users\Jose_Enrique\Documents\VSCode Projects\Python\redimensionar_imagenes\Exportadas sin fondo 2"

WHITE_THRESH = 220   # umbral para detectar 'blanco candidato'
GRABCUT_ITERS = 5
MIN_SIZE = 250  # tamaño mínimo para procesar

# Procesar todo o solo una parte
PROCESS_SUBSET = True
SUBSET_SIZE = 50

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def make_unique_filename(base_name, existing):
    """Evita sobrescribir archivos: añade _1, _2, ... si hace falta."""
    name = base_name
    stem = Path(base_name).stem
    ext = Path(base_name).suffix or ".png"
    i = 1
    while name in existing:
        name = f"{stem}_{i}{ext}"
        i += 1
    existing.add(name)
    return name


def imread_utf8(path):
    """Lee imagen desde rutas con acentos devolviendo BGR (como cv2.imread)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def save_png_bytes(path, img_bgra):
    """Guarda imagen BGRA como PNG en ruta Unicode."""
    success, buf = cv2.imencode('.png', img_bgra)
    if not success:
        raise IOError("No se pudo codificar PNG")
    Path(path).write_bytes(buf.tobytes())


def remove_background_grabcut(img_bgr, white_thresh=WHITE_THRESH, iter_count=GRABCUT_ITERS):
    """
    Elimina solo el fondo blanco conectado al borde exterior,
    preservando zonas blancas internas (como dentro de un donut).
    """
    h, w = img_bgr.shape[:2]

    # 1) detectar píxeles casi blancos
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    white_mask = (rgb[:,:,0] >= white_thresh) & (rgb[:,:,1] >= white_thresh) & (rgb[:,:,2] >= white_thresh)
    white_mask = white_mask.astype(np.uint8)

    # 2) fondo seguro: solo componentes blancas conectadas al borde
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    bg_seed = np.zeros((h, w), dtype=np.uint8)
    for lab in range(1, num_labels):
        left = stats[lab, cv2.CC_STAT_LEFT]
        top = stats[lab, cv2.CC_STAT_TOP]
        width = stats[lab, cv2.CC_STAT_WIDTH]
        height = stats[lab, cv2.CC_STAT_HEIGHT]
        if left == 0 or top == 0 or (left + width) >= w or (top + height) >= h:
            bg_seed[labels == lab] = 1  # solo bordes

    # 3) probable objeto = todo lo demás
    obj_mask = (1 - bg_seed).astype(np.uint8)

    # 4) semillas seguras de primer plano
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    sure_fg = cv2.erode(obj_mask, kernel, iterations=3)

    # 5) máscara inicial para GrabCut
    mask_gc = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
    mask_gc[bg_seed == 1] = cv2.GC_BGD
    mask_gc[sure_fg == 1] = cv2.GC_FGD

    # 6) ejecutar GrabCut
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(img_bgr, mask_gc, None, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_MASK)
    except Exception:
        # Fallback simple
        alpha = (1 - bg_seed) * 255
        b, g, r = cv2.split(img_bgr)
        return cv2.merge([b, g, r, alpha])

    # 7) máscara final (primer plano o probable primer plano)
    mask_final = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # 8) suavizar bordes
    mask_blur = cv2.GaussianBlur(mask_final, (7, 7), 0)
    alpha = np.clip(mask_blur, 0, 255).astype(np.uint8)

    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, alpha])


def process_and_save(input_path, output_path):
    """Siempre usa GrabCut."""
    try:
        img = imread_utf8(input_path)
        if img is None:
            print(f"⚠️ No se pudo leer: {input_path}")
            return False
        h, w = img.shape[:2]
        if h < MIN_SIZE and w < MIN_SIZE:
            print(f"⏩ Descartada por tamaño: {input_path}")
            return False
        bgra = remove_background_grabcut(img)
        save_png_bytes(output_path, bgra)
        return True
    except Exception as e:
        print(f"❌ Error procesando {input_path}: {e}")
        return False


def main(input_folders, output_folder):
    existing = set()
    total = 0
    ok = 0
    all_files = []

    for folder in input_folders:
        folder = Path(folder)
        if not folder.exists():
            print(f"⚠️ Carpeta no encontrada: {folder}")
            continue
        for entry in folder.iterdir():
            if entry.is_file():
                all_files.append(entry)

    # Si solo queremos un subconjunto
    if PROCESS_SUBSET:
        all_files = random.sample(all_files, min(SUBSET_SIZE, len(all_files)))

    for entry in all_files:
        total += 1
        out_name = make_unique_filename(entry.name, existing)
        out_path = Path(output_folder) / out_name
        if process_and_save(entry, out_path):
            ok += 1

    print(f"\n✅ Procesadas {ok} de {total} imágenes. Guardadas en: {output_folder}")


if __name__ == "__main__":
    main(INPUT_FOLDERS, OUTPUT_FOLDER)
