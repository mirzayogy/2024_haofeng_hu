import cv2
import numpy as np
import os

def ekspor_png(img_or_path, folder_name, file_name):
    """
    Simpan gambar sebagai PNG.
    Bisa menerima input berupa:
      - path (str) ke file gambar
      - numpy.ndarray (gambar)

    Parameters
    ----------
    img_or_path : str atau np.ndarray
        Path file gambar atau array citra.
    folder_name : str
        Folder tujuan penyimpanan.
    file_name : str
        Nama file PNG (contoh: 'hasil.png')
    """
    # Jika input adalah path string
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path)
        if img is None:
            raise ValueError(f"Gagal membaca gambar dari path: {img_or_path}")
    # Jika input sudah berupa array
    elif isinstance(img_or_path, np.ndarray):
        img = img_or_path
    else:
        raise TypeError("img_or_path harus string path atau numpy.ndarray")

    # Konversi ke uint8 bila perlu
    if img.dtype != np.uint8:
        img_to_save = (img * 255).clip(0, 255).astype(np.uint8)
    else:
        img_to_save = img

    # Pastikan folder ada
    os.makedirs(folder_name, exist_ok=True)

    # Simpan PNG
    out_path = os.path.join(folder_name, file_name)
    cv2.imwrite(out_path, img_to_save)
    # print(f"Gambar disimpan ke {out_path}")
