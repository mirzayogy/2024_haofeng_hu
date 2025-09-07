import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import numpy as np
import math

def show_image_matplotlib(img: np.ndarray, title: str = None, title_2: str = None):
    """
    Menampilkan gambar dari numpy.ndarray menggunakan matplotlib.
    img : np.ndarray (RGB atau grayscale)
    """
    # plt.figure(figsize=(6,6))
    if img.ndim == 2:  # grayscale
        plt.imshow(img, cmap="gray")
    else:  # warna
        plt.imshow(img)
    if title:
        plt.title(f"{title}\n")
    if title_2:
        plt.title(f"{title}\n")
    plt.axis("off")
    plt.show()

def show_image_from_path(an_image, label1="", label2=""):
    """
    Tampilkan satu gambar dari path.
    
    Params
    ------
    an_image : path ke gambar
    """

    filename = os.path.basename(an_image)
    img = plt.imread(an_image)

    if(label1 == ""):
        label1 = filename

    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{label1}\n{label2}", fontsize=10)
    plt.show()

def show_rgb_image(r_channel, g_channel, b_channel):
    """
    Gabungkan channel R,G,B lalu tampilkan gambar tanpa menyimpan.
    """
    # Pastikan semua channel ada di range [0..255] uint8
    if r_channel.dtype != np.uint8:
        r_channel = (r_channel * 255).clip(0, 255).astype(np.uint8)
    if g_channel.dtype != np.uint8:
        g_channel = (g_channel * 255).clip(0, 255).astype(np.uint8)
    if b_channel.dtype != np.uint8:
        b_channel = (b_channel * 255).clip(0, 255).astype(np.uint8)

    # Gabungkan ke RGB
    img_rgb = cv2.merge([r_channel, g_channel, b_channel])

    # Tampilkan
    plt.figure(figsize=(6,6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

    return img_rgb

def show_image_group_auto(image_group, max_cols=6, cell_size=(3, 3), hide_empty=True):
    """
    Tampilkan grid gambar dengan ukuran grid otomatis.
    
    Params
    ------
    image_group : object dengan atribut .image berupa list path gambar
    max_cols    : kolom maksimum yang dipakai (default 6)
    cell_size   : (width, height) inch per sel (default (3,3))
    hide_empty  : jika True, axes kosong dihapus (lebih rapi)
    """
    # Ambil daftar path gambar
    paths = list(getattr(image_group, "image", []))
    n = len(paths)

    if n == 0:
        print("Tidak ada gambar untuk ditampilkan.")
        return

    # Hitung kolom & baris otomatis
    cols = min(max_cols, max(1, math.ceil(math.sqrt(n))))
    rows = math.ceil(n / cols)

    # Figure size proporsional jumlah sel
    fig_w = max(1, cols * cell_size[0])
    fig_h = max(1, rows * cell_size[1])

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_w, fig_h),
                             subplot_kw={'xticks':[], 'yticks':[]})
    # Pastikan axes flattened (kompatibel untuk kasus rows/cols == 1)
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    # Render tiap gambar
    shown = 0
    for i, ax in enumerate(axes):
        if i >= n:
            # Habis gambar
            if hide_empty:
                fig.delaxes(ax)
            continue

        img_path = paths[i]
        try:
            ax.imshow(imread(img_path))
            _, filename = os.path.split(img_path)
            ax.set_title(filename)
            shown += 1
        except Exception as e:
            # Gagal baca gambar: kosongkan axes ini
            ax.text(0.5, 0.5, f"Gagal baca:\n{os.path.basename(img_path)}",
                    ha='center', va='center', fontsize=8)
            ax.set_frame_on(True)

    plt.tight_layout()
    plt.show()

    # Opsional: return fig/axes kalau mau dipakai lanjut
    return fig, axes
