import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

def show_images_from_paths(paths, titles=None, max_cols=5, cell_size=(3,3)):
    """
    Menampilkan gambar dari list path.

    Params
    ------
    paths : list[str]
        Daftar path gambar
    titles : list[str] atau None
        Judul tambahan untuk tiap gambar (opsional)
    max_cols : int
        Jumlah kolom maksimum (default 5)
    cell_size : tuple
        Ukuran per sel dalam inches (default (3,3))
    """
    if not paths:
        print("Tidak ada gambar.")
        return

    n = len(paths)
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*cell_size[0], rows*cell_size[1]),
                             subplot_kw={'xticks': [], 'yticks': []})
    
    # # flatten agar mudah di-loop
    # if isinstance(axes, plt.Axes):
    #     axes = [axes]
    # else:
    #     axes = axes.flatten()

    # for i, ax in enumerate(axes):
    #     if i >= n:
    #         fig.delaxes(ax)
    #         continue

    #     img_path = paths[i]
    #     try:
    #         ax.imshow(imread(img_path))
    #         _, filename = os.path.split(img_path)
    #         if titles and i < len(titles):
    #             ax.set_title(f"{filename}\n{titles[i]}", fontsize=9)
    #         else:
    #             ax.set_title(filename, fontsize=9)
    #     except Exception:
    #         ax.text(0.5, 0.5, f"Gagal baca:\n{os.path.basename(img_path)}",
    #                 ha='center', va='center', fontsize=8)
    #         ax.set_frame_on(True)

    # plt.tight_layout()
    # plt.show()
