
import os
from tqdm import tqdm  # pastikan sudah install: pip install tqdm
import csv


def batch(
        input_folder: str, 
        output_csv: str,
        exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
        recursive: bool = False
        ):
    
    # 1) Ambil isi folder
    all_files = []
    walker = os.walk(input_folder) if recursive else [(input_folder, [], os.listdir(input_folder))]
    for root, _, files in walker:
        for fname in files:
            if fname.lower().endswith(exts):
                all_files.append(os.path.abspath(os.path.join(root, fname)))
    all_files.sort()
    rows = []
    iter = 0
    for fpath in tqdm(all_files, desc="Processing images", unit="img"):
        filename = os.path.basename(fpath)

    # 2) Proses Disini

        rows.append([
            iter,
            filename
        ])
        iter = iter + 1

    # 3) Tulis ke CSV
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    write_header = True
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["id","filename", "uiqm", "uciqe", ])
        writer.writerows(rows)

    print(f"Selesai. Diproses {len(rows)} "
            f"Total sekarang (CSV) ≈ {len(rows)}.")
    return len(rows)

batch("../UIEB/pernah", "hasil/tes.csv")
