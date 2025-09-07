import cv2 # type: ignore
import numpy as np # type: ignore

def enhance_L_with_D_lab(img_bgr: np.ndarray, eps: float = 1e-6):
    """
    Hitung D pada ruang Lab dan hasilkan gambar dengan L' = L + D,
    digabungkan kembali dengan kanal a,b asli.

    Parameters
    ----------
    img_bgr : np.ndarray
        Gambar BGR (uint8 atau float [0,1]).
    eps : float
        Konstanta kecil untuk log.

    Returns
    -------
    out_bgr : np.ndarray
        Gambar hasil (BGR, uint8).
    D_map : np.ndarray
        Peta D (float32, [H,W]).
    D_mean : float
        Nilai D rata-rata.
    """
    # pastikan uint8 untuk konversi warna OpenCV, simpan skala float jika perlu
    if img_bgr.dtype != np.uint8:
        bgr_u8 = (img_bgr * 255.0).clip(0, 255).astype(np.uint8)
    else:
        bgr_u8 = img_bgr

    # BGR -> Lab (OpenCV skala 0..255 untuk L,a,b)
    lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:, :, 0] / 255.0        # normalisasi ke [0,1]
    a = lab[:, :, 1]                # biarkan pada skala OpenCV (0..255)
    b = lab[:, :, 2]

    # siapkan L_k (L0 di index 0)
    Ls = [L]
    sigmas = [1, 2, 4]
    betas  = [0.7, 0.5, 0.25]

    for s in sigmas:
        ksize = int(6 * s + 1)
        L_blur = cv2.GaussianBlur(L, (ksize, ksize), s, borderType=cv2.BORDER_REFLECT101)
        Ls.append(L_blur.astype(np.float32))

    # D map
    D_map = np.zeros_like(L, dtype=np.float32)
    for k, beta in enumerate(betas, start=1):
        D_map += beta * (np.log(Ls[k-1] + eps) - np.log(Ls[k] + eps))

    D_mean = float(np.mean(D_map))

    # L' = L + D (klip ke [0,1])
    L_new = np.clip(L + D_map, 0.0, 1.0)

    # satukan kembali ke Lab (skala OpenCV)
    L_new_u8 = (L_new * 255.0).round().clip(0, 255).astype(np.uint8)
    lab_out = cv2.merge([L_new_u8, a.astype(np.uint8), b.astype(np.uint8)])

    # Lab -> BGR
    out_bgr = cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)

    # return out_bgr, D_map, D_mean
    return out_bgr
