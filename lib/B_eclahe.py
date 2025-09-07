import cv2
import numpy as np

def _clip_and_redistribute_hist(hist, clip_limit):
    """Clip histogram lalu redistribusi kelebihan secara merata ke semua bin."""
    excess = np.maximum(hist - clip_limit, 0)
    clipped = hist - excess
    n_bins = hist.size
    # total excess yang akan dibagi rata
    total_excess = excess.sum()
    if total_excess > 0:
        # bagikan excess secara merata
        add_each = total_excess // n_bins
        remainder = int(total_excess % n_bins)
        clipped += add_each
        if remainder > 0:
            # sisa didistribusikan satu-satu dari bin awal
            clipped[:remainder] += 1
    return clipped

def _tile_lut(tile_uint8, mu=0.3, q=1e-6):
    """Bangun LUT 256-bin untuk satu ubin (uint8) memakai alpha_e."""
    flat = tile_uint8.ravel()
    l = flat.size               # ℓ = jumlah piksel dalam ubin
    Q = 256                     # jumlah level intensitas
    m = float(flat.mean())      # mean
    sigma = float(flat.std())   # std

    alpha_e = (Q / l) * (1.0 + mu * 2.55 + (sigma / (m + q)))
    clip_limit_per_bin = alpha_e * (l / Q)         # = (1 + μ*2.55 + σ/(m+q))

    # histogram 256-bin
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    # clip + redistribusi
    hist = _clip_and_redistribute_hist(hist.astype(np.int64),
                                       clip_limit=int(np.floor(clip_limit_per_bin)))

    # CDF → LUT [0..255]
    cdf = hist.cumsum().astype(np.float64)
    cdf_min = cdf[np.nonzero(cdf)].min() if np.any(cdf) else 0.0
    denom = (l - cdf_min) if (l - cdf_min) > 0 else 1.0
    lut = np.round((cdf - cdf_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
    return lut

def _apply_per_tile_L(l_img_u8, tile_grid_size=(8,8), mu=0.3, q=1e-6):
    """Terapkan CLAHE termodifikasi pada kanal L (uint8)."""
    h, w = l_img_u8.shape
    ty, tx = tile_grid_size
    # ukuran ubin (terakhir bisa sedikit berbeda)
    ys = np.linspace(0, h, ty+1, dtype=int)
    xs = np.linspace(0, w, tx+1, dtype=int)

    out = l_img_u8.copy()
    for i in range(ty):
        for j in range(tx):
            y0, y1 = ys[i], ys[i+1]
            x0, x1 = xs[j], xs[j+1]
            tile = l_img_u8[y0:y1, x0:x1]
            lut = _tile_lut(tile, mu=mu, q=q)
            out[y0:y1, x0:x1] = cv2.LUT(tile, lut)
    return out

def apply_clahe_modified(
    img: np.ndarray,
    mode: str = "lab",           # "lab" (disarankan) atau "bgr" atau "gray"
    tile_grid_size=(8,8),
    mu: float = 0.3,
    q: float = 1e-6
) -> np.ndarray:
    """
    CLAHE termodifikasi dengan alpha_e = Q/l * (1 + μ*2.55 + σ/(m+q)) per ubin.

    - Input: grayscale (H,W) atau BGR (H,W,3). Dapat float [0,1] atau uint8.
    - Output: dtype & skala sama dengan input.
    """
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("img harus numpy.ndarray")

    orig_dtype = img.dtype
    is_float01 = (img.dtype != np.uint8)

    # ke uint8 [0..255] agar histogram jelas
    if is_float01:
        img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    else:
        img_u8 = img

    if img_u8.ndim == 2 or mode.lower() == "gray":
        L_eq = _apply_per_tile_L(img_u8 if img_u8.ndim == 2 else cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY),
                                 tile_grid_size=tile_grid_size, mu=mu, q=q)
        out_u8 = L_eq if img_u8.ndim == 2 else cv2.cvtColor(L_eq, cv2.COLOR_GRAY2BGR)
    elif mode.lower() == "lab":
        lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        L_eq = _apply_per_tile_L(L, tile_grid_size=tile_grid_size, mu=mu, q=q)
        out_u8 = cv2.cvtColor(cv2.merge([L_eq, A, B]), cv2.COLOR_LAB2BGR)
    elif mode.lower() == "bgr":
        b, g, r = cv2.split(img_u8)
        b = _apply_per_tile_L(b, tile_grid_size=tile_grid_size, mu=mu, q=q)
        g = _apply_per_tile_L(g, tile_grid_size=tile_grid_size, mu=mu, q=q)
        r = _apply_per_tile_L(r, tile_grid_size=tile_grid_size, mu=mu, q=q)
        out_u8 = cv2.merge([b, g, r])
    else:
        raise ValueError("mode harus 'lab', 'bgr', atau 'gray'.")

    # kembalikan ke skala asal
    if is_float01:
        out = (out_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)
        if orig_dtype == np.float64:
            out = out.astype(np.float64)
    else:
        out = out_u8
    return out
