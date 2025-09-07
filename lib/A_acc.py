import torch  # type: ignore
import numpy as np # type: ignore
import cv2 # type: ignore


def compensate_color(img_bgr: np.ndarray, color: str = "red",
                        device=None):
    """
    Kompensasi channel merah sesuai dua tahap:
      1. Rumus Ir'(x,y) = Ir(x,y) + (1 - IrM/IgM)*(1 - Ir(x,y))*Ig(x,y)
      2. Normalisasi: Îr(x,y) = (Ir'(x,y) / Ir'M) * 0.5

    Parameters
    ----------
    img_bgr : np.ndarray
        Gambar input BGR (cv2.imread), uint8 [0..255].
    device : {"cuda","cpu"}
        Perangkat eksekusi.
    return_means : bool
        Jika True, kembalikan juga nilai mean.

    Returns
    -------
    out_bgr : np.ndarray
        Gambar hasil kompensasi (BGR, uint8).
    info : dict (opsional)
        {"IrM":..., "IgM":..., "IrM_prime":..., "IrM_hat":...}
    """
    if img_bgr is None or not isinstance(img_bgr, np.ndarray):
        raise ValueError("Input harus berupa numpy array gambar (BGR).")

    # ke tensor [C,H,W] [0,1]
    t = torch.from_numpy(img_bgr).to(device=device, dtype=torch.float32) / 255.0
    t = t.permute(2, 0, 1).contiguous()  # [3,H,W], BGR

    b, g, r = t[0], t[1], t[2]

    # step 1: kompensasi merah
    IrM = r.mean()
    IgM = g.mean()
    IbM = b.mean()
    eps = torch.finfo(torch.float32).eps
    
    if color == "red":
        scale = 1.0 - (IrM / (IgM + eps))
        Iaksen = r + scale * (1.0 - r) * g
    elif color == "green":
        scale = 1.0 - (IgM / (IrM + eps))
        Iaksen = g + scale * (1.0 - g) * r
    elif color == "blue":
        scale = 1.0 - (IbM / (IgM + eps))
        Iaksen = b + scale * (1.0 - b) * g
    
    Iaksen = torch.clamp(Iaksen, 0.0, 1.0)

    # step 2: normalisasi merah
    IaksenM = Iaksen.mean()
    Itopi = (Iaksen / (IaksenM + eps)) * 0.5
    Itopi = torch.clamp(Itopi, 0.0, 1.0)
    return Itopi
    # gabungkan kembali hasil dengan r_hat
    # out = torch.stack([b, g, Itopi], dim=0)  # [3,H,W]
    # out = (out * 255.0).round().clamp(0, 255).to(torch.uint8)
    # out = out.permute(1, 2, 0).contiguous()  # [H,W,3] BGR

    # out_bgr = out.detach().cpu().numpy()
    # return out_bgr

def compensate_color_bluish(img_bgr: np.ndarray, color: str = "red",
                        device=None):
   
    if img_bgr is None or not isinstance(img_bgr, np.ndarray):
        raise ValueError("Input harus berupa numpy array gambar (BGR).")

    # ke tensor [C,H,W] [0,1]
    t = torch.from_numpy(img_bgr).to(device=device, dtype=torch.float32) / 255.0
    t = t.permute(2, 0, 1).contiguous()  # [3,H,W], BGR

    b, g, r = t[0], t[1], t[2]

    # step 1: kompensasi merah
    IrM = r.mean()
    IgM = g.mean()
    IbM = b.mean()
    eps = torch.finfo(torch.float32).eps
    
    scale = 1.0 - (IgM / (IbM + eps))
    Iaksen = g + scale * (1.0 - g) * b
    Iaksen = torch.clamp(Iaksen, 0.0, 1.0)
    
    if color == "red":
        scale = 1.0 - (IrM / (IgM + eps))
        Iaksen = r + scale * (1.0 - r) * Iaksen
        Iaksen = torch.clamp(Iaksen, 0.0, 1.0)


    # step 2: normalisasi merah
    IaksenM = Iaksen.mean()
    Itopi = (Iaksen / (IaksenM + eps)) * 0.5
    Itopi = torch.clamp(Itopi, 0.0, 1.0)
    return Itopi


def compensate_color_wrapper(image_path, color, device):
    """
    Wrapper: versi path file, otomatis baca gambar lalu hitung metrik Lab di GPU.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError("Gambar tidak ditemukan atau path salah.")
    return compensate_color(img_bgr=img_bgr, color=color, cie=True, device=device)


def merge_rgb(r_channel, g_channel, b_channel):
    """
    Menggabungkan tiga channel R, G, B menjadi satu citra warna.
    
    Parameters
    ----------
    r_channel, g_channel, b_channel : np.ndarray
        Channel citra hasil kompensasi, ukuran [H,W].
        dtype bisa float [0..1] atau uint8 [0..255].
    
    Returns
    -------
    img_rgb : np.ndarray
        Citra warna RGB gabungan.
    """
    # pastikan tipe konsisten
    if r_channel.dtype != g_channel.dtype or r_channel.dtype != b_channel.dtype:
        raise ValueError("Semua channel harus punya dtype yang sama")

    # stack channel
    img_rgb = cv2.merge([r_channel, g_channel, b_channel])  # hasil RGB
    # img_rgb = cv2.merge([b_channel, g_channel, r_channel])  # hasil RGB

    return img_rgb


def increase_dynamic_range(channel: np.ndarray):
    """
    Proses clipping & normalisasi sesuai rumus:
      1. Clipping berdasarkan persentil 0.3% dan 99.7%
      2. Normalisasi ke [0,1]
    
    Parameters
    ----------
    channel : np.ndarray
        Array 2D (1 channel) intensitas, dtype bisa uint8 atau float.
    
    Returns
    -------
    Ic_clipped : np.ndarray
        Channel setelah clipping.
    Ic_final : np.ndarray
        Channel setelah normalisasi (float32 [0,1]).
    info : dict
        Nilai I_cmin, I_cmax, I_0.3, I_99.7
    """
    channel = channel.astype(np.float32)

    # ambil threshold berdasarkan persentil
    I_03 = np.percentile(channel, 0.3)
    I_997 = np.percentile(channel, 99.7)

    # clipping
    Ic_clipped = np.where(channel < I_03, I_03, channel)
    Ic_clipped = np.where(Ic_clipped > I_997, I_997, Ic_clipped)

    # dapatkan min & max setelah clipping
    I_cmin = Ic_clipped.min()
    I_cmax = Ic_clipped.max()

    # normalisasi
    Ic_final = (Ic_clipped - I_cmin) / (I_cmax - I_cmin + 1e-8)

    # return Ic_clipped, Ic_final, {
    #     "I_cmin": float(I_cmin),
    #     "I_cmax": float(I_cmax),
    #     "I_0.3": float(I_03),
    #     "I_99.7": float(I_997),
    # }

    return Ic_final


def get_color_corrected(image_path, device=None):
    """
    Wrapper: versi path file, otomatis baca gambar lalu hitung metrik Lab di GPU.
    """
    metrics = get_ccf_wrapper(image_path, device) # metrics['CCF']/metrics['cast']

    img_bgr = cv2.imread(image_path)
    t = torch.from_numpy(img_bgr).to(device=device, dtype=torch.float32) / 255.0
    t = t.permute(2, 0, 1).contiguous()  # [3,H,W], BGR

    b, g, r = t[0], t[1], t[2]
   
    cast = metrics['cast']
    if(cast == 'greenish'):
        Ir = compensate_color(img_bgr, color="red", device=device)
        Ib = compensate_color(img_bgr, color="blue", device=device)
        out = torch.stack([Ir, g, Ib], dim=0)  # [3,H,W]
    elif(cast == "yellowish"):
        Ig = compensate_color(img_bgr, color="green", device=device)
        Ib = compensate_color(img_bgr, color="blue", device=device)
        out = torch.stack([r, Ig, Ib], dim=0)  # [3,H,W]
    elif(cast == "bluish"):
        Ig = compensate_color_bluish(img_bgr, color="green", device=device)
        Ir = compensate_color_bluish(img_bgr, color="red", device=device)
        out = torch.stack([Ir, Ig, b], dim=0)  # [3,H,W]
    else:
        out = torch.stack([r, g, b], dim=0)  # [3,H,W]
    out = (out * 255.0).round().clamp(0, 255).to(torch.uint8)
    out = out.permute(1, 2, 0).contiguous()  # [H,W,3] BGR
    out_bgr = out.detach().cpu().numpy()

    return out_bgr, cast

def get_color_corrected_increased_dynamic_range(image_path, device=None):
    im_corrected, cast = get_color_corrected(image_path, device=device)
    b, g, r = cv2.split(im_corrected)

    im_increased_r = increase_dynamic_range(r)
    im_increased_g = increase_dynamic_range(g)
    im_increased_b = increase_dynamic_range(b)

    merged = merge_rgb(im_increased_r, im_increased_g, im_increased_b)

    return merged, cast



def get_ccf(img_array, cie=False, device=None):
    """
    Hitung m_a, m_b, M, D_a, D_b, D, dan cast menggunakan PyTorch GPU.
    """
    if img_array is None:
        raise ValueError("Input harus berupa numpy array gambar.")

    # Konversi BGR -> Lab (CPU OpenCV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)

    # Convert ke tensor di GPU
    a = torch.tensor(a, dtype=torch.float32, device=device)
    b = torch.tensor(b, dtype=torch.float32, device=device)

    if cie:
        a -= 128.0
        b -= 128.0

    N = a.numel()

    # Mean
    m_a = torch.sum(a) / N
    m_b = torch.sum(b) / N

    # Magnitude M
    M = torch.sqrt(m_a**2 + m_b**2)

    # Deviation
    D_a = torch.sum(torch.abs(a - m_a)) / N
    D_b = torch.sum(torch.abs(b - m_b)) / N
    D = torch.sqrt(D_a**2 + D_b**2)

    CCF = float(M.item() / D.item()) if D.item() != 0 else float("inf")

    if(CCF < 1):
        cast = "no-color-cast"
    else:
        # Cast classification
        ratio = torch.abs(m_a / m_b) if m_b != 0 else torch.tensor(float("inf"), device=device)
        if (m_a < 0) and (ratio >= 1):
            cast = "greenish"
        elif (m_b < 0) and (ratio < 1):
            cast = "bluish"
        else:
            cast = "yellowish"

    return {
        # "m_a": float(m_a.item()),
        # "m_b": float(m_b.item()),
        # "M": float(M.item()),
        # "D_a": float(D_a.item()),
        # "D_b": float(D_b.item()),
        # "D": float(D.item()),
        "CCF": float(M.item() / D.item()) if D.item() != 0 else float("inf"),
        "cast": cast
    }

def get_ccf_wrapper(image_path, cie=True, device=None):
    """
    Wrapper: versi path file, otomatis baca gambar lalu hitung metrik Lab di GPU.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Gambar tidak ditemukan atau path salah.")
    return get_ccf(img, cie=cie, device=device)




