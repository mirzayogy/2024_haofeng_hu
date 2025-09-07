import torch
import numpy as np
import cv2

from lib.ccf import get_ccf_wrapper

def compensate_color(img_bgr: np.ndarray, color: str = "red",
                        device: str = "cuda"):
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
                        device: str = "cuda"):
   
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


def compensate_color_wrapper(image_path, color):
    """
    Wrapper: versi path file, otomatis baca gambar lalu hitung metrik Lab di GPU.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError("Gambar tidak ditemukan atau path salah.")
    return compensate_color(img_bgr=img_bgr, color=color, cie=True, device="cuda")


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
    img_rgb = cv2.merge([b_channel, g_channel, r_channel])  # hasil RGB

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


def get_color_corrected(image_path, device="cuda"):
    """
    Wrapper: versi path file, otomatis baca gambar lalu hitung metrik Lab di GPU.
    """
    metrics = get_ccf_wrapper(image_path) # metrics['CCF']/metrics['cast']

    img_bgr = cv2.imread(image_path)
    t = torch.from_numpy(img_bgr).to(device=device, dtype=torch.float32) / 255.0
    t = t.permute(2, 0, 1).contiguous()  # [3,H,W], BGR

    b, g, r = t[0], t[1], t[2]
   
    cast = metrics['cast']
    if(cast == 'greenish'):
        Ir = compensate_color(img_bgr, color="red")
        Ib = compensate_color(img_bgr, color="blue")
        out = torch.stack([Ir, g, Ib], dim=0)  # [3,H,W]
    elif(cast == "yellowish"):
        Ig = compensate_color(img_bgr, color="green")
        Ib = compensate_color(img_bgr, color="blue")
        out = torch.stack([r, Ig, Ib], dim=0)  # [3,H,W]
    elif(cast == "bluish"):
        Ig = compensate_color_bluish(img_bgr, color="green")
        Ir = compensate_color_bluish(img_bgr, color="red")
        out = torch.stack([Ir, Ig, b], dim=0)  # [3,H,W]
        
    out = (out * 255.0).round().clamp(0, 255).to(torch.uint8)
    out = out.permute(1, 2, 0).contiguous()  # [H,W,3] BGR
    out_bgr = out.detach().cpu().numpy()

    return out_bgr, cast

def get_color_corrected_increased_dynamic_range(image_path, device="cuda"):
    im_corrected, cast = get_color_corrected(image_path)
    b, g, r = cv2.split(im_corrected)

    im_increased_r = increase_dynamic_range(r)
    im_increased_g = increase_dynamic_range(g)
    im_increased_b = increase_dynamic_range(b)

    merged = merge_rgb(im_increased_r, im_increased_g, im_increased_b)

    return merged




