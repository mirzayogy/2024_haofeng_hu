import cv2
import torch

def lab_metrics_gpu(img_array, cie=False, device="cuda"):
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

    # Cast classification
    ratio = torch.abs(m_a / m_b) if m_b != 0 else torch.tensor(float("inf"), device=device)
    if (m_a < 0) and (ratio >= 1):
        cast = "greenish"
    elif (m_b < 0) and (ratio < 1):
        cast = "bluish"
    else:
        cast = "yellowish"

    return {
        "m_a": float(m_a.item()),
        "m_b": float(m_b.item()),
        "M": float(M.item()),
        "D_a": float(D_a.item()),
        "D_b": float(D_b.item()),
        "D": float(D.item()),
        "cast": cast
    }

# -------- contoh pemanggilan ----------
if __name__ == "__main__":
    img = cv2.imread("../UIEB/pernah/18_img_.png")
    metrics = lab_metrics_gpu(img, cie=True, device=None)
    print(metrics['cast'])
