import cv2 # ignore
import numpy as np

def mean_ab_M_array(img_array, cie=False):
    """
    Menghitung mean chromaticity m_a, m_b, dan M dari gambar (numpy array).
    
    Parameters:
        img_array (numpy.ndarray): gambar dalam format BGR (seperti hasil cv2.imread)
        cie (bool): jika True, konversi channel a,b ke rentang [-128, 127]
    
    Returns:
        (m_a, m_b, M): tuple nilai rata-rata channel a, b, dan M
    """
    if img_array is None or not isinstance(img_array, np.ndarray):
        raise ValueError("Input harus berupa numpy array gambar.")

    # Konversi BGR ke Lab
    lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2Lab)

    # Pisahkan channel
    L, a, b = cv2.split(lab)

    # Jika ingin pakai skala asli CIE Lab [-128, 127]
    if cie:
        a = a.astype(np.float32) - 128
        b = b.astype(np.float32) - 128

    # Ukuran gambar
    p, q = a.shape

    # Hitung mean
    m_a = np.sum(a) / (p * q)
    m_b = np.sum(b) / (p * q)

    # Hitung M
    M = np.sqrt(m_a**2 + m_b**2)

    # return m_a, m_b, M


    if m_b == 0:
        ratio = np.inf
    else:
        ratio = abs(m_a / m_b)
    if (m_a < 0) and (ratio >= 1):
        cast = "greenish"
    elif (m_b < 0) and (ratio < 1):
        cast = "bluish"
    else:
        cast = "yellowish"

    return M, cast


def mean_ab_M(image_path, cie=False):
    """
    Wrapper: versi path file, otomatis baca gambar lalu hitung m_a, m_b, dan M.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Gambar tidak ditemukan atau path salah.")
    return mean_ab_M_array(img, cie=cie)


# # Contoh penggunaan
# if __name__ == "__main__":
#     img_path = "contoh.jpg"

#     # Default (rentang OpenCV 0–255)
#     m_a, m_b, M = mean_ab_M(img_path, cie=False)
#     print("[OpenCV scale] m_a:", m_a, " m_b:", m_b, " M:", M)

#     # Versi rentang CIE Lab asli [-128,127]
#     m_a, m_b, M = mean_ab_M(img_path, cie=True)
#     print("[CIE Lab scale] m_a:", m_a, " m_b:", m_b, " M:", M)


def deviation_ab_D_array(img_array, cie=False):
    """
    Menghitung D_a, D_b, dan D dari gambar (numpy array).
    
    Parameters:
        img_array (numpy.ndarray): gambar dalam format BGR (seperti hasil cv2.imread)
        cie (bool): jika True, konversi channel a,b ke rentang [-128, 127]
    
    Returns:
        (D_a, D_b, D): tuple nilai deviasi rata-rata channel a, b, dan gabungannya D
    """
    if img_array is None or not isinstance(img_array, np.ndarray):
        raise ValueError("Input harus berupa numpy array gambar.")

    # Konversi BGR ke Lab
    lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2Lab)

    # Pisahkan channel
    L, a, b = cv2.split(lab)

    # Jika ingin pakai skala asli CIE Lab [-128, 127]
    if cie:
        a = a.astype(np.float32) - 128
        b = b.astype(np.float32) - 128

    # Ukuran gambar
    p, q = a.shape
    N = p * q

    # Mean dari channel a dan b
    m_a = np.sum(a) / N
    m_b = np.sum(b) / N

    # Hitung deviasi rata-rata
    D_a = np.sum(np.abs(a - m_a)) / N
    D_b = np.sum(np.abs(b - m_b)) / N

    # Gabungan
    D = np.sqrt(D_a**2 + D_b**2)

    # return D_a, D_b, D
    return D


def deviation_ab_D(image_path, cie=False):
    """
    Wrapper: versi path file.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Gambar tidak ditemukan atau path salah.")
    return deviation_ab_D_array(img, cie=cie)


# # Contoh penggunaan
# if __name__ == "__main__":
#     img_path = "contoh.jpg"

#     D_a, D_b, D = deviation_ab_D(img_path, cie=True)
#     print("D_a:", D_a, " D_b:", D_b, " D:", D)

def getCCF(image_path):
    M, cast = mean_ab_M(image_path,cie = True)
    D = deviation_ab_D(image_path, cie = True)

    return M/D, cast

def _classify_cast(m_a: float, m_b: float) -> str:
    # |ma/mb| dengan penanganan mb=0
    if m_b == 0:
        ratio = np.inf
    else:
        ratio = abs(m_a / m_b)

    if (m_a < 0) and (ratio >= 1):
        return "greenish"
    elif (m_b < 0) and (ratio < 1):
        return "bluish"
    else:
        return "yellowish"