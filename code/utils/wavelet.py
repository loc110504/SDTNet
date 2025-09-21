import pywt
import numpy as np

def decompose_dwt2(image: np.ndarray, wavelet: str = 'db2'):
    """
    Thực hiện DWT2D, tách thành LL, LH, HL, HH.
    """
    LL, (LH, HL, HH) = pywt.dwt2(image, wavelet)
    return LL, LH, HL, HH


def extract_LH( original,
                wavelet: str = 'db2'):
    """
    Tái tạo hai thành phần:
      - L: chỉ giữ LL, zero hóa LH/HL/HH
      - H: chỉ giữ chi tiết LH+HL+HH, zero hóa LL
    Trả về hai ảnh cùng kích thước với original.
    """
    
    LL, LH, HL, HH = decompose_dwt2(original)
    zeros_high = np.zeros_like(LH)
    # Low-frequency only
    L = pywt.idwt2((LL, (zeros_high, zeros_high, zeros_high)), wavelet)

    # High-frequency only
    H = pywt.idwt2((np.zeros_like(LL), (LH, HL, HH)), wavelet)

    return L, H