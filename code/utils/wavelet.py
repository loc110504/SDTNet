import pywt
import numpy as np

def decompose_dwt2(image: np.ndarray, wavelet: str = 'db2'):
    LL, (LH, HL, HH) = pywt.dwt2(image, wavelet)
    return LL, LH, HL, HH

def minmax01(x: np.ndarray, eps=1e-8):
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if abs(mx - mn) < eps:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def extract_LH(original: np.ndarray, wavelet: str = 'db2'):
    """
    Tái tạo và scale về [0,1]:
      - L: chỉ giữ LL, zero hóa LH/HL/HH
      - H: chỉ giữ LH+HL+HH, zero hóa LL
    """
    LL, LH, HL, HH = decompose_dwt2(original, wavelet)
    zeros_high = np.zeros_like(LH)

    # Low-frequency only
    L = pywt.idwt2((LL, (zeros_high, zeros_high, zeros_high)), wavelet)

    # High-frequency only
    H = pywt.idwt2((np.zeros_like(LL), (LH, HL, HH)), wavelet)

    # Scale về [0,1] trước khi trả về
    L = minmax01(L)
    H = minmax01(H)

    return L, H