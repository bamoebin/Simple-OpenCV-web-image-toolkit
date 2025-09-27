import numpy as np
import cv2


def _to_bgr(image: np.ndarray) -> np.ndarray:
    """Ensure the image is a 3-channel BGR uint8 image if originally color.
    If grayscale, keep single channel.
    """
    if image is None:
        raise ValueError("Input image is None")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize arbitrary-range image to uint8 [0,255]. Preserves channel count."""
    if img is None:
        raise ValueError("Image to normalize is None")
    arr = img.astype(np.float32)
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v == min_v:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - min_v) / (max_v - min_v) * 255.0
    return arr.astype(np.uint8)


def apply_convolution(image: np.ndarray, kernel_type: str = "average") -> np.ndarray:
    """Apply spatial filtering via unified 'convolution' operation.

    Previously there was a separate 'filter' operation (low/high/band). Those
    have been merged here to avoid redundancy:
      - low  pass  => kernel_type='average' (or use 'gaussian')
      - high pass  => kernel_type='edge' (edge emphasis / Laplacian variant)
      - band pass  => effectively close to identity; can approximate using 'sharpen'

    Added 'gaussian' for clarity replacing the old low-pass option.
    """
    image = _to_bgr(image)
    kt = (kernel_type or "average").lower()
    if kt == "average":
        kernel = np.ones((3, 3), np.float32) / 9.0
        return cv2.filter2D(image, -1, kernel)
    if kt == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    if kt == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)
    if kt == "edge":
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)
    raise ValueError(f"Unknown kernel_type: {kernel_type}")


def apply_zero_padding(image: np.ndarray, padding_size: int = 10) -> np.ndarray:
    image = _to_bgr(image)
    pad = int(padding_size) if padding_size is not None else 10
    if pad < 0:
        pad = 0
    padded_img = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_img


## Removed redundant apply_filter (low/high/band) – functionality merged into apply_convolution


def apply_fourier_transform(image: np.ndarray) -> np.ndarray:
    img = _to_bgr(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1.0)  # avoid log(0)
    mag_uint8 = _normalize_to_uint8(magnitude_spectrum)
    return mag_uint8


def reduce_periodic_noise(image: np.ndarray, radius: int = 30) -> np.ndarray:
    img = _to_bgr(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    r = int(radius) if radius is not None else 30
    r = max(1, r)
    mask = np.ones((rows, cols), np.float32)
    mask[crow - r:crow + r, ccol - r:ccol + r] = 0.0

    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return _normalize_to_uint8(img_back)


def process_image(operation: str, image: np.ndarray, **kwargs) -> np.ndarray:
    op = (operation or "").strip().lower()
    if op == "convolution":
        return apply_convolution(image, kernel_type=kwargs.get("kernel_type", "average"))
    elif op == "zero_padding":
        return apply_zero_padding(image, padding_size=int(kwargs.get("padding_size", 10)))
    elif op == "fourier_transform":
        return apply_fourier_transform(image)
    elif op == "reduce_periodic_noise":
        return reduce_periodic_noise(image, radius=int(kwargs.get("radius", 30)))
    else:
        raise ValueError(f"Unsupported operation: {operation}")
