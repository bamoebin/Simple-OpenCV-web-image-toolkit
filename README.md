# FastAPI OpenCV App - Notebook Ops Integration

## Install

- Create/activate a Python env
- Install deps:

```powershell
pip install -r requirements.txt
```

## Run

From this folder:

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000, then use "Notebook Operations" in the sidebar.

## New endpoints

- GET /notebook_ops/ — form
- POST /notebook_ops/ — process upload with params:
  - operation: convolution | zero_padding | fourier_transform | reduce_periodic_noise
  - kernel_type (for convolution): average | gaussian | sharpen | edge
  - padding_size (for zero_padding): int
  - radius (for reduce_periodic_noise): int

Notes:
- Previously there was a separate 'filter' operation (low/high/band). It was redundant with convolution kernels, so its options were merged into 'convolution' for simplicity.

Results are saved to `static/uploads` and rendered in `templates/result.html`.
