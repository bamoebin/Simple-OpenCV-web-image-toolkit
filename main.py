import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from skimage.exposure import match_histograms  # pastikan paket scikit-image sudah terinstal

import numpy as np
import cv2
import matplotlib.pyplot as plt

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")

if not os.path.exists("static/histograms"):
    os.makedirs("static/histograms")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Endpoint ini digunakan untuk mengupload gambar yang akan diproses. Gambar yang diupload akan dibaca dan diubah menjadi array numpy untuk diproses lebih lanjut.
@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    file_path = save_image(img, "uploaded")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": file_path,
        "modified_image_path": file_path
    })
# Endpoint ini digunakan untuk melakukan operasi aritmatika pada gambar. Operasi yang tersedia adalah penjumlahan, pengurangan, nilai maksimum, nilai minimum, dan inversi.
@app.post("/operation/", response_class=HTMLResponse)
async def perform_operation(
    request: Request,
    file: UploadFile = File(...),
    operation: str = Form(...),
    value: int = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    original_path = save_image(img, "original")

    if operation == "add":
        result_img = cv2.add(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "subtract":
        result_img = cv2.subtract(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "max":
        result_img = np.maximum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "min":
        result_img = np.minimum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "inverse":
        result_img = cv2.bitwise_not(img)

    modified_path = save_image(result_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })
# Endpoint ini digunakan untuk melakukan operasi logika pada gambar. Operasi yang tersedia adalah AND, XOR, dan NOT.
@app.post("/logic_operation/", response_class=HTMLResponse)
async def perform_logic_operation(
    request: Request,
    file1: UploadFile = File(...),
    file2: UploadFile = File(None),
    operation: str = Form(...)
):
    image_data1 = await file1.read()
    np_array1 = np.frombuffer(image_data1, np.uint8)
    img1 = cv2.imdecode(np_array1, cv2.IMREAD_COLOR)

    original_path = save_image(img1, "original")

    if operation == "not":
        result_img = cv2.bitwise_not(img1)
        modified_path = save_image(result_img, "modified")
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "original_image_path": original_path,
            "modified_image_path": modified_path
        })
    else:
        if file2 is None:
            return HTMLResponse("Operasi AND dan XOR memerlukan dua gambar.", status_code=400)
        image_data2 = await file2.read()
        np_array2 = np.frombuffer(image_data2, np.uint8)
        img2 = cv2.imdecode(np_array2, cv2.IMREAD_COLOR)

        # Cek apakah dimensi sama
        if img1.shape != img2.shape:
            # Lakukan crop otomatis ke dimensi terkecil
            img1_cropped, img2_cropped, final_dim, orig_dim1, orig_dim2 = handle_dimension_mismatch(img1, img2)
            
            # Simpan citra yang sudah di-crop
            img1_cropped_path = save_image(img1_cropped, "img1_cropped")
            img2_cropped_path = save_image(img2_cropped, "img2_cropped")
            
            # Lakukan operasi pada citra yang sudah di-crop
            if operation == "and":
                result_img = cv2.bitwise_and(img1_cropped, img2_cropped)
            elif operation == "xor":
                result_img = cv2.bitwise_xor(img1_cropped, img2_cropped)
            
            modified_path = save_image(result_img, "modified")
            
            return templates.TemplateResponse("logic_result.html", {
                "request": request,
                "original_image_path": original_path,
                "img2_path": save_image(img2, "img2_original"),
                "img1_cropped_path": img1_cropped_path,
                "img2_cropped_path": img2_cropped_path,
                "modified_image_path": modified_path,
                "operation": operation.upper(),
                "dimension_mismatch": True,
                "original_dim1": f"{orig_dim1[0]}x{orig_dim1[1]}",
                "original_dim2": f"{orig_dim2[0]}x{orig_dim2[1]}",
                "cropped_dim": f"{final_dim[0]}x{final_dim[1]}"
            })
        else:
            # Dimensi sama, lakukan operasi normal
            if operation == "and":
                result_img = cv2.bitwise_and(img1, img2)
            elif operation == "xor":
                result_img = cv2.bitwise_xor(img1, img2)
            
            modified_path = save_image(result_img, "modified")
            img2_path = save_image(img2, "img2_original")
            
            return templates.TemplateResponse("logic_result.html", {
                "request": request,
                "original_image_path": original_path,
                "img2_path": img2_path,
                "modified_image_path": modified_path,
                "operation": operation.upper(),
                "dimension_mismatch": False
            })

# Endpoint untuk menampilkan form operasi logika
@app.get("/logic_operation_form/", response_class=HTMLResponse)
async def logic_operation_form(request: Request):
    return templates.TemplateResponse("logic_operation.html", {"request": request})

# Endpoint ini digunakan untuk mengubah gambar berwarna menjadi grayscale.
@app.get("/grayscale/", response_class=HTMLResponse)
async def grayscale_form(request: Request):
    # Menampilkan form untuk upload gambar ke grayscale
    return templates.TemplateResponse("grayscale.html", {"request": request})

# Endpoint ini digunakan untuk mengubah gambar berwarna menjadi grayscale.
@app.post("/grayscale/", response_class=HTMLResponse)
async def convert_grayscale(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    original_path = save_image(img, "original")
    modified_path = save_image(gray_img, "grayscale")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

# Endpoint ini digunakan untuk menampilkan histogram dari gambar yang diupload.
@app.get("/histogram/", response_class=HTMLResponse)
async def histogram_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk histogram
    return templates.TemplateResponse("histogram.html", {"request": request})

# Endpoint ini digunakan untuk menampilkan histogram dari gambar yang diupload.
@app.post("/histogram/", response_class=HTMLResponse)
async def generate_histogram(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Pastikan gambar berhasil diimpor
    if img is None:
        return HTMLResponse("Tidak dapat membaca gambar yang diunggah", status_code=400)

    # Buat histogram grayscale dan berwarna
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_histogram_path = save_histogram(gray_img, "grayscale")

    color_histogram_path = save_color_histogram(img)

    return templates.TemplateResponse("histogram.html", {
        "request": request,
        "grayscale_histogram_path": grayscale_histogram_path,
        "color_histogram_path": color_histogram_path
    })


# Endpoint ini digunakan untuk melakukan equalisasi histogram pada gambar yang diupload.
@app.get("/equalize/", response_class=HTMLResponse)
async def equalize_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk equalisasi histogram
    return templates.TemplateResponse("equalize.html", {"request": request})

# Endpoint ini digunakan untuk melakukan equalisasi histogram pada gambar yang diupload.
@app.post("/equalize/", response_class=HTMLResponse)
async def equalize_histogram(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    equalized_img = cv2.equalizeHist(img)

    original_path = save_image(img, "original")
    modified_path = save_image(equalized_img, "equalized")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

# Endpoint ini digunakan untuk melakukan spesifikasi histogram pada gambar yang diupload berdasarkan gambar referensi.
@app.get("/specify/", response_class=HTMLResponse)
async def specify_form(request: Request):
    # Menampilkan halaman untuk upload gambar dan referensi untuk spesifikasi histogram
    return templates.TemplateResponse("specify.html", {"request": request})

# Endpoint ini digunakan untuk melakukan spesifikasi histogram pada gambar yang diupload berdasarkan gambar referensi.
@app.post("/specify/", response_class=HTMLResponse)
async def specify_histogram(request: Request, file: UploadFile = File(...), ref_file: UploadFile = File(...)):
    # Baca gambar yang diunggah dan gambar referensi
    image_data = await file.read()
    ref_image_data = await ref_file.read()

    np_array = np.frombuffer(image_data, np.uint8)
    ref_np_array = np.frombuffer(ref_image_data, np.uint8)
		
		#jika ingin grayscale
    #img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    #ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_GRAYSCALE)

    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Membaca gambar dalam format BGR
    ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_COLOR)  # Membaca gambar referensi dalam format BGR


    if img is None or ref_img is None:
        return HTMLResponse("Gambar utama atau gambar referensi tidak dapat dibaca.", status_code=400)

    # Spesifikasi histogram menggunakan match_histograms dari skimage #grayscale
    #specified_img = match_histograms(img, ref_img, multichannel=False)
		    # Spesifikasi histogram menggunakan match_histograms dari skimage untuk gambar berwarna
    specified_img = match_histograms(img, ref_img, channel_axis=-1)
    # Konversi kembali ke format uint8 jika diperlukan
    specified_img = np.clip(specified_img, 0, 255).astype('uint8')

    original_path = save_image(img, "original")
    modified_path = save_image(specified_img, "specified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

# Endpoint ini digunakan untuk menampilkan statistik dasar dari gambar yang diupload, seperti mean dan standar deviasi.
@app.post("/statistics/", response_class=HTMLResponse)
async def calculate_statistics(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    mean_intensity = np.mean(img)
    std_deviation = np.std(img)

    image_path = save_image(img, "statistics")

    return templates.TemplateResponse("statistics.html", {
        "request": request,
        "mean_intensity": mean_intensity,
        "std_deviation": std_deviation,
        "image_path": image_path
    })

def save_image(image, prefix):
    filename = f"{prefix}_{uuid4()}.png"
    path = os.path.join("static/uploads", filename)
    cv2.imwrite(path, image)
    return f"/static/uploads/{filename}"

def save_histogram(image, prefix):
    histogram_path = f"static/histograms/{prefix}_{uuid4()}.png"
    plt.figure()
    plt.hist(image.ravel(), 256, [0, 256])
    plt.savefig(histogram_path)
    plt.close()
    return f"/{histogram_path}"

def save_color_histogram(image):
    color_histogram_path = f"static/histograms/color_{uuid4()}.png"
    plt.figure()
    for i, color in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.savefig(color_histogram_path)
    plt.close()
    return f"/{color_histogram_path}"

def handle_dimension_mismatch(img1, img2):
    """
    Menangani perbedaan dimensi antara dua citra dengan crop ke dimensi terkecil
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Ambil dimensi terkecil
    min_height = min(h1, h2)
    min_width = min(w1, w2)
    
    # Crop kedua citra ke ukuran yang sama (dari pusat)
    # Untuk img1
    start_y1 = (h1 - min_height) // 2
    start_x1 = (w1 - min_width) // 2
    img1_cropped = img1[start_y1:start_y1+min_height, start_x1:start_x1+min_width]
    
    # Untuk img2
    start_y2 = (h2 - min_height) // 2
    start_x2 = (w2 - min_width) // 2
    img2_cropped = img2[start_y2:start_y2+min_height, start_x2:start_x2+min_width]
    
    return img1_cropped, img2_cropped, (min_height, min_width), (h1, w1), (h2, w2)

