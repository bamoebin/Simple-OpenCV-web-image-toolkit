# FastAPI OpenCV Image Processor

Aplikasi web untuk pemrosesan citra digital menggunakan FastAPI dan OpenCV dengan fitur crop otomatis untuk operasi bitwise.

## 🚀 Fitur Utama

### 1. **Operasi Aritmatika Citra**
- Penjumlahan (Add)
- Pengurangan (Subtract) 
- Maksimum (Max)
- Minimum (Min)
- Inverse

### 2. **Operasi Logika Bitwise** ⭐ NEW!
- **AND** - Intersection antara dua citra
- **XOR** - Deteksi perbedaan antara dua citra
- **NOT** - Negasi/inverse citra
- **🔧 Auto Crop Feature**: Jika dimensi citra berbeda, sistem otomatis melakukan crop dari bagian tengah ke ukuran terkecil

### 3. **Pemrosesan Citra Lainnya**
- Konversi Grayscale
- Histogram Generation & Visualization
- Histogram Equalization
- Histogram Specification
- Statistical Analysis (Mean & Standard Deviation)

## 🛠️ Teknologi yang Digunakan

- **FastAPI** - Modern, fast web framework
- **OpenCV** - Computer vision dan image processing
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Scikit-image** - Image processing algorithms
- **Jinja2** - Template engine
- **Bootstrap** - Responsive UI framework

## 📦 Instalasi

1. Clone repository ini:
```bash
git clone https://github.com/bamoebin/Simple-OpenCV-web-image-toolkit-.git
cd Simple-OpenCV-web-image-toolkit-
```

2. Install dependencies:
```bash
pip install fastapi uvicorn opencv-python numpy matplotlib scikit-image python-multipart
```

3. Jalankan aplikasi:
```bash
uvicorn main:app --reload
```

4. Buka browser dan akses `http://127.0.0.1:8000`

## 🎯 Fitur Baru: Auto Crop untuk Operasi Bitwise

### Masalah yang Dipecahkan
Operasi bitwise (AND, XOR) memerlukan kedua citra memiliki dimensi yang sama persis. Sebelumnya, jika user mengupload citra dengan dimensi berbeda, akan terjadi error.

### Solusi yang Diterapkan
1. **Deteksi Otomatis**: Sistem mendeteksi perbedaan dimensi secara otomatis
2. **Center Crop**: Melakukan crop dari bagian tengah kedua citra ke dimensi terkecil
3. **Visual Feedback**: Menampilkan informasi detail tentang proses crop yang dilakukan
4. **Comparison Display**: Menampilkan citra original, citra yang di-crop, dan hasil operasi

### Cara Kerja Auto Crop
```python
def handle_dimension_mismatch(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Ambil dimensi terkecil
    min_height = min(h1, h2)
    min_width = min(w1, w2)
    
    # Crop dari bagian tengah
    img1_cropped = img1[start_y1:start_y1+min_height, start_x1:start_x1+min_width]
    img2_cropped = img2[start_y2:start_y2+min_height, start_x2:start_x2+min_width]
    
    return img1_cropped, img2_cropped, dimensions_info
```

## 📱 Penggunaan

### Operasi Logika dengan Auto Crop:
1. Akses halaman "Operasi Logika" dari homepage
2. Upload gambar pertama
3. Pilih operasi (AND/XOR/NOT)
4. Upload gambar kedua (untuk AND/XOR)
5. Sistem akan otomatis mendeteksi dan menangani perbedaan dimensi
6. Lihat hasil dengan informasi detail tentang proses crop

## 🎓 Tujuan Edukatif

Aplikasi ini dirancang untuk pembelajaran Digital Image Processing dengan:
- **Visual Learning**: Immediate feedback untuk setiap operasi
- **Interactive Interface**: User-friendly untuk fokus pada konsep
- **Comprehensive Coverage**: Dari operasi dasar hingga advanced
- **Real-world Problem Solving**: Auto crop feature mengatasi masalah praktis

## 📁 Struktur Project

```
fastapi-opencv-26agustus/
├── main.py                 # Backend logic & API endpoints
├── templates/              # HTML templates
│   ├── base.html
│   ├── home.html
│   ├── logic_operation.html # Form operasi logika
│   ├── logic_result.html   # Display hasil dengan info crop
│   └── ...
├── static/
│   ├── uploads/           # Citra hasil pemrosesan
│   └── histograms/        # Grafik histogram
└── README.md
```

## 🔍 Detail Teknis Auto Crop

- **Algoritma**: Center crop untuk mempertahankan area penting
- **Strategi**: Menggunakan dimensi terkecil dari kedua citra
- **Preservation**: Mempertahankan aspect ratio dalam area yang di-crop
- **Performance**: Efficient memory usage dengan NumPy operations

## 🤝 Kontribusi

Contributions are welcome! Please feel free to submit a Pull Request.

**Developed for Digital Image Processing Learning** 📚
