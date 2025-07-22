# R-Sense-Code

## 📝 Catatan Teknis Penting

* **Fungsi Utama**: Aplikasi ini berfungsi sebagai **backend service**. Tugas utamanya adalah menerima data, memprosesnya dengan model ML, dan mengirimkan hasilnya ke `WEBHOOK_URL` yang telah ditentukan di dalam kode.
* **Lokasi Model**: Semua file Machine Learning (`scaler`, `pca`, dan model `.pkl`) harus ditempatkan di dalam direktori `./model/`.
* **Uji Coba dengan Data Dummy**: Kode ini sudah dilengkapi dengan generator data dummy untuk sensor dan GPS. Ini memungkinkan pengujian tanpa perlu perangkat IoT fisik. Lihat bagian "Cara Menjalankan" untuk instruksi aktivasi.

## 🚀 Instalasi dan Pengaturan

### Prasyarat
* Python 3.8+
* pip (Python package installer)

### Langkah-langkah Instalasi

1.  **Clone repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[username-anda]/[nama-repository-anda].git
    cd [nama-repository-anda]
    ```

2.  **Buat dan aktifkan virtual environment (wajib):**
    * **macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install semua dependensi:**
    Buat file `requirements.txt` dengan konten di bawah ini:
    ```
    Flask
    numpy
    pandas
    requests
    scikit-learn
    joblib
    scipy
    paho-mqtt
    urllib3
    pywavelets
    ```
    Lalu jalankan perintah instalasi:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Cara Menjalankan

### 1. Menjalankan Server
Pastikan virtual environment sudah aktif, lalu jalankan server Flask dengan perintah:
```bash
python app.py
```
Server akan berjalan di `http://0.0.0.0:5000`.

### 2. (Opsional) Menggunakan Data Dummy untuk Uji Coba
Untuk melakukan uji coba tanpa mengirim data dari perangkat eksternal, Anda bisa mengaktifkan generator data dummy di dalam file `app.py`.

* **Untuk mengaktifkan data sensor dummy:**
    Di dalam fungsi `process_loop`, lakukan modifikasi berikut:
    ```python
    # Comment baris ini untuk menonaktifkan penerimaan data dari antrian
    # raw = self.raw_queue.get(timeout=1) 
    
    # Uncomment baris ini untuk menggunakan data getaran dummy
    raw = self.generate_dummy_vibration()
    ```

* **Untuk mengaktifkan data GPS dummy:**
    Masih di dalam fungsi `process_loop`, lakukan modifikasi berikut:
    ```python
    # Comment baris ini untuk mencari GPS asli
    # gps = self.find_closest_gps(ntp_time) or ...
    
    # Uncomment baris ini untuk menggunakan data GPS dummy
    gps = self.generate_dummy_gps()
    ```

## 📁 Struktur Proyek
```
[nama-proyek]/
│
├── model/
│   ├── rf_model_updated_waveletss.pkl
│   ├── scaler_updated_waveletss.pkl
│   └── pca_model_updated_waveletss.pkl
│
├── templates/
│   └── index.html
│
├── app.py              # Kode utama aplikasi Flask
├── requirements.txt    # Daftar library yang dibutuhkan
└── README.md           # File ini
