# Sistem Rekomendasi Produk

Repositori ini berisi sistem rekomendasi produk menggunakan clustering KMeans dan SVD (Singular Value Decomposition) untuk collaborative filtering. Aplikasi ini dibangun dengan Streamlit untuk visualisasi interaktif.

## Struktur Proyek

```
/RepositoriAnda
│
├── /data
│   └── clean_data_manual.csv       # Dataset yang digunakan untuk pelatihan dan clustering
│
├── /src
│   └── kmeansvd.py                 # Script untuk clustering dan pelatihan model
│   └── model.pkl                   # Model dan hasil clustering yang disimpan
│
├── /App
│   └── app.py                      # Aplikasi Streamlit untuk rekomendasi produk
│
└── README.md
└── requirements.txt
```

## Cara Menjalankan Proyek

### 1. Clone Repositori

```bash
git clone https://github.com/username-anda/repositori-anda.git
cd repositori-anda
```

### 2. Install Dependensi

Buat environment virtual (opsional tapi disarankan):

```bash
python -m venv env
source env/bin/activate  # Untuk Linux/Mac
# atau
env\Scripts\activate    # Untuk Windows
```

Install library yang diperlukan:

```bash
pip install -r requirements.txt
```

### 3. Siapkan Data

Pastikan file dataset `clean_data_manual.csv` ditempatkan di folder `/data`.

### 4. Latih Model

Jalankan script pelatihan untuk menghasilkan model clustering dan sistem rekomendasi:

```bash
python src/kmeansvd.py
```

### 5. Jalankan Aplikasi Streamlit

Mulai aplikasi Streamlit untuk rekomendasi produk:

```bash
streamlit run App/app.py
```

Buka link localhost yang disediakan di browser Anda untuk berinteraksi dengan aplikasi.

## Fitur

- Clustering produk berdasarkan ulasan dan rating menggunakan KMeans.
- Rekomendasi berbasis collaborative filtering menggunakan SVD.
- Aplikasi web interaktif menggunakan Streamlit.
- Menampilkan metrik evaluasi seperti RMSE, MAE, dan Davies-Bouldin Index.

## Dependensi

Lihat `requirements.txt` untuk semua paket yang dibutuhkan.

## Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE), sehingga Anda bebas menggunakan, mengubah, dan mendistribusikan ulang dengan syarat mencantumkan kredit kepada penulis asli.
---

