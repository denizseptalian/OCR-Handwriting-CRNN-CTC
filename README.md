# Pembaca Patok Blok/TPH — Deploy ke Streamlit Cloud

## Struktur repo (semua file di root, sejajar dengan app.py)

```
repo/
├── app.py                  <- aplikasi (versi selaras model baru)
├── char_cnn_fp16.tflite    <- model dari training BARU (CharCNN_DataFIks)
├── labels.txt              <- dari training yang sama
├── requirements.txt
├── packages.txt
├── .gitignore
└── .streamlit/
    └── config.toml
```

## Langkah deploy

1. **Ambil model dari Google Drive** hasil notebook perbaikan:
   `MyDrive/CharCNN_DataFIks/char_cnn_fp16.tflite` dan `labels.txt`.
   PENTING: harus dari training BARU. Model lama dilatih dengan
   preprocessing berbeda dan akurasinya akan kacau di app ini.

2. **Push ke GitHub sebagai file biasa, BUKAN Git LFS.**
   Streamlit Cloud tidak mengunduh file LFS — app akan menampilkan
   error "pointer Git LFS" kalau ini terjadi. FP16 (~1.8 MB) aman
   jauh di bawah batas 100 MB GitHub, jadi LFS tidak diperlukan.
   Kalau repo lama terlanjur pakai LFS: `git lfs untrack "*.tflite"`,
   hapus dari .gitattributes, lalu commit ulang file-nya.

3. **Deploy di share.streamlit.io** → New app → pilih repo/branch →
   Main file path: `app.py` → Advanced settings → Python 3.11 → Deploy.

4. **Uji pertama:** buka dari HP, ambil foto patok. Kalau hasil aneh,
   coba matikan toggle "Hapus background" di ⚙️ Pengaturan —
   itu satu-satunya langkah yang tidak ada di pipeline training.

## Catatan

- App otomatis memilih model yang ada: fp16 → fp32 → int8.
  Cukup satu file model di repo (fp16 disarankan).
- Suara (gTTS) butuh internet dari sisi server — di Streamlit Cloud
  selalu tersedia; hanya gagal kalau HP pengguna offline saat memutar.
- `packages.txt` memasang libglib2.0-0 untuk OpenCV headless.
