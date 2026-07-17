# ============================================================
# Pembaca Patok Blok/TPH — CharCNN TFLite + Streamlit (versi HP)
# Fitur: kamera langsung, deteksi garis pemisah, baca Blok/TPH,
#        hasil besar mudah dibaca, suara otomatis (TTS Indonesia)
#
# SELARAS DENGAN MODEL BARU (CharCNN_Patok_Perbaikan.ipynb):
#  - Binarisasi = CLAHE + adaptiveThreshold + fallback Otsu + close(3,3)x1
#    (identik dengan preprocessing training -> akurasi konsisten)
#  - char_to_input tanpa deskew (model tidak dilatih dengan deskew)
#  - Pemisahan karakter menempel (proyeksi vertikal)
#  - Constraint decoding: Blok = huruf lalu angka, TPH = angka
#  - Dukungan model INT8 (skala kuantisasi otomatis)
# ============================================================
import io
import os
import re

import cv2
import numpy as np
import streamlit as st
from gtts import gTTS
from PIL import Image

# --- TFLite interpreter: LiteRT (pengganti resmi tflite-runtime) ---
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

# ============================================================
# Konfigurasi
# ============================================================
# Cari file model yang tersedia (urutan: FP16 disarankan, lalu FP32, INT8)
_KANDIDAT_MODEL = ["char_cnn_fp16.tflite", "char_cnn_fp32.tflite", "char_cnn_int8.tflite"]
MODEL_PATH = next((m for m in _KANDIDAT_MODEL if os.path.exists(m)), _KANDIDAT_MODEL[0])
LABELS_PATH = "labels.txt"
IMG_SIZE = 32

# Isi dengan URL aplikasi "Ripeness Detector" (scan buah sawit) Anda yang sudah
# online, agar tombol "Lanjut Scan Buah Sawit" bisa langsung membukanya.
# Kosongkan ("") kalau belum ada / belum mau dipakai dulu.
URL_APP_SCAN_BUAH = ""

st.set_page_config(
    page_title="Pembaca Patok",
    page_icon="🌴",
    layout="centered",                      # layout sempit = pas untuk HP
    initial_sidebar_state="collapsed",
)

# --- CSS untuk tampilan mobile: font besar, tombol lebar, kartu hasil ---
st.markdown("""
<style>
/* Rapatkan padding atas supaya hemat layar HP */
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* Tombol & input full-width, tinggi nyaman untuk jempol */
.stButton > button, .stDownloadButton > button {
    width: 100%; min-height: 3rem; font-size: 1.1rem; border-radius: 12px;
}

/* Kartu hasil besar */
.hasil-card {
    border-radius: 16px; padding: 1rem 1.2rem; margin: 0.4rem 0;
    text-align: center;
}
.hasil-blok { background: #dcfce7; border: 2px solid #22c55e; }
.hasil-tph  { background: #fee2e2; border: 2px solid #ef4444; }
.hasil-label { font-size: 0.95rem; font-weight: 600; color: #374151;
               text-transform: uppercase; letter-spacing: 1px; }
.hasil-nilai { font-size: 3.2rem; font-weight: 800; line-height: 1.1;
               font-family: monospace; color: #111827; }

/* Judul lebih ringkas di HP */
h1 { font-size: 1.5rem !important; }

/* Kamera: frame rasio 4:5 (pas untuk patok Blok/garis/TPH), bukan full portrait */
[data-testid="stCameraInput"] { width: 100% !important; position: relative; }
[data-testid="stCameraInput"] > div { width: 100% !important; }
[data-testid="stCameraInput"] video {
    width: 100% !important;
    aspect-ratio: 4 / 5;
    height: auto !important;
    max-height: 55vh;
    object-fit: cover;             /* preview = persis area yang akan diproses */
    border-radius: 12px;
}
/* Panduan bidik: kotak putus-putus di tengah preview */
[data-testid="stCameraInput"]::after {
    content: "";
    position: absolute;
    top: 8%; left: 12%; right: 12%; bottom: 22%;
    border: 3px dashed rgba(255, 255, 255, 0.75);
    border-radius: 14px;
    pointer-events: none;
}
/* Garis bantu tengah: sejajarkan garis patok (pemisah Blok/TPH) dengan garis ini */
[data-testid="stCameraInput"]::before {
    content: "";
    position: absolute;
    top: 43%;                       /* tengah vertikal kotak panduan */
    left: 12%; right: 12%;
    border-top: 3px dashed rgba(59, 130, 246, 0.9);   /* biru */
    pointer-events: none;
    z-index: 2;
}
[data-testid="stCameraInput"] img {
    width: 100% !important;
    height: auto !important;
    border-radius: 12px;
}
[data-testid="stCameraInput"] button {
    min-height: 3.2rem; font-size: 1.15rem; border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    # --- Diagnosa file model sebelum dimuat ---
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model tidak ditemukan: `{MODEL_PATH}`. "
                 f"Pastikan file ada di root repo, sejajar dengan app.py. "
                 f"Isi folder saat ini: {os.listdir('.')}")
        st.stop()

    size_kb = os.path.getsize(MODEL_PATH) / 1024
    with open(MODEL_PATH, "rb") as f:
        header = f.read(64)

    # File TFLite asli punya magic 'TFL3' di byte ke-4..8
    if header[4:8] != b"TFL3":
        if header.startswith(b"version https://git-lfs"):
            st.error(f"File model adalah pointer Git LFS ({size_kb:.0f} KB), bukan model asli. "
                     f"Streamlit Cloud tidak mengunduh file LFS — push ulang sebagai file biasa.")
        else:
            st.error(f"File model rusak/bukan TFLite valid ({size_kb:.0f} KB). "
                     f"Download ulang dari Google Drive dan push ulang.")
        st.stop()

    interp = Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            classes = [line.strip() for line in f if line.strip()]
    else:
        classes = [str(i) for i in range(10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    return interp, inp, out, classes


interp, inp, out, CLASSES = load_model()


# ============================================================
# Pipeline deteksi — IDENTIK dengan notebook perbaikan
# ============================================================
DIGIT_IDX = list(range(10))
LETTER_IDX = list(range(10, len(CLASSES)))

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_qr_detector = cv2.QRCodeDetector()

# Pola payload QR sesuai format generator: "AFD {Afdeling} - BLOK {Blok} - TPH {Nomor}"
_POLA_QR = re.compile(r"AFD\s*([^\-]+?)\s*-\s*BLOK\s*([^\-]+?)\s*-\s*TPH\s*(.+)", re.IGNORECASE)


def baca_qr(bgr):
    """Coba deteksi & decode QR Code pada gambar (mendukung >1 QR sekaligus).
    Return teks payload QR pertama yang berhasil dibaca, atau None kalau
    tidak ada QR Code yang terdeteksi sama sekali."""
    try:
        retval, decoded_info, _, _ = _qr_detector.detectAndDecodeMulti(bgr)
        if retval:
            for teks in decoded_info:
                if teks:
                    return teks
    except Exception:
        pass
    # Fallback: deteksi single QR (kadang lebih andal utk 1 QR yang jelas)
    try:
        data, _, _ = _qr_detector.detectAndDecode(bgr)
        return data or None
    except Exception:
        return None


def urai_payload_qr(teks):
    """Urai payload QR menjadi dict {afdeling, blok, tph}, atau None kalau
    formatnya tidak cocok dengan pola generator barcode TPH."""
    if not teks:
        return None
    m = _POLA_QR.search(teks)
    if not m:
        return None
    afdeling, blok, tph = (g.strip() for g in m.groups())
    return {"afdeling": afdeling, "blok": blok, "tph": tph}


def binarize_full(gray, block_size=41, c_thresh=15):
    """Grayscale -> biner (teks=255). SAMA dengan training:
    CLAHE + adaptif + fallback Otsu + close(3,3) x1."""
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    gray = _clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=block_size, C=c_thresh)
    fg = (binary > 0).mean()
    # Hasil adaptif tak wajar (blur/kontras rendah) -> Otsu
    if fg < 0.02 or fg > 0.60:
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return binary


def x_overlap_ratio(a, b):
    left = max(a[0], b[0])
    right = min(a[0] + a[2], b[0] + b[2])
    if right <= left:
        return 0.0
    return (right - left) / min(a[2], b[2])


def y_gap(a, b):
    return max(0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))


def to_model_input(char_bin):
    """Crop biner satu karakter -> input model 32x32 [0,1].

    IDENTIK dengan char_to_input di training (tanpa deskew — model
    tidak dilatih dengan deskew; menambahkannya justru menggeser
    distribusi input dari yang dipelajari model).
    """
    inv = 255 - char_bin
    hh, ww = inv.shape
    side = int(max(hh, ww) * 1.3)
    canvas = np.full((side, side), 255, np.uint8)
    y0 = (side - hh) // 2
    x0 = (side - ww) // 2
    canvas[y0:y0 + hh, x0:x0 + ww] = inv
    small = cv2.resize(canvas, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)[None, :, :, None] / 255.0


def split_touching(char_bin):
    """Pisah karakter menempel via lembah proyeksi vertikal.
    Return daftar (x, y, w, h) relatif terhadap crop."""
    h, w = char_bin.shape
    n = max(1, int(round(w / (h * 0.9))))
    if n == 1:
        return [(0, 0, w, h)]
    col = (char_bin > 0).sum(0).astype(float)
    ksz = max(1, w // 30)
    if ksz > 1:
        col = np.convolve(col, np.ones(ksz) / ksz, mode="same")
    min_gap = int(0.4 * h)
    cuts = []
    for c in np.argsort(col):
        c = int(c)
        if c < min_gap or c > w - min_gap:
            continue
        if all(abs(c - x) >= min_gap for x in cuts):
            cuts.append(c)
        if len(cuts) == n - 1:
            break
    bounds = [0] + sorted(cuts) + [w]
    out = []
    for i in range(len(bounds) - 1):
        x0, x1 = bounds[i], bounds[i + 1]
        if x1 - x0 >= 0.15 * h:
            out.append((x0, 0, x1 - x0, h))
    return out or [(0, 0, w, h)]


def predict_char(xin, allowed_idx=None, top_k=3):
    """Prediksi 1 karakter dengan constraint decoding.

    allowed_idx: batasi ke indeks kelas tertentu (DIGIT_IDX / LETTER_IDX).
    Mendukung model float (FP32/FP16) maupun INT8 (kuantisasi otomatis).
    Return (label_terbaik, conf_terbaik, [top-k alternatif]).
    """
    x = xin
    if inp["dtype"] == np.uint8:                 # model INT8
        scale, zp = inp["quantization"]
        x = np.round(x / scale + zp).astype(np.uint8)
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    prob = interp.get_tensor(out["index"])[0].astype(np.float32)
    if out["dtype"] == np.uint8:                 # de-kuantisasi output INT8
        o_scale, o_zp = out["quantization"]
        prob = (prob - o_zp) * o_scale
    if allowed_idx is not None:
        mask = np.zeros_like(prob)
        mask[allowed_idx] = prob[allowed_idx]
        prob = mask
    order = prob.argsort()[::-1][:top_k]
    alternatif = [(CLASSES[i], float(prob[i])) for i in order]
    return alternatif[0][0], alternatif[0][1], alternatif


def baca_patok(bgr, block_size=41, c_thresh=15, min_h_ratio=0.05,
               hapus_bg=True, hilangkan_noise=True, auto_roi=True):
    """Deteksi garis pemisah + baca Blok (atas) & TPH (bawah).

    hapus_bg: ratakan background (noda, bayangan, gradasi cahaya) SEBELUM threshold.
    hilangkan_noise: buang blob non-karakter (goresan, kotoran, bintik) SEBELUM klasifikasi.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 0) HAPUS BACKGROUND (preprocessing sebelum model):
    #    estimasi background dengan blur besar, lalu bagi -> noda/bayangan/gradasi rata,
    #    yang tersisa hanya goresan gelap (tulisan)
    if hapus_bg:
        k = max(31, (min(gray.shape) // 10) | 1)   # kernel besar, selalu ganjil
        bg = cv2.medianBlur(gray, min(k, 99))
        gray = cv2.divide(gray, bg, scale=255)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 1) Binarisasi IDENTIK dengan training (CLAHE + adaptif + Otsu fallback)
    binary = binarize_full(gray, block_size=block_size, c_thresh=c_thresh)

    H_img, W_img = binary.shape

    # 1a) CARI GARIS PATOK via Hough — jauh lebih andal daripada morfologi untuk
    #     garis tipis/agak miring, dan tidak peduli seberapa lebar frame di sekitarnya.
    def cari_garis(bin_img):
        bin_img = np.ascontiguousarray(bin_img, dtype=np.uint8)
        Hh, Ww = bin_img.shape
        min_len = max(20, int(0.15 * Ww))
        lines = cv2.HoughLinesP(bin_img, 1, np.pi / 180, threshold=35,
                                minLineLength=min_len, maxLineGap=max(8, Ww // 40))
        if lines is None:
            return None
        best = None
        for l in lines:
            x1, y1, x2, y2 = np.array(l).ravel()[:4]
            dx, dy = x2 - x1, y2 - y1
            length = (dx ** 2 + dy ** 2) ** 0.5
            if length < min_len:
                continue
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            if not (angle < 20 or angle > 160):   # harus mendekati horizontal
                continue
            if best is None or length > best[0]:
                best = (length, min(x1, x2), max(x1, x2), (y1 + y2) // 2)
        if best is None:
            return None
        _, xl, xr, yc = best
        return xl, xr, yc

    garis_hasil = cari_garis(binary)

    # 1b) FOKUS OTOMATIS: kalau garis ditemukan, proses hanya area di sekitarnya.
    #     Membuat hasil tidak tergantung framing kamera - coretan lain otomatis terabaikan.
    if auto_roi and garis_hasil is not None:
        xl, xr, yc = garis_hasil
        lw = xr - xl
        mx = int(0.25 * lw)
        my = int(1.4 * lw)
        x0 = max(0, xl - mx)
        x1 = min(W_img, xr + mx)
        y0 = max(0, yc - my)
        y1 = min(H_img, yc + my)
        if (x1 - x0) < 0.9 * W_img or (y1 - y0) < 0.9 * H_img:
            return baca_patok(bgr[y0:y1, x0:x1], block_size, c_thresh, min_h_ratio,
                              hapus_bg, hilangkan_noise, auto_roi=False)

    # 1c) HAPUS PITA GARIS dari citra biner SEBELUM analisis komponen.
    #     Ini kunci: garis yang menyentuh/berdekatan dengan karakter bisa
    #     menyatukan tulisan Blok dan TPH jadi satu blob raksasa via closing.
    #     Menghapus pitanya dulu memutus sambungan itu.
    sep_box = None
    if garis_hasil is not None:
        xl, xr, yc = garis_hasil
        tebal_garis = max(4, int(0.02 * H_img))
        y0b = max(0, yc - tebal_garis)
        y1b = min(H_img, yc + tebal_garis)
        x0b = max(0, xl - int(0.03 * W_img))
        x1b = min(W_img, xr + int(0.03 * W_img))
        sep_box = (x0b, y0b, x1b - x0b, y1b - y0b)
        binary[y0b:y1b, x0b:x1b] = 0
        separator_y = yc
        sep_found = True
    else:
        separator_y = H_img // 2
        sep_found = False

    # 1d) HILANGKAN NOISE: buang blob yang jelas bukan karakter (garis sudah dibuang di atas)
    if hilangkan_noise:
        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = np.zeros(n_lbl, dtype=bool)
        for i in range(1, n_lbl):
            x, y, w, hh, area = stats[i]
            density = area / max(w * hh, 1)
            # Bintik/kotoran kecil
            if area < 0.0008 * H_img * W_img:
                continue
            # Terlalu pendek untuk jadi karakter
            if hh < 0.6 * min_h_ratio * H_img:
                continue
            # Goresan tipis panjang: bbox besar tapi isinya kosong
            if density < 0.06:
                continue
            # Noda pekat hampir kotak penuh (tulisan tidak pernah sepadat ini)
            if density > 0.92 and w > 0.03 * W_img and hh > 0.03 * H_img:
                continue
            keep[i] = True
        binary = np.where(keep[lbl], 255, 0).astype(np.uint8)

    # 2) Kontur karakter (garis pemisah sudah tidak ada di citra biner)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_contours = [cv2.boundingRect(c) for c in contours
                     if cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] > 0.001 * H_img * W_img]

    # 3) Filter + merge fragmen (tidak lintas garis pemisah, tidak lintas jarak jauh)
    boxes = [list(b) for b in char_contours if b[3] > min_h_ratio * H_img]

    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                a_atas = (a[1] + a[3] / 2) < separator_y
                b_atas = (b[1] + b[3] / 2) < separator_y
                if a_atas != b_atas:
                    continue
                # Jangan gabung kalau tinggi keduanya jauh berbeda (kemungkinan bukan
                # fragmen huruf yang sama, tapi dua karakter berbeda yang kebetulan dekat)
                if max(a[3], b[3]) > 2.2 * min(a[3], b[3]):
                    continue
                if x_overlap_ratio(a, b) > 0.4 and y_gap(a, b) < 0.4 * max(a[3], b[3]):
                    x0 = min(a[0], b[0])
                    y0 = min(a[1], b[1])
                    x1 = max(a[0] + a[2], b[0] + b[2])
                    y1 = max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    boxes.pop(j)
                    merged = True
                    break
            if merged:
                break

    # 3b) Buang box yang menempel tepi gambar (karakter asli selalu utuh di dalam frame)
    tepi = max(2, int(0.005 * min(H_img, W_img)))
    boxes = [b for b in boxes
             if b[1] > tepi and (b[1] + b[3]) < H_img - tepi
             and b[0] > tepi and (b[0] + b[2]) < W_img - tepi]

    # 4) Bagi ke Blok (atas) & TPH (bawah)
    blok_boxes = [b for b in boxes if (b[1] + b[3] / 2) < separator_y]
    tph_boxes = [b for b in boxes if (b[1] + b[3] / 2) >= separator_y]

    # 4b) Di tiap sisi, kelompokkan per baris dan ambil HANYA baris terdekat garis pemisah.
    #     Tulisan Blok selalu tepat di atas garis, TPH tepat di bawah — blob lain
    #     (coretan di pojok atas/bawah) berada di baris berbeda dan dibuang.
    def baris_terdekat(box_list, ambil_terbawah):
        if len(box_list) <= 1:
            return box_list
        urut = sorted(box_list, key=lambda b: b[1] + b[3] / 2)
        rows = []
        for b in urut:
            cy = b[1] + b[3] / 2
            placed = False
            for row in rows:
                row_cy = np.mean([rb[1] + rb[3] / 2 for rb in row])
                row_h = np.mean([rb[3] for rb in row])
                if abs(cy - row_cy) < 0.7 * row_h:
                    row.append(b)
                    placed = True
                    break
            if not placed:
                rows.append([b])
        if len(rows) == 1:
            return rows[0]
        rows.sort(key=lambda row: np.mean([rb[1] + rb[3] / 2 for rb in row]))
        return rows[-1] if ambil_terbawah else rows[0]

    blok_boxes = sorted(baris_terdekat(blok_boxes, ambil_terbawah=True), key=lambda b: b[0])
    tph_boxes = sorted(baris_terdekat(tph_boxes, ambil_terbawah=False), key=lambda b: b[0])

    # 4c) PISAH KARAKTER MENEMPEL (mis. "23" yang catnya menyatu) via
    #     lembah proyeksi vertikal, lalu urutkan ulang kiri->kanan
    def pecah_menempel(box_list):
        hasil = []
        for (x, y, w, hh) in box_list:
            for (sx, sy, sw, sh) in split_touching(binary[y:y + hh, x:x + w]):
                hasil.append([x + sx, y + sy, sw, sh])
        return sorted(hasil, key=lambda b: b[0])

    blok_boxes = pecah_menempel(blok_boxes)
    tph_boxes = pecah_menempel(tph_boxes)

    # 5) Prediksi + CONSTRAINT DECODING (format patok tetap):
    #    Blok = karakter pertama HURUF, sisanya ANGKA (P17, A1, ...)
    #    TPH  = ANGKA semua
    blok_preds = []
    for i, (x, y, w, hh) in enumerate(blok_boxes):
        allowed = LETTER_IDX if i == 0 else DIGIT_IDX
        blok_preds.append(
            predict_char(to_model_input(binary[y:y + hh, x:x + w]), allowed_idx=allowed))
    tph_preds = [predict_char(to_model_input(binary[y:y + hh, x:x + w]),
                              allowed_idx=DIGIT_IDX)
                 for (x, y, w, hh) in tph_boxes]

    nomor_blok = "".join(p[0] for p in blok_preds)
    nomor_tph = "".join(p[0] for p in tph_preds)

    # 6) Visualisasi (garis & font tebal supaya terlihat di layar kecil)
    vis = rgb.copy()
    tebal = max(2, W_img // 300)
    font_scale = max(0.8, W_img / 600)
    if sep_box is not None:
        sx, sy, sw, sh = sep_box
        cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), tebal)
    cv2.line(vis, (0, separator_y), (W_img, separator_y), (0, 0, 255), tebal)
    for (x, y, w, hh), (ch, cf, _) in zip(blok_boxes, blok_preds):
        cv2.rectangle(vis, (x, y), (x + w, y + hh), (0, 200, 0), tebal)
        cv2.putText(vis, ch, (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 200, 0), tebal + 1)
    for (x, y, w, hh), (ch, cf, _) in zip(tph_boxes, tph_preds):
        cv2.rectangle(vis, (x, y), (x + w, y + hh), (255, 0, 0), tebal)
        cv2.putText(vis, ch, (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), tebal + 1)

    # 7) Versi background dihapus: kertas putih bersih + tulisan hitam
    bersih = cv2.cvtColor(255 - binary, cv2.COLOR_GRAY2RGB)
    vis_bersih = bersih.copy()
    if sep_box is not None:
        sx, sy, sw, sh = sep_box
        cv2.rectangle(vis_bersih, (sx, sy), (sx + sw, sy + sh), (255, 200, 0), tebal)
    cv2.line(vis_bersih, (0, separator_y), (W_img, separator_y), (0, 0, 255), tebal)
    for (x, y, w, hh), (ch, cf, _) in zip(blok_boxes, blok_preds):
        cv2.rectangle(vis_bersih, (x, y), (x + w, y + hh), (0, 170, 0), tebal)
        cv2.putText(vis_bersih, ch, (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 170, 0), tebal + 1)
    for (x, y, w, hh), (ch, cf, _) in zip(tph_boxes, tph_preds):
        cv2.rectangle(vis_bersih, (x, y), (x + w, y + hh), (220, 0, 0), tebal)
        cv2.putText(vis_bersih, ch, (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 0, 0), tebal + 1)

    return {
        "nomor_blok": nomor_blok,
        "nomor_tph": nomor_tph,
        "blok_preds": blok_preds,
        "tph_preds": tph_preds,
        "sep_found": sep_found,
        "vis": vis,
        "vis_bersih": vis_bersih,
        "bersih": bersih,
        "binary": binary,
    }


# ============================================================
# MODE "ALA GOOGLE LENS" — deteksi area teks di mana pun dalam foto,
# sorot semi-transparan, pengguna memilih area mana dibaca sebagai apa
# ============================================================
def deteksi_area_teks(binary, min_h_ratio=0.03):
    """Temukan kelompok teks (kata/baris) di seluruh foto tanpa
    bergantung pada garis pemisah. Return daftar box (x, y, w, h)."""
    H, W = binary.shape
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    char_mask = np.zeros_like(binary)
    widths = []
    for i in range(1, n_lbl):
        x, y, w, hh, area = stats[i]
        if hh < min_h_ratio * H or hh > 0.6 * H:
            continue
        density = area / max(w * hh, 1)
        if density < 0.06 or density > 0.95:
            continue
        if w > 0.9 * W:            # garis horizontal panjang, bukan karakter
            continue
        char_mask[lbl == i] = 255
        widths.append(w)
    if not widths:
        return []
    # Gabungkan karakter berdekatan jadi satu "kata" (dilasi horizontal)
    kw = max(9, int(np.median(widths) * 1.2)) | 1
    word_mask = cv2.dilate(char_mask,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3)))
    cnts, _ = cv2.findContours(word_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, hh = cv2.boundingRect(c)
        if hh < min_h_ratio * H or w * hh < 0.0015 * H * W:
            continue
        # Buang box raksasa (tekstur kulit batang yang menyatu), bukan teks
        if w * hh > 0.30 * H * W or hh > 0.5 * H or w > 0.95 * W:
            continue
        if not (0.3 <= w / max(hh, 1) <= 8):     # aspek tak wajar utk 1-4 karakter
            continue
        # Harus berisi 1-6 komponen setinggi karakter (bukan gerombolan noise)
        sub = char_mask[y:y + hh, x:x + w]
        n_sub, _, stats_sub, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        n_char = sum(1 for k in range(1, n_sub) if stats_sub[k][3] > 0.45 * hh)
        if not (1 <= n_char <= 6):
            continue
        boxes.append((x, y, w, hh))
    boxes.sort(key=lambda b: (b[1], b[0]))   # urut atas->bawah, kiri->kanan
    return boxes


def baca_area(binary, box, pola="bebas"):
    """Baca satu area teks. pola: 'blok' (huruf lalu angka),
    'tph' (angka semua), 'bebas' (tanpa constraint)."""
    x, y, w, hh = box
    crop = binary[y:y + hh, x:x + w]
    cnts, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cboxes = [list(cv2.boundingRect(c)) for c in cnts]
    cboxes = [b for b in cboxes if b[3] > 0.45 * hh]
    if not cboxes:
        return "", []
    # Gabung fragmen huruf pecah (versi ringkas)
    merged = True
    while merged:
        merged = False
        for i in range(len(cboxes)):
            for j in range(i + 1, len(cboxes)):
                a, b = cboxes[i], cboxes[j]
                if x_overlap_ratio(a, b) > 0.4 and y_gap(a, b) < 0.4 * max(a[3], b[3]):
                    x0 = min(a[0], b[0]); y0 = min(a[1], b[1])
                    x1 = max(a[0] + a[2], b[0] + b[2])
                    y1 = max(a[1] + a[3], b[1] + b[3])
                    cboxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    cboxes.pop(j)
                    merged = True
                    break
            if merged:
                break
    # Pisah karakter menempel, urut kiri->kanan
    final = []
    for (cx, cy, cw, ch) in cboxes:
        for (sx, sy, sw, sh) in split_touching(crop[cy:cy + ch, cx:cx + cw]):
            final.append((cx + sx, cy + sy, sw, sh))
    final.sort(key=lambda b: b[0])
    preds = []
    for i, (cx, cy, cw, ch) in enumerate(final):
        if pola == "tph":
            allowed = DIGIT_IDX
        elif pola == "blok":
            allowed = LETTER_IDX if i == 0 else DIGIT_IDX
        else:
            allowed = None
        preds.append(predict_char(to_model_input(crop[cy:cy + ch, cx:cx + cw]),
                                  allowed_idx=allowed))
    return "".join(p[0] for p in preds), preds


def sorot_area(rgb, boxes, teks_per_area=None):
    """Gambar sorotan semi-transparan gaya Google Lens di tiap area."""
    vis = rgb.copy()
    overlay = vis.copy()
    for (x, y, w, hh) in boxes:
        pad = max(3, hh // 10)
        cv2.rectangle(overlay, (x - pad, y - pad), (x + w + pad, y + hh + pad),
                      (255, 255, 255), -1)
    vis = cv2.addWeighted(overlay, 0.40, vis, 0.60, 0)
    tebal = max(2, rgb.shape[1] // 300)
    for idx, (x, y, w, hh) in enumerate(boxes):
        pad = max(3, hh // 10)
        cv2.rectangle(vis, (x - pad, y - pad), (x + w + pad, y + hh + pad),
                      (59, 130, 246), tebal)
        label = f"{idx + 1}"
        if teks_per_area and teks_per_area[idx]:
            label += f": {teks_per_area[idx]}"
        cv2.putText(vis, label, (x, max(y - pad - 6, 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.7, rgb.shape[1] / 700),
                    (59, 130, 246), tebal)
    return vis


# ============================================================
# TTS — suara Bahasa Indonesia
# ============================================================
def eja(teks):
    """Eja per karakter supaya jelas didengar: 'P67' -> 'P, 6, 7'."""
    return ", ".join(teks)


@st.cache_data(show_spinner=False)
def buat_audio(kalimat: str) -> bytes:
    tts = gTTS(text=kalimat, lang="id", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()


# Kalimat lanjutan yang disambungkan di akhir setiap pembacaan suara, supaya
# alur kerja lapangan lanjut otomatis: baca patok/QR -> lalu scan buah sawit.
KALIMAT_LANJUT_BUAH = ("Silakan lanjutkan, arahkan kamera ke buah sawit "
                       "untuk pemindaian berikutnya.")


def tombol_lanjut_scan_buah():
    """Bagian UI setelah hasil pembacaan: ajakan lanjut ke tahap scan buah sawit."""
    st.divider()
    st.subheader("➡️ Lanjut ke Pemindaian Buah Sawit")
    if URL_APP_SCAN_BUAH:
        st.link_button("🍇 Buka Aplikasi Scan Buah Sawit", URL_APP_SCAN_BUAH,
                       use_container_width=True)
    else:
        st.info("Tombol ini belum terhubung ke aplikasi scan buah sawit. Isi "
                "variabel `URL_APP_SCAN_BUAH` di bagian atas kode dengan alamat "
                "aplikasi Ripeness Detector Anda yang sudah online, atau kirim "
                "kode aplikasi tersebut supaya digabungkan langsung ke sini.")


# ============================================================
# MENU BARU — Scan QR Code (TERPISAH dari kamera model OCR di atas;
# tidak menyentuh CharCNN sama sekali, hanya decode QR + suara)
# ============================================================
def render_menu_scan_qr():
    st.subheader("📱 Scan QR Code — Barcode TPH")
    st.caption("Arahkan kamera ke stiker QR Code hasil cetak barcode TPH, "
               "atau upload fotonya.")

    suara_aktif_qr = st.toggle("🔊 Bacakan hasil lewat suara", value=True, key="suara_qr")

    tab_kamera_qr, tab_upload_qr = st.tabs(["📷 Kamera", "🖼️ Upload"])
    img_file_qr = None
    with tab_kamera_qr:
        foto_qr = st.camera_input("Arahkan ke QR Code, lalu ambil foto",
                                  label_visibility="collapsed", key="kamera_qr")
        if foto_qr is not None:
            img_file_qr = foto_qr
    with tab_upload_qr:
        up_qr = st.file_uploader("Pilih foto QR Code", type=["png", "jpg", "jpeg", "bmp"],
                                 label_visibility="collapsed", key="upload_qr")
        if up_qr is not None:
            img_file_qr = up_qr

    if img_file_qr is None:
        st.info("📷 Ambil foto QR Code atau upload gambar untuk memulai.")
        return

    pil_img_qr = Image.open(img_file_qr).convert("RGB")
    bgr_qr = cv2.cvtColor(np.array(pil_img_qr), cv2.COLOR_RGB2BGR)

    with st.spinner("Membaca QR Code..."):
        teks_qr = baca_qr(bgr_qr)
        data_qr = urai_payload_qr(teks_qr)

    if data_qr:
        st.success("📡 QR Code terbaca!")
        st.markdown(f"""
        <div class="hasil-card hasil-blok">
            <div class="hasil-label">Nomor Blok</div>
            <div class="hasil-nilai">{data_qr['blok']}</div>
        </div>
        <div class="hasil-card hasil-tph">
            <div class="hasil-label">Nomor TPH</div>
            <div class="hasil-nilai">{data_qr['tph']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Afdeling (AFD): **{data_qr['afdeling']}**")

        if suara_aktif_qr:
            kalimat = (
                f"Terdeteksi lewat kode Q R. Afdeling {eja(data_qr['afdeling'])}. "
                f"Nomor Blok, {eja(data_qr['blok'])}. "
                f"Nomor T P H, {eja(data_qr['tph'])}. " + KALIMAT_LANJUT_BUAH
            )
            try:
                st.audio(buat_audio(kalimat), format="audio/mp3", autoplay=True)
            except Exception:
                st.caption("🔇 Suara gagal dibuat (cek koneksi internet).")

        tombol_lanjut_scan_buah()
    elif teks_qr:
        st.warning(f"QR Code terbaca, tapi formatnya tidak dikenali: `{teks_qr}`. "
                   "Pastikan ini QR Code hasil generator barcode TPH.")
    else:
        st.error("Tidak ada QR Code terdeteksi pada foto. Pastikan QR Code terlihat "
                 "jelas, tidak blur, dan cukup dekat/terang.")


# ============================================================
# UI — mobile-first
# ============================================================
st.title("🌴 Pembaca Patok / QR Code — Blok / TPH")

menu_utama = st.radio(
    "Pilih Menu",
    ["🌴 Baca Patok (OCR Karakter)", "📱 Scan QR Code"],
    horizontal=True,
    label_visibility="collapsed",
)
st.divider()

if menu_utama == "📱 Scan QR Code":
    render_menu_scan_qr()
    st.stop()

# ------------------------------------------------------------
# Kalau sampai di sini berarti menu_utama == "Baca Patok (OCR Karakter)"
# — SELURUH kode di bawah ini TIDAK BERUBAH dari versi sebelumnya.
# ------------------------------------------------------------

# Pengaturan disembunyikan dalam expander (hemat layar HP)
with st.expander("⚙️ Pengaturan"):
    mode_lens = st.toggle("🔍 Mode pilih area (ala Google Lens)", value=False,
                          help="Deteksi semua area teks di foto, lalu pilih sendiri "
                               "area mana yang dibaca sebagai Blok/TPH. "
                               "Tidak bergantung pada garis pemisah/framing.")
    suara_aktif = st.toggle("🔊 Bacakan hasil lewat suara", value=True)
    hapus_bg = st.toggle("🧹 Hapus background sebelum diproses model", value=True)
    fokus_otomatis = st.toggle("🎯 Fokus otomatis ke area patok", value=True)
    hilangkan_noise = st.toggle("✨ Hilangkan noise/kotoran non-karakter", value=True)
    tampil_biner = st.toggle("Tampilkan gambar biner (debug)", value=False)
    block_size = st.slider("Block size threshold (ganjil)", 21, 81, 41, step=2)
    c_thresh = st.slider("Konstanta C threshold", 5, 35, 15)
    min_h = st.slider("Tinggi min. karakter (% gambar)", 2, 15, 5) / 100.0

# Tab: kamera dulu (use case utama di lapangan), upload kedua
tab_kamera, tab_upload = st.tabs(["📷 Kamera", "🖼️ Upload"])

img_file = None
dari_kamera = False
with tab_kamera:
    st.caption("Posisikan patok di dalam kotak, sejajarkan garis patok dengan garis biru.")
    foto = st.camera_input("Arahkan ke patok, lalu ambil foto",
                           label_visibility="collapsed")
    if foto is not None:
        img_file = foto
        dari_kamera = True
with tab_upload:
    up = st.file_uploader("Pilih foto patok", type=["png", "jpg", "jpeg", "bmp"],
                          label_visibility="collapsed")
    if up is not None:
        img_file = up
        dari_kamera = False

if img_file is None:
    st.info("📷 Ambil foto patok atau upload gambar untuk memulai.")
    st.stop()

# Decode gambar
pil_img = Image.open(img_file).convert("RGB")
bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Foto dari kamera: crop tengah 4:5 (samakan dengan preview),
# lalu crop lagi ke KOTAK PANDUAN putus-putus — hanya area itu yang diproses.
# Di MODE LENS: kotak panduan dilewati (seluruh foto dipindai).
if dari_kamera:
    Hf, Wf = bgr.shape[:2]
    target = 4 / 5  # lebar : tinggi
    if Wf / Hf > target:      # terlalu lebar -> pangkas kiri-kanan
        new_w = int(Hf * target)
        x0 = (Wf - new_w) // 2
        bgr = bgr[:, x0:x0 + new_w]
    else:                      # terlalu tinggi -> pangkas atas-bawah
        new_h = int(Wf / target)
        y0 = (Hf - new_h) // 2
        bgr = bgr[y0:y0 + new_h, :]

    if not mode_lens:
        # Kotak panduan (harus sama dengan CSS ::after): top 8%, kiri/kanan 12%, bottom 22%
        Hf, Wf = bgr.shape[:2]
        bgr = bgr[int(0.08 * Hf):int((1 - 0.22) * Hf),
                  int(0.12 * Wf):int((1 - 0.12) * Wf)]

# ============================================================
# MODE LENS: deteksi semua area teks -> sorot -> pengguna memilih
# ============================================================
if mode_lens:
    with st.spinner("Mencari area teks..."):
        rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if hapus_bg:
            k = max(31, (min(gray_full.shape) // 10) | 1)
            bgf = cv2.medianBlur(gray_full, min(k, 99))
            gray_full = cv2.divide(gray_full, bgf, scale=255)
            gray_full = cv2.normalize(gray_full, None, 0, 255,
                                      cv2.NORM_MINMAX).astype(np.uint8)
        binary_full = binarize_full(gray_full, block_size=block_size, c_thresh=c_thresh)
        area_boxes = deteksi_area_teks(binary_full, min_h_ratio=max(0.02, min_h * 0.6))

    if not area_boxes:
        st.error("Tidak ada area teks terdeteksi. Coba foto lebih dekat, "
                 "atau sesuaikan parameter di ⚙️ Pengaturan.")
        st.stop()

    # Baca bebas semua area dulu (untuk label sorotan, seperti Lens)
    teks_bebas = []
    for b in area_boxes:
        t, _ = baca_area(binary_full, b, pola="bebas")
        teks_bebas.append(t)

    st.image(sorot_area(rgb_full, area_boxes, teks_bebas),
             caption="Area teks terdeteksi — angka biru = nomor area",
             use_container_width=True)

    # Tebakan awal: area pertama yang mengandung huruf -> Blok;
    # area angka-semua pertama setelahnya -> TPH
    opsi = ["(tidak dipakai)"] + [f"Area {i+1}: «{t or '?'}»"
                                  for i, t in enumerate(teks_bebas)]
    guess_blok, guess_tph = 0, 0
    for i, t in enumerate(teks_bebas):
        if t and any(c.isalpha() for c in t) and guess_blok == 0:
            guess_blok = i + 1
        elif t and t.isdigit() and guess_tph == 0 and (guess_blok == 0 or i + 1 > guess_blok):
            guess_tph = i + 1
    if guess_blok == 0 and len(area_boxes) >= 1:
        guess_blok = 1
    if guess_tph == 0 and len(area_boxes) >= 2:
        guess_tph = 2

    kol1, kol2 = st.columns(2)
    with kol1:
        pilih_blok = st.selectbox("Baca sebagai NOMOR BLOK", opsi, index=guess_blok)
    with kol2:
        pilih_tph = st.selectbox("Baca sebagai NOMOR TPH", opsi, index=guess_tph)

    nomor_blok, blok_preds = "", []
    nomor_tph, tph_preds = "", []
    if pilih_blok != "(tidak dipakai)":
        idx = opsi.index(pilih_blok) - 1
        nomor_blok, blok_preds = baca_area(binary_full, area_boxes[idx], pola="blok")
    if pilih_tph != "(tidak dipakai)":
        idx = opsi.index(pilih_tph) - 1
        nomor_tph, tph_preds = baca_area(binary_full, area_boxes[idx], pola="tph")

    if nomor_blok or nomor_tph:
        st.markdown(f"""
        <div class="hasil-card hasil-blok">
            <div class="hasil-label">Nomor Blok</div>
            <div class="hasil-nilai">{nomor_blok or '—'}</div>
        </div>
        <div class="hasil-card hasil-tph">
            <div class="hasil-label">Nomor TPH</div>
            <div class="hasil-nilai">{nomor_tph or '—'}</div>
        </div>
        """, unsafe_allow_html=True)

        if suara_aktif:
            bagian = []
            if nomor_blok:
                bagian.append(f"Nomor Blok, {eja(nomor_blok)}")
            if nomor_tph:
                bagian.append(f"Nomor T P H, {eja(nomor_tph)}")
            try:
                kalimat = "Terdeteksi. " + ". ".join(bagian) + ". " + KALIMAT_LANJUT_BUAH
                audio_bytes = buat_audio(kalimat)
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            except Exception:
                st.caption("🔇 Suara gagal dibuat (cek koneksi internet).")

        tombol_lanjut_scan_buah()

        with st.expander("📊 Detail confidence per karakter"):
            for ch, cf, top3 in blok_preds:
                alt = ", ".join(f"{c} {p*100:.0f}%" for c, p in top3)
                st.write(f"Blok — **{ch}** : {cf*100:.0f}%  \n_alternatif: {alt}_")
            for ch, cf, top3 in tph_preds:
                alt = ", ".join(f"{c} {p*100:.0f}%" for c, p in top3)
                st.write(f"TPH — **{ch}** : {cf*100:.0f}%  \n_alternatif: {alt}_")
    else:
        st.info("Pilih area di atas untuk dibaca sebagai Blok / TPH.")

    if tampil_biner:
        st.image(binary_full, caption="Gambar biner (debug)",
                 use_container_width=True, clamp=True)
    st.stop()

# Proses (MODE OTOMATIS — garis pemisah)
with st.spinner("Memproses..."):
    hasil = baca_patok(bgr, block_size=block_size, c_thresh=c_thresh, min_h_ratio=min_h,
                       hapus_bg=hapus_bg, hilangkan_noise=hilangkan_noise,
                       auto_roi=fokus_otomatis)

terdeteksi = bool(hasil["nomor_blok"] or hasil["nomor_tph"])

# ============================================================
# HASIL — kartu besar di paling atas (yang paling penting dulu)
# ============================================================
if terdeteksi:
    st.markdown(f"""
    <div class="hasil-card hasil-blok">
        <div class="hasil-label">Nomor Blok</div>
        <div class="hasil-nilai">{hasil['nomor_blok'] or '—'}</div>
    </div>
    <div class="hasil-card hasil-tph">
        <div class="hasil-label">Nomor TPH</div>
        <div class="hasil-nilai">{hasil['nomor_tph'] or '—'}</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Suara otomatis saat teks terdeteksi ---
    if suara_aktif:
        bagian = []
        if hasil["nomor_blok"]:
            bagian.append(f"Nomor Blok, {eja(hasil['nomor_blok'])}")
        if hasil["nomor_tph"]:
            bagian.append(f"Nomor T P H, {eja(hasil['nomor_tph'])}")
        kalimat = "Terdeteksi. " + ". ".join(bagian) + ". " + KALIMAT_LANJUT_BUAH
        try:
            audio_bytes = buat_audio(kalimat)
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
        except Exception:
            st.caption("🔇 Suara gagal dibuat (cek koneksi internet).")

    tombol_lanjut_scan_buah()
else:
    st.error("Tidak ada karakter terdeteksi. Coba foto ulang lebih dekat, "
             "atau sesuaikan parameter di ⚙️ Pengaturan.")

if not hasil["sep_found"]:
    st.warning("Garis pemisah tidak terdeteksi — memakai tengah gambar sebagai batas.")

# Gambar hasil deteksi di bawah kartu
st.image(hasil["vis"], caption="Hasil deteksi", use_container_width=True)

with st.expander("🧹 Lihat gambar yang masuk ke model (bersih)"):
    st.image(hasil["vis_bersih"],
             caption="Setelah hapus background & hilangkan noise",
             use_container_width=True)
    ok_png, buf_png = cv2.imencode(".png", cv2.cvtColor(hasil["bersih"], cv2.COLOR_RGB2BGR))
    if ok_png:
        st.download_button("⬇️ Download gambar bersih (PNG)", data=buf_png.tobytes(),
                           file_name="patok_bersih.png", mime="image/png",
                           use_container_width=True)

if tampil_biner:
    st.image(hasil["binary"], caption="Gambar biner (debug)",
             use_container_width=True, clamp=True)

# Detail confidence dilipat, tidak memenuhi layar (termasuk top-3 alternatif)
if terdeteksi:
    with st.expander("📊 Detail confidence per karakter"):
        for ch, cf, top3 in hasil["blok_preds"]:
            alt = ", ".join(f"{c} {p*100:.0f}%" for c, p in top3)
            st.write(f"Blok — **{ch}** : {cf*100:.0f}%  \n_alternatif: {alt}_")
        for ch, cf, top3 in hasil["tph_preds"]:
            alt = ", ".join(f"{c} {p*100:.0f}%" for c, p in top3)
            st.write(f"TPH — **{ch}** : {cf*100:.0f}%  \n_alternatif: {alt}_")
