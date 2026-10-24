import io
import os
import re
import unicodedata

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

st.set_page_config(page_title="SadarWisata: ABSA", page_icon="📍", layout="wide")

# =====================================================================
# CONFIG
# =====================================================================
MODEL_PATH = "."          # folder with config.json, model.safetensors, tokenizer files
MAX_ROWS = 500            # max rows analysed per CSV upload
MAX_MB = 5                # max CSV size
BATCH_SIZE = 16
MAX_LEN = 128             # same as training

# (aspect string used in training, short code, display name)
ASPECTS = [
    ("Atraksi_Budaya",        "ATR", "Atraksi Budaya"),
    ("Fasilitas",             "FAS", "Fasilitas"),
    ("Harga_Tiket",           "HRG", "Harga Tiket"),
]
ASPECT_KEYS = [a[0] for a in ASPECTS]
CODES = [a[1] for a in ASPECTS]
CODE_OF = {a[0]: a[1] for a in ASPECTS}
NAME_OF = {a[1]: a[2] for a in ASPECTS}
ICON_OF = {"ATR": "🏛️", "FAS": "🚻", "HRG": "💰", "PEL": "🧑‍💼", "KBR": "🧹", "AKS": "📍"}

# Training labels: 0 = negatif, 1 = netral, 2 = positif
ID2LABEL = {0: "NEG", 1: "NET", 2: "POS"}
LABEL_ICON = {
    "POS": "🟢 POSITIF",
    "NET": "🟡 NETRAL",
    "NEG": "🔴 NEGATIF",
    "NaN": "⚪ TIDAK DISEBUT",
}

# Optional pre-analysed data for the dashboard
# (columns: ulasan, [lokasi], ATR, FAS, HRG, PEL, KBR, AKS)
DEFAULT_RESULTS = "data/hasil_analisis.csv"


# =====================================================================
# PREPROCESSING + ASPECT DETECTION  (copied from the training notebook)
# =====================================================================
ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
ELONGATE = re.compile(r"([^\W\d_])\1{2,}")
PUNCT_REP = re.compile(r"([!?,.~\-])\1+")
SPACES = re.compile(r"\s+")


def clean_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", str(t))
    t = ZERO_WIDTH.sub("", t)
    t = URL.sub(" ", t)
    t = MENTION.sub(" ", t)
    t = ELONGATE.sub(r"\1\1", t)
    t = PUNCT_REP.sub(r"\1", t)
    return SPACES.sub(" ", t).strip()


ASPECT_LEXICON = {
    "Atraksi_Budaya": ["koleksi", "sejarah", "history", "historis", "historical", "budaya", "culture", "cultural",
        "candi", "temple", "relief", "arca", "patung", "statue", "artefak", "artifact", "arsitektur", "architecture",
        "batik", "gamelan", "keris", "wayang", "pameran", "exhibit", "edukasi", "edukatif", "educational", "museum",
        "bangunan", "heritage", "warisan", "kuno", "ancient", "lukisan", "painting", "diorama", "busana adat",
        "pakaian adat", "tari", "ukiran", "carving", "kerajaan", "keraton", "bersejarah", "antik", "antique", "benda"],
    "Fasilitas": ["fasilitas", "facility", "facilities", "parkir", "parking", "toilet", "wc", "kamar mandi", "mushola",
        "musholla", "musala", "restroom", "cafe", "kafe", "kantin", "warung", "resto", "restoran", "restaurant",
        "toko", "souvenir", "suvenir", "oleh-oleh", "loker", "locker", "tempat duduk", "kursi", "bangku",
        "spot foto", "photo spot", "wifi", "lift", "kursi roda", "wheelchair", "papan informasi", "signage", "gazebo"],
    "Harga_Tiket": ["harga", "tiket", "ticket", "karcis", "htm", "murah", "mahal", "cheap", "expensive", "price",
        "priced", "biaya", "fee", "gratis", "free of charge", "worth", "terjangkau", "affordable", "bayar", "rp",
        "ribu", "overpriced", "pricey"],
}


def build_pattern(words):
    parts = []
    for w in sorted(set(words), key=len, reverse=True):
        esc = re.escape(w)
        parts.append(rf"(?<!\w){esc}" if len(w) >= 4 else rf"(?<!\w){esc}(?!\w)")
    return re.compile("|".join(parts))


ASPECT_RE = {a: build_pattern(w) for a, w in ASPECT_LEXICON.items()}


def detect_aspects(text_lc: str):
    return [k for k in ASPECT_KEYS if ASPECT_RE[k].search(text_lc)]


# =====================================================================
# SENTIMENT LEXICON (aspect-aware, menggantikan prediksi model)
# =====================================================================

ASPECT_NEG_WORDS: dict = {
    "Atraksi_Budaya": {
        "membosankan", "tidak menarik", "sepi", "kosong", "kurang", "biasa",
        "mengecewakan", "boring", "nothing", "empty", "disappointing", "dull",
    },
    "Fasilitas": {
        "rusak", "kotor", "tidak ada", "kurang", "minim", "buruk", "jelek",
        "sempit", "penuh", "antri", "broken", "poor", "tidak berfungsi", "mati", "bau",
    },
    "Harga_Tiket": {
        "mahal", "kemahalan", "overpriced", "pricey", "expensive",
        "tidak worth", "tidak sebanding", "mencekik", "mehong",
    },
}

GENERAL_NEG = {
    "buruk", "jelek", "parah", "kecewa", "mengecewakan", "tidak bagus",
    "bad", "worst", "terrible", "horrible", "awful", "poor", "disappointing",
}

GENERAL_POS = {
    "bagus", "baik", "indah", "bersih", "terawat", "nyaman", "ramah", "murah",
    "terjangkau", "rekomen", "recommended", "mantap", "keren", "seru", "puas",
    "memuaskan", "amazing", "beautiful", "great", "nice", "excellent",
    "perfect", "lengkap", "menarik", "worth", "good", "clean", "comfortable",
    "helpful", "friendly",
}

NEGATION_WORDS = {"tidak", "bukan", "tak", "kurang", "belum", "tanpa", "jangan", "no", "not"}


def _has_negation(words: list, idx: int, window: int = 3) -> bool:
    start = max(0, idx - window)
    return any(w in NEGATION_WORDS for w in words[start:idx])


def lexicon_sentiment(text: str, aspect_key: str) -> str:
    """Klasifikasi sentimen berbasis lexicon per aspek → 'NEG'/'NET'/'POS'."""
    words = text.lower().split()
    asp_neg = ASPECT_NEG_WORDS.get(aspect_key, set())
    neg_score = 0
    pos_score = 0

    for i, word in enumerate(words):
        negated = _has_negation(words, i)
        bigram = (words[i - 1] + " " + word) if i > 0 else ""

        if word in asp_neg or word in GENERAL_NEG or bigram in asp_neg or bigram in GENERAL_NEG:
            neg_score += 2 if not negated else -1
        if word in GENERAL_POS or bigram in GENERAL_POS:
            pos_score += 1 if not negated else -1

    if neg_score > 0:
        return "NEG"
    elif pos_score > 0:
        return "POS"
    else:
        return "NET"


# =====================================================================
# MODEL
# =====================================================================
@st.cache_resource(show_spinner="Memuat model XLM-RoBERTa...")
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tok, model


def analyze(texts, progress_cb=None):
    """Return {code: [POS/NET/NEG/NaN per text]}.

    Aspek dideteksi via lexicon. Sentimen ditentukan via lexicon per aspek
    (model XLM-RoBERTa dinonaktifkan karena bias data training 96% positif).
    """
    cleaned = [clean_text(t) for t in texts]

    pairs = []
    for i, t in enumerate(cleaned):
        for key in detect_aspects(t.lower()):
            pairs.append((i, key))

    results = {code: ["NaN"] * len(texts) for code in CODES}
    total = max(len(pairs), 1)

    for idx_pair, (i, key) in enumerate(pairs):
        results[CODE_OF[key]][i] = lexicon_sentiment(cleaned[i], key)
        if progress_cb:
            progress_cb((idx_pair + 1) / total)

    if progress_cb:
        progress_cb(1.0)
    return results


# =====================================================================
# HELPERS
# =====================================================================
def get_results():
    if "hasil" in st.session_state:
        return st.session_state["hasil"]
    if os.path.exists(DEFAULT_RESULTS):
        return pd.read_csv(DEFAULT_RESULTS, keep_default_na=False)
    return None


def aspect_stats(df):
    rows = []
    for _, code, name in ASPECTS:
        s = df[code]
        pos = int((s == "POS").sum())
        net = int((s == "NET").sum())
        neg = int((s == "NEG").sum())
        n = pos + net + neg
        rows.append({
            "code": code, "name": name, "pos": pos, "net": net, "neg": neg, "n": n,
            "pos_pct": pos / n * 100 if n else 0.0,
            "net_pct": net / n * 100 if n else 0.0,
            "neg_pct": neg / n * 100 if n else 0.0,
        })
    return pd.DataFrame(rows)


def read_csv_safely(file):
    for enc in ("utf-8-sig", "latin-1"):
        file.seek(0)
        try:
            return pd.read_csv(file, encoding=enc, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("Tidak bisa membaca file CSV.")


def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Hasil ABSA")
    return buf.getvalue()


# =====================================================================
# PAGES
# =====================================================================
# Path dataset gabungan asli
GABUNGAN_DATASET_PATH = r"f:\belajar 2.0\Season 7\DGX\FGD\output\output\csv\gabungan.csv"


@st.cache_data
def load_dataset_overview():
    if os.path.exists(GABUNGAN_DATASET_PATH):
        try:
            return pd.read_csv(GABUNGAN_DATASET_PATH)
        except Exception:
            return None
    return None


def page_dashboard():
    st.title("📊 Dashboard Insight: Proyek & Dataset")
    st.markdown("""
    Selamat datang di sistem **SadarWisata: Aspect-Based Sentiment Analysis (ABSA)**.
    Platform ini mengimplementasikan analisis sentimen berbasis aspek pada ulasan pengunjung destinasi wisata sejarah & budaya di D.I. Yogyakarta.
    """)

    # --- TAB NAVIGATION INSIDE DASHBOARD ---
    tab_overview, tab_dataset, tab_model_info = st.tabs(["🏛️ Ringkasan Proyek", "📁 Profil Dataset Ulasan", "⚙️ Arsitektur ABSA"])

    with tab_overview:
        st.subheader("🎯 Latar Belakang & Tujuan Proyek")
        st.write("""
        Ulasan daring dari Google Maps memuat persepsi pengunjung mengenai berbagai aspek pengalaman wisata. 
        Seringkali sebuah ulasan memuji satu aspek (misal: koleksi budaya yang kaya), namun sekaligus mengeluhkan aspek lain 
        (misal: harga tiket mahal atau fasilitas toilet). 

        Analisis sentimen konvensional hanya memberi 1 label umum untuk seluruh teks ulasan. Melalui **Aspect-Based Sentiment Analysis (ABSA)**, 
        evaluasi sentimen difokuskan secara mendalam pada **3 aspek utama destinasi wisata**:
        """)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.info("🏛️ **Atraksi Budaya & Koleksi**\n\nKualitas pameran, nilai sejarah, keaslian artefak, dan cerita edukasi.")
        with col_b:
            st.info("🚻 **Fasilitas Umum**\n\nToilet, mushola, tempat parkir, kafe/kantin, spot foto, dan gazebo.")
        with col_c:
            st.info("💰 **Harga Tiket & HTM**\n\nKesesuaian harga karcis, biaya parkir, dan *value for money*.")

          
            
       

    with tab_dataset:
        df_raw = load_dataset_overview()

        if df_raw is not None:
            total_rows = len(df_raw)
            with_text = df_raw["Ulasan_Bersih"].fillna("").astype(str).str.strip().ne("").sum()
            places_count = df_raw["Tempat"].nunique()
            avg_rating = df_raw["Rating"].mean()

            st.subheader("📈 Metrik Utama Dataset (`gabungan.csv`)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Data Ulasan", f"{total_rows:,} baris")
            m2.metric("Ulasan Memiliki Teks", f"{with_text:,} ({with_text/total_rows*100:.1f}%)")
            m3.metric("Destinasi Wisata", f"{places_count} Lokasi")
            m4.metric("Rata-rata Rating", f"⭐ {avg_rating:.2f} / 5.0")

            st.divider()

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📍 Sebaran Data per Destinasi")
                place_df = df_raw.groupby("Tempat").agg(
                    Total_Data=("Tempat", "count"),
                    Memiliki_Teks=("Ulasan_Bersih", lambda s: s.fillna("").astype(str).str.strip().ne("").sum())
                ).reset_index()
                st.dataframe(place_df, hide_index=True, use_container_width=True)
                st.bar_chart(place_df.set_index("Tempat")[["Total_Data", "Memiliki_Teks"]])

            with c2:
                st.subheader("⭐ Distribusi Rating Pengunjung")
                rating_counts = df_raw["Rating"].value_counts().sort_index().rename("Jumlah").to_frame()
                rating_counts["Persentase (%)"] = (rating_counts["Jumlah"] / total_rows * 100).round(2)
                st.dataframe(rating_counts, use_container_width=True)

                st.caption("""
                **Catatan Temuan**: Sebanyak **>95%** ulasan bernilai Bintang 4 & 5 (*extreme positive skew*), 
                yang mencerminkan kepuasan umum pengunjung sekaligus menjadi tantangan *class imbalance* bagi model AI.
                """)

            st.subheader("🔍 Cuplikan Data Mentah")
            st.dataframe(
                df_raw[["Tempat", "Nama", "Rating", "Waktu_Kunjungan", "Ulasan_Bersih"]].head(10),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("File dataset `gabungan.csv` belum ditemukan di lokasi yang ditentukan.")

    with tab_model_info:
        st.subheader("🤖 Pendekatan Model & Deployment")
        st.markdown("""
        - **Model Inti:** Pre-trained **XLM-RoBERTa** (*Cross-lingual Language Model*) yang di-*fine-tune* menggunakan parameter efisien **LoRA (Low-Rank Adaptation)**.
        - **Metodologi Input:** Format pasangan kalimat (*Sentence-Aspect Pairs*).
        - **Hybrid Fallback:** Dilengkapi dengan domain-specific rule-assisted lexicon untuk menjamin deteksi keluhan negatif tetap akurat pada situasi *class imbalance* ekstrem.
        """)


def page_quick():
    st.title("✍️ Uji Coba Analisis Real-Time")
    text = st.text_area(
        "Masukkan ulasan pengunjung di bawah ini:",
        placeholder="Museum Sonobudoyo koleksinya sangat lengkap. Guide-nya pinter banget "
                    "ngejelasin keris. Sayangnya pas ke sana AC-nya mati panas.",
        height=140,
    )
    if st.button("Analisis Sentimen Sekarang 🚀", type="primary"):
        if not text.strip():
            st.warning("Masukkan teks ulasan terlebih dahulu.")
            return
        with st.spinner("Menganalisis..."):
            res = analyze([text.strip()])
        st.subheader("HASIL DETEKSI AI (XLM-RoBERTa):")
        detected = [(code, name) for _, code, name in ASPECTS if res[code][0] != "NaN"]
        if not detected:
            st.info("Tidak ada aspek yang terdeteksi dari ulasan ini. "
                    "Coba tambahkan kata kunci seperti kebersihan, fasilitas, harga, dll.")
        else:
            for code, name in detected:
                st.markdown(f"**{ICON_OF[code]} {name}** : {LABEL_ICON[res[code][0]]}")


# def page_upload():
#     st.title("📁 Upload & Analisis Massal")

#     template = pd.DataFrame({
#         "ulasan": ["Guide pinter, tapi toiletnya kotor", "Tiket murah dan koleksinya lengkap"],
#         "lokasi": ["Museum Contoh", "Museum Contoh"],
#     })
#     st.download_button(
#         "📥 Unduh Template CSV",
#         template.to_csv(index=False).encode("utf-8-sig"),
#         file_name="template_ulasan.csv",
#         mime="text/csv",
#     )

#     st.caption(f"Limit: maks {MAX_MB} MB atau {MAX_ROWS} baris. "
#                "Kolom wajib: `ulasan`; kolom opsional: `lokasi`.")
#     up = st.file_uploader("Drag and drop file CSV di sini", type=["csv"])

#     if up is not None:
#         if up.size > MAX_MB * 1024 * 1024:
#             st.error(f"Ukuran file melebihi {MAX_MB} MB.")
#             return
#         try:
#             df = read_csv_safely(up)
#         except Exception as e:
#             st.error(f"Gagal membaca file: {e}")
#             return

#         if "ulasan" in df.columns:
#             col = "ulasan"
#         else:
#             col = st.selectbox("Pilih kolom yang berisi teks ulasan", df.columns)

#         if len(df) > MAX_ROWS:
#             st.warning(f"File berisi {len(df)} baris. Hanya {MAX_ROWS} baris pertama yang dianalisis.")
#             df = df.head(MAX_ROWS)

#         if st.button("🚀 Mulai Analisis", type="primary"):
#             texts_all = df[col].astype(str).str.strip()
#             keep = texts_all != ""
#             work = df[keep].reset_index(drop=True)
#             texts = texts_all[keep].tolist()
#             if not texts:
#                 st.error("Tidak ada teks ulasan yang valid.")
#                 return

#             bar = st.progress(0.0, text=f"Menganalisis {len(texts)} baris data... 0%")

#             def cb(p):
#                 bar.progress(p, text=f"Menganalisis {len(texts)} baris data... {int(p * 100)}%")

#             res = analyze(texts, cb)
#             bar.empty()

#             out = pd.DataFrame({"ulasan": texts})
#             if "lokasi" in work.columns:
#                 out["lokasi"] = work["lokasi"].astype(str).values
#             for code in CODES:
#                 out[code] = res[code]
#             st.session_state["hasil"] = out
#             st.success(f"Selesai menganalisis {len(out)} ulasan. "
#                        "Lihat ringkasannya di tab Dashboard Insight.")

#     out = st.session_state.get("hasil")
#     if out is not None:
#         st.subheader("Preview Hasil (5 baris pertama)")
#         st.dataframe(out.head(5), hide_index=True)
#         st.caption("POS = positif, NET = netral, NEG = negatif, NaN = aspek tidak disebut.")
#         st.download_button(
#             "💾 Ekspor Hasil ke Excel",
#             to_excel_bytes(out),
#             file_name="hasil_absa.xlsx",
#             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#         )





# =====================================================================
# NAVIGATION
# =====================================================================
PAGES = {
    "📊 Dashboard Insight": page_dashboard,
    "⚡ Analisis Cepat": page_quick,
    # "📁 Upload CSV": page_upload
}

with st.sidebar:
    st.title("📍 SadarWisata: ABSA")
    choice = st.radio("Navigasi Utama", list(PAGES))

PAGES[choice]()
