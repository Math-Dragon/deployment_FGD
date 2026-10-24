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
    ("Pelayanan_Staf",        "PEL", "Pelayanan Staf"),
    ("Kebersihan_Kenyamanan", "KBR", "Kebersihan & Kenyamanan"),
    ("Aksesibilitas_Lokasi",  "AKS", "Aksesibilitas & Lokasi"),
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
    "Pelayanan_Staf": ["pelayanan", "layanan", "service", "staf", "staff", "petugas", "pemandu", "guide", "tour guide",
        "ramah", "friendly", "sopan", "helpful", "karyawan", "penjaga", "satpam", "pegawai", "dijelaskan",
        "menjelaskan", "penjelasan", "explain", "explained", "welcome", "sambutan", "disambut", "pramuwisata"],
    "Kebersihan_Kenyamanan": ["bersih", "clean", "kotor", "dirty", "rapi", "rapih", "neat", "terawat", "terjaga",
        "maintained", "nyaman", "comfortable", "comfy", "adem", "sejuk", "teduh", "asri", "tenang", "peaceful",
        "quiet", "sepi", "ramai", "crowded", "panas", "hot", "bau", "smell", "aman", "suasana", "atmosphere",
        "ambience", "ambiance", "tertata", "terurus"],
    "Aksesibilitas_Lokasi": ["lokasi", "location", "akses", "access", "accessible", "jalan", "road", "strategis",
        "strategic", "dekat", "near", "jauh", "far", "gang", "sempit", "narrow", "petunjuk arah", "google maps",
        "gmaps", "maps", "antri", "antre", "antrean", "queue", "jam buka", "buka", "tutup", "closed", "reservasi",
        "booking", "transportasi", "grab", "gojek", "becak", "andong", "jarak", "pusat kota", "malioboro"],
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

    Same recipe as training: the lexicon decides which aspects a review
    mentions, and the model rates only those (review, aspect) pairs.
    """
    tok, model = load_model()
    cleaned = [clean_text(t) for t in texts]

    pairs = []
    for i, t in enumerate(cleaned):
        for key in detect_aspects(t.lower()):
            pairs.append((i, key))

    results = {code: ["NaN"] * len(texts) for code in CODES}

    for start in range(0, len(pairs), BATCH_SIZE):
        chunk = pairs[start:start + BATCH_SIZE]
        enc = tok(
            [cleaned[i] for i, _ in chunk],   # review FIRST ...
            [k for _, k in chunk],            # ... aspect SECOND (as in training)
            return_tensors="pt", padding=True,
            truncation="only_first", max_length=MAX_LEN,
        )
        with torch.inference_mode():
            ids = model(**enc).logits.argmax(-1).tolist()
        for (i, k), j in zip(chunk, ids):
            results[CODE_OF[k]][i] = ID2LABEL[j]
        if progress_cb:
            progress_cb(min((start + BATCH_SIZE) / len(pairs), 1.0))

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
def page_dashboard():
    st.title("📊 Dashboard Insight")
    df = get_results()
    if df is None or df.empty:
        st.info(
            "Belum ada data untuk ditampilkan. Analisis file di tab **Upload CSV**, "
            f"atau simpan hasil analisis sebelumnya di `{DEFAULT_RESULTS}`."
        )
        return
    if not all(c in df.columns for c in CODES + ["ulasan"]):
        st.error(f"Data harus memiliki kolom: ulasan, {', '.join(CODES)}.")
        return

    if "lokasi" in df.columns:
        options = ["Semua lokasi"] + sorted(df["lokasi"].astype(str).unique())
        pilih = st.selectbox("Filter Lokasi", options)
        if pilih != "Semua lokasi":
            df = df[df["lokasi"].astype(str) == pilih]

    stats = aspect_stats(df)
    valid = stats[stats["n"] > 0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Ulasan", f"{len(df):,}")
    if valid.empty:
        c2.metric("Aspek Terbaik", "-")
        c3.metric("Isu Utama", "-")
    else:
        best = valid.loc[valid["pos_pct"].idxmax()]
        c2.metric("Aspek Terbaik", f"{best['name']} ({best['pos_pct']:.0f}%)")
        worst = valid.loc[valid["neg_pct"].idxmax()]
        if worst["neg_pct"] > 0:
            c3.metric("Isu Utama", f"{worst['name']} ({worst['neg_pct']:.0f}%)")
        else:
            c3.metric("Isu Utama", "Tidak ada")

    st.subheader("Grafik Sentimen per Aspek")
    chart = stats.set_index("name")[["pos_pct", "net_pct", "neg_pct"]].rename(
        columns={"pos_pct": "Positif (%)", "net_pct": "Netral (%)", "neg_pct": "Negatif (%)"}
    )
    st.bar_chart(chart, color=["#2e9e5b", "#e0a526", "#d64545"])
    st.caption(
        "Persentase dihitung dari ulasan yang menyebut aspek tersebut. "
        + " | ".join(f"{r['code']}: {r['n']} ulasan" for _, r in stats.iterrows())
    )

    st.subheader("Tabel Keluhan Utama (Fokus Perbaikan)")
    neg_mask = (df[CODES] == "NEG").any(axis=1)
    keluhan = df[neg_mask].copy()
    if keluhan.empty:
        st.success("Tidak ada keluhan terdeteksi oleh model.")
        return
    keluhan["Aspek Dideteksi"] = keluhan.apply(
        lambda r: ", ".join(f"{NAME_OF[c]} (Negatif)" for c in CODES if r[c] == "NEG"), axis=1
    )
    st.dataframe(
        keluhan[["ulasan", "Aspek Dideteksi"]].rename(columns={"ulasan": "Teks Ulasan"}).head(100),
        hide_index=True,
    )


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
        for _, code, name in ASPECTS:
            st.markdown(f"**{ICON_OF[code]} {name}** : {LABEL_ICON[res[code][0]]}")


def page_upload():
    st.title("📁 Upload & Analisis Massal")

    template = pd.DataFrame({
        "ulasan": ["Guide pinter, tapi toiletnya kotor", "Tiket murah dan koleksinya lengkap"],
        "lokasi": ["Museum Contoh", "Museum Contoh"],
    })
    st.download_button(
        "📥 Unduh Template CSV",
        template.to_csv(index=False).encode("utf-8-sig"),
        file_name="template_ulasan.csv",
        mime="text/csv",
    )

    st.caption(f"Limit: maks {MAX_MB} MB atau {MAX_ROWS} baris. "
               "Kolom wajib: `ulasan`; kolom opsional: `lokasi`.")
    up = st.file_uploader("Drag and drop file CSV di sini", type=["csv"])

    if up is not None:
        if up.size > MAX_MB * 1024 * 1024:
            st.error(f"Ukuran file melebihi {MAX_MB} MB.")
            return
        try:
            df = read_csv_safely(up)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            return

        if "ulasan" in df.columns:
            col = "ulasan"
        else:
            col = st.selectbox("Pilih kolom yang berisi teks ulasan", df.columns)

        if len(df) > MAX_ROWS:
            st.warning(f"File berisi {len(df)} baris. Hanya {MAX_ROWS} baris pertama yang dianalisis.")
            df = df.head(MAX_ROWS)

        if st.button("🚀 Mulai Analisis", type="primary"):
            texts_all = df[col].astype(str).str.strip()
            keep = texts_all != ""
            work = df[keep].reset_index(drop=True)
            texts = texts_all[keep].tolist()
            if not texts:
                st.error("Tidak ada teks ulasan yang valid.")
                return

            bar = st.progress(0.0, text=f"Menganalisis {len(texts)} baris data... 0%")

            def cb(p):
                bar.progress(p, text=f"Menganalisis {len(texts)} baris data... {int(p * 100)}%")

            res = analyze(texts, cb)
            bar.empty()

            out = pd.DataFrame({"ulasan": texts})
            if "lokasi" in work.columns:
                out["lokasi"] = work["lokasi"].astype(str).values
            for code in CODES:
                out[code] = res[code]
            st.session_state["hasil"] = out
            st.success(f"Selesai menganalisis {len(out)} ulasan. "
                       "Lihat ringkasannya di tab Dashboard Insight.")

    out = st.session_state.get("hasil")
    if out is not None:
        st.subheader("Preview Hasil (5 baris pertama)")
        st.dataframe(out.head(5), hide_index=True)
        st.caption("POS = positif, NET = netral, NEG = negatif, NaN = aspek tidak disebut.")
        st.download_button(
            "💾 Ekspor Hasil ke Excel",
            to_excel_bytes(out),
            file_name="hasil_absa.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def page_info():
    st.title("ℹ️ Informasi")
    t1, t2, t3 = st.tabs(["Proyek", "Dataset", "Model"])
    with t1:
        st.info("Konten menyusul.")
    with t2:
        st.info("Konten menyusul.")
    with t3:
        st.info("Konten menyusul.")


# =====================================================================
# NAVIGATION
# =====================================================================
PAGES = {
    "📊 Dashboard Insight": page_dashboard,
    "⚡ Analisis Cepat": page_quick,
    "📁 Upload CSV": page_upload,
    "ℹ️ Informasi": page_info,
}

with st.sidebar:
    st.title("📍 SadarWisata: ABSA")
    choice = st.radio("Navigasi Utama", list(PAGES))

PAGES[choice]()
