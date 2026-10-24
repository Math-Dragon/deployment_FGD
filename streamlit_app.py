import io
import os

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

st.set_page_config(page_title="SadarWisata: ABSA", page_icon="📍", layout="wide")

# =====================================================================
# CONFIG - adjust these to match how the model was trained
# =====================================================================
MODEL_PATH = "."          # folder with config.json, model.safetensors, tokenizer files
MAX_ROWS = 500            # max rows analysed per CSV upload
MAX_MB = 5                # max CSV size
BATCH_SIZE = 16
MAX_LEN = 128

# code -> display name
ASPECTS = {
    "EDU": "Edukasi & Informasi",
    "FAS": "Fasilitas & Bersih",
    "HRG": "Harga & Tiket",
}
# The aspect text paired with the review when calling the model.
# CHANGE THESE if you used different aspect strings when training.
ASPECT_QUERY = {"EDU": "edukasi", "FAS": "fasilitas", "HRG": "harga"}

# If your config.json has generic names (LABEL_0, LABEL_1, ...), map them here:
# e.g. {"LABEL_0": "NEG", "LABEL_1": "NaN", "LABEL_2": "POS"}
LABEL_OVERRIDE = {}

# Optional pre-analysed data for the dashboard (columns: ulasan, [lokasi], EDU, FAS, HRG)
DEFAULT_RESULTS = "data/hasil_analisis.csv"

LABEL_ICON = {"POS": "🟢 POSITIF", "NEG": "🔴 NEGATIF", "NaN": "⚪ TIDAK DISEBUT"}


# =====================================================================
# MODEL
# =====================================================================
@st.cache_resource(show_spinner="Memuat model XLM-RoBERTa...")
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tok, model


def normalize_label(raw):
    """Map whatever the model calls a class to POS / NEG / NaN."""
    if raw in LABEL_OVERRIDE:
        return LABEL_OVERRIDE[raw]
    s = str(raw).lower()
    if "pos" in s:
        return "POS"
    if "neg" in s:
        return "NEG"
    return "NaN"


def check_labels(model):
    mapped = {normalize_label(v) for v in model.config.id2label.values()}
    if "POS" not in mapped or "NEG" not in mapped:
        st.warning(
            "Label model tidak dikenali "
            f"({list(model.config.id2label.values())}). "
            "Isi LABEL_OVERRIDE di bagian CONFIG agar hasilnya benar."
        )


def analyze(texts, progress_cb=None):
    """Return {aspect_code: [label per text]}."""
    tok, model = load_model()
    results = {code: [] for code in ASPECTS}
    n_batches = -(-len(texts) // BATCH_SIZE)
    total = max(n_batches * len(ASPECTS), 1)
    done = 0
    for code in ASPECTS:
        query = ASPECT_QUERY[code]
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = tok(
                [query] * len(batch), batch,
                return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_LEN,
            )
            with torch.inference_mode():
                ids = model(**enc).logits.argmax(-1).tolist()
            results[code] += [normalize_label(model.config.id2label[j]) for j in ids]
            done += 1
            if progress_cb:
                progress_cb(done / total)
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
    for code, name in ASPECTS.items():
        pos = int((df[code] == "POS").sum())
        neg = int((df[code] == "NEG").sum())
        n = pos + neg
        rows.append({
            "code": code, "name": name, "pos": pos, "neg": neg, "n": n,
            "pos_pct": pos / n * 100 if n else 0.0,
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
        worst = valid.loc[valid["neg_pct"].idxmax()]
        c2.metric("Aspek Terbaik", f"{best['name']} ({best['pos_pct']:.0f}%)")
        c3.metric("Isu Utama", f"{worst['name']} ({worst['neg_pct']:.0f}%)")

    st.subheader("Grafik Sentimen per Aspek")
    chart = stats.set_index("name")[["pos_pct", "neg_pct"]].rename(
        columns={"pos_pct": "Positif (%)", "neg_pct": "Negatif (%)"}
    )
    st.bar_chart(chart, color=["#2e9e5b", "#d64545"])
    st.caption(
        "Persentase dihitung dari ulasan yang menyebut aspek tersebut. "
        + " | ".join(f"{r['code']}: {r['n']} ulasan" for _, r in stats.iterrows())
    )

    st.subheader("Tabel Keluhan Utama (Fokus Perbaikan)")
    neg_mask = (df[list(ASPECTS)] == "NEG").any(axis=1)
    keluhan = df[neg_mask].copy()
    if keluhan.empty:
        st.success("Tidak ada keluhan terdeteksi.")
        return
    keluhan["Aspek Dideteksi"] = keluhan.apply(
        lambda r: ", ".join(f"{c} (Negatif)" for c in ASPECTS if r[c] == "NEG"), axis=1
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
        _, model = load_model()
        check_labels(model)
        st.subheader("HASIL DETEKSI AI (XLM-RoBERTa):")
        icons = {"EDU": "🏛️", "FAS": "🚻", "HRG": "💰"}
        for code, name in ASPECTS.items():
            st.markdown(f"**{icons[code]} {name}** : {LABEL_ICON[res[code][0]]}")


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

    st.caption(f"Limit: maks {MAX_MB} MB atau {MAX_ROWS} baris. Kolom wajib: `ulasan`; kolom opsional: `lokasi`.")
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
            work = df[texts_all != ""].reset_index(drop=True)
            texts = texts_all[texts_all != ""].tolist()
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
            for code in ASPECTS:
                out[code] = res[code]
            st.session_state["hasil"] = out
            st.success(f"Selesai menganalisis {len(out)} ulasan. Lihat ringkasannya di tab Dashboard Insight.")

    out = st.session_state.get("hasil")
    if out is not None:
        st.subheader("Preview Hasil (5 baris pertama)")
        st.dataframe(out.head(5), hide_index=True)
        st.download_button(
            "💾 Ekspor Hasil ke Excel",
            to_excel_bytes(out),
            file_name="hasil_absa.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def page_info():
    st.title("ℹ️ Informasi")
    st.subheader("Tentang Proyek")
    st.info("Konten menyusul.")
    st.subheader("Dataset")
    st.info("Konten menyusul.")
    st.subheader("Model")
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
