import io
import uuid
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ====== Storage ======
STORAGE_DIR = Path(__file__).parent / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# ====== Error ======
class ClassificationError(Exception):
    pass

REQ_COLS = ["Ерөнхий ангилал", "Төрөл", "Ангилал", "Тайлбар", "Бренд", "Сегмент"]

# ---------- Fast readers ----------
def _read_excel_bytes(file_bytes: bytes, label: str, usecols=None) -> pd.DataFrame:
    try:
        return pd.read_excel(
            io.BytesIO(file_bytes),
            engine="openpyxl",
            dtype=str,
            na_filter=False,
            usecols=usecols
        )
    except Exception as e:
        raise ClassificationError(f"'{label}' файлыг уншиж чадсангүй: {e}")

def _read_excel_all_sheets_bytes(file_bytes: bytes, label: str, usecols=None) -> dict[str, pd.DataFrame]:
    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
        return {name: xls.parse(name, dtype=str, na_filter=False, usecols=usecols) for name in xls.sheet_names}
    except Exception as e:
        raise ClassificationError(f"'{label}' Excel-ийг нээж чадсангүй: {e}")

def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)

def _non_empty(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().ne("")

# ======================================================================
# CORE: Энгийн TF-IDF + cosine (таны хүссэн логик)
# ======================================================================
def _classify_core(
    sales_bytes: bytes,
    category_bytes: bytes,
    manual_bytes: Optional[bytes],
    threshold: float,
) -> Tuple[str, pd.DataFrame]:
    # 1) Sales — зөвхөн хэрэгтэй багана
    sales_df = _read_excel_bytes(sales_bytes, "sales.xlsx", usecols=lambda c: c == "Барааны нэр")
    if "Барааны нэр" not in sales_df.columns:
        raise ClassificationError("sales.xlsx дотор 'Барааны нэр' багана байх ёстой!")
    sales_df["Барааны нэр"] = _norm(sales_df["Барааны нэр"])
    sales_df = sales_df[_non_empty(sales_df["Барааны нэр"])].copy()
    products = sales_df["Барааны нэр"].unique()

    # 2) Category — бүх sheet, зөвхөн REQ_COLS
    cat_sheets = _read_excel_all_sheets_bytes(category_bytes, "Nomin_ba3.xlsx", usecols=lambda c: c in REQ_COLS[:-1])
    category_df = pd.DataFrame()
    for sheet, df in cat_sheets.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df["Сегмент"] = sheet
        for c in REQ_COLS:
            if c not in df.columns:
                df[c] = ""
            df[c] = _norm(df[c])
        category_df = pd.concat([category_df, df[REQ_COLS]], ignore_index=True)

    if category_df.empty:
        raise ClassificationError("Ангиллын Excel-д хүчинтэй мөр олдсонгүй.")

    # 3) Түлхүүр текст
    category_df["түлхүүр_текст"] = (
        category_df["Төрөл"] + " " +
        category_df["Төрөл"] + " " +
        category_df["Ерөнхий ангилал"] + " " +
        category_df["Ангилал"] + " " +
        category_df["Тайлбар"] + " " +
        category_df["Бренд"]
    ).str.strip()

    # сул түлхүүрүүдийг цэвэрлэх (2+ үг)
    wc = category_df["түлхүүр_текст"].str.replace(r"[^\w\s]+", " ", regex=True).str.split().map(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    category_df = category_df[wc >= 2].reset_index(drop=True)

    # Хэрэв хоосон бол default
    if len(products) == 0 or len(category_df) == 0:
        final_result = sales_df.copy()
        for c in ["Ангилал", "Төрөл", "Ерөнхий ангилал"]:
            final_result[c] = "UNCLASSIFIED"
        final_result["Сегмент"] = "—"
        final_result["Магадлал"] = np.float32(0.0)
        return uuid.uuid4().hex[:12], final_result

    # 4) TF-IDF + cosine
    texts = list(products) + list(category_df["түлхүүр_текст"].values)
    vectorizer = TfidfVectorizer(lowercase=False)  # доороо lowercase хийчихсэн
    tfidf = vectorizer.fit_transform(texts)
    prod_vecs = tfidf[:len(products)]
    cat_vecs  = tfidf[len(products):]
    sim = cosine_similarity(prod_vecs, cat_vecs)

    # 5) Оноолт
    idx = sim.argmax(axis=1)
    sc  = sim.max(axis=1).astype(np.float32)
    classified = pd.DataFrame({
        "Барааны нэр": products,
        "Ангилал": category_df.iloc[idx]["Ангилал"].to_numpy(),
        "Төрөл": category_df.iloc[idx]["Төрөл"].to_numpy(),
        "Ерөнхий ангилал": category_df.iloc[idx]["Ерөнхий ангилал"].to_numpy(),
        "Сегмент": category_df.iloc[idx]["Сегмент"].to_numpy(),
        "Магадлал": sc,
    })

    # 6) Manual override (сонголт)
    if manual_bytes is not None:
        manual = _read_excel_bytes(manual_bytes, "manual_fix.xlsx")
        if "Барааны нэр" not in manual.columns:
            raise ClassificationError("manual_fix.xlsx дотор 'Барааны нэр' багана байх ёстой!")
        manual["Барааны нэр"] = _norm(manual["Барааны нэр"])
        classified = classified.merge(manual, on="Барааны нэр", how="left", suffixes=("", "_гар"))

        low = classified["Магадлал"] < threshold

        def override(col: str):
            g = f"{col}_гар"
            if g in classified.columns:
                mask = low & classified[g].astype(str).str.strip().ne("")
                classified.loc[mask, col] = classified.loc[mask, g]

        for c in ["Ангилал", "Төрөл", "Ерөнхий ангилал", "Сегмент"]:
            override(c)

        classified = classified.drop(columns=[c for c in classified.columns if c.endswith("_гар")], errors="ignore")

    # 7) Буцааж тараах + NaN-гүй
    final_result = sales_df.merge(classified, on="Барааны нэр", how="left").fillna("")
    return uuid.uuid4().hex[:12], final_result

# ======================================================================
# PUBLIC API
# ======================================================================
def run_classification(
    sales_bytes: bytes,
    category_bytes: bytes,
    manual_bytes: Optional[bytes] = None,
    probability_threshold: float = 0.15,
    batch_size: int = 2000,    # signature хадгалж үлдээв
    max_features: int = 10000, # signature хадгалж үлдээв
    engine: str = "script",
) -> Tuple[str, pd.DataFrame]:
    return _classify_core(
        sales_bytes=sales_bytes,
        category_bytes=category_bytes,
        manual_bytes=manual_bytes,
        threshold=probability_threshold,
    )

def save_excel_file(df: pd.DataFrame, job_id: str) -> Path:
    out_path = STORAGE_DIR / f"angilsan_{job_id}.xlsx"
    df = df.fillna("")
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:   # 👈 engine-г сольсон
        df.to_excel(w, index=False, sheet_name="Results")
    return out_path

