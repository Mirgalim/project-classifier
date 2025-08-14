import io
import uuid
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import linear_kernel

STORAGE_DIR = Path(__file__).parent / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

class ClassificationError(Exception):
    pass


# ---------- Helpers ----------
def _read_excel_required(file_bytes: bytes, label: str) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    except Exception as e:
        raise ClassificationError(f"'{label}' файлыг уншиж чадсангүй: {e}")

def _normalize_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str, copy=False).str.strip().str.lower()

def _ensure_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    """Ensure required columns exist; create empty ones if missing."""
    for col in required:
        if col not in df.columns:
            df[col] = ""
    return df[required]

def _create_key_text(df: pd.DataFrame) -> pd.Series:
    # No duplicate fields; strict order
    return (
        df['Төрөл'].astype(str) + " " +
        df['Ерөнхий ангилал'].astype(str) + " " +
        df['Ангилал'].astype(str) + " " +
        df['Тайлбар'].astype(str) + " " +
        df['Бренд'].astype(str)
    )

def _save_xlsx(df: pd.DataFrame, path: Path) -> None:
    # Use only openpyxl to avoid env issues with xlsxwriter
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')


# ---------- Core ----------
def run_classification(
    sales_bytes: bytes,
    category_bytes: bytes,
    manual_bytes: Optional[bytes] = None,
    probability_threshold: float = 0.15,
    batch_size: int = 2000,     # kept for signature compatibility; not used
    max_features: int = 20000,
) -> Tuple[str, pd.DataFrame]:

    # 1) Sales
    sales_df = _read_excel_required(sales_bytes, "sales.xlsx")
    if 'Барааны нэр' not in sales_df.columns:
        raise ClassificationError("sales.xlsx дотор 'Барааны нэр' багана байх ёстой!")
    sales_df['Барааны нэр'] = _normalize_text_series(sales_df['Барааны нэр'])
    unique_products = sales_df['Барааны нэр'].dropna().unique()

    if unique_products.size == 0:
        job_id = uuid.uuid4().hex[:12]
        out_path = STORAGE_DIR / f"angilsan_{job_id}.xlsx"
        _save_xlsx(sales_df, out_path)
        return job_id, sales_df

    # 2) Category sheets (robust)
    try:
        cat_xls = pd.ExcelFile(io.BytesIO(category_bytes), engine="openpyxl")
    except Exception as e:
        raise ClassificationError(f"Ангиллын Excel-ийг нээж чадсангүй: {e}")

    needed_cols = ['Ерөнхий ангилал', 'Төрөл', 'Ангилал', 'Тайлбар', 'Бренд', 'Сегмент']
    base_cols = ['Ерөнхий ангилал', 'Төрөл', 'Ангилал', 'Тайлбар', 'Бренд']  # without 'Сегмент'

    category_dfs = []
    skipped = []

    for sheet_name in cat_xls.sheet_names:
        try:
            sheet_df = cat_xls.parse(sheet_name, dtype=str)
            # Ensure all required base cols exist (create empty if missing)
            sheet_df = _ensure_columns(sheet_df, base_cols).fillna('')
            # Normalize text cols
            for col in base_cols:
                sheet_df[col] = sheet_df[col].astype(str).str.strip().str.lower()
            sheet_df['Сегмент'] = sheet_name
            category_dfs.append(sheet_df)
        except Exception as e:
            skipped.append(f"{sheet_name}: {e}")
            continue

    if not category_dfs:
        msg = "Ангиллын Excel-д хүчинтэй sheet олдсонгүй."
        if skipped:
            msg += " Алгассан шитүүд: " + "; ".join(skipped[:5])
        raise ClassificationError(msg)

    category_df = pd.concat(category_dfs, ignore_index=True)
    del category_dfs, cat_xls
    gc.collect()

    # Make sure final df has all needed cols (safety)
    category_df = _ensure_columns(category_df, needed_cols)

    # 3) Key text (category only)
    category_df['түлхүүр_текст'] = _create_key_text(category_df)
    cat_texts = category_df['түлхүүр_текст'].values

    # 4) TF-IDF (fit on categories, transform products)
    if cat_texts.size == 0:
        raise ClassificationError("Ангиллын текст хоосон байна.")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        dtype=np.float32,
        lowercase=False,
        norm='l2'
    )

    cat_vecs = vectorizer.fit_transform(cat_texts)       # Nc x D
    prod_vecs = vectorizer.transform(unique_products)    # Np x D

    # 5) Top-1 match: try NN first, else sparse linear_kernel
    try:
        nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(cat_vecs)
        distances, indices = nn.kneighbors(prod_vecs, n_neighbors=1, return_distance=True)
        best_idx = indices.reshape(-1).astype(np.int32)
        best_sim = (1.0 - distances.reshape(-1)).astype(np.float32)
        del nn, distances, indices
    except Exception:
        # Fallback: blockwise linear_kernel on sparse matrices
        prod_vecs = prod_vecs.tocsr()
        cat_vecs = cat_vecs.tocsr()
        n_products = prod_vecs.shape[0]
        block = max(1000, min(20000, n_products))
        best_idx = np.zeros(n_products, dtype=np.int32)
        best_sim = np.zeros(n_products, dtype=np.float32)
        for i in range(0, n_products, block):
            j = min(i + block, n_products)
            sims = linear_kernel(prod_vecs[i:j], cat_vecs)  # (j-i, Nc)
            idx = sims.argmax(axis=1)
            sc = sims.max(axis=1)
            if sparse.issparse(sims):
                idx = np.asarray(idx).ravel()
                sc = np.asarray(sc).ravel()
            best_idx[i:j] = idx.astype(np.int32, copy=False)
            best_sim[i:j] = sc.astype(np.float32, copy=False)
            del sims
            gc.collect()

    del prod_vecs, cat_vecs
    gc.collect()

    # 6) Build classification
    picked = category_df.iloc[best_idx][['Ангилал', 'Төрөл', 'Ерөнхий ангилал', 'Сегмент']].reset_index(drop=True)
    classified = pd.DataFrame({'Барааны нэр': unique_products, 'Магадлал': best_sim}, copy=False)
    classified = pd.concat([classified, picked], axis=1)

    # 7) Manual overrides
    if manual_bytes is not None:
        manual_df = _read_excel_required(manual_bytes, "manual_fix.xlsx")
        if 'Барааны нэр' not in manual_df.columns:
            raise ClassificationError("manual_fix.xlsx дотор 'Барааны нэр' багана байх ёстой!")
        manual_df['Барааны нэр'] = _normalize_text_series(manual_df['Барааны нэр'])

        classified = classified.merge(manual_df, on='Барааны нэр', how='left', suffixes=('', '_гар'))

        low_conf = classified['Магадлал'] < float(probability_threshold)
        for col in ['Ангилал', 'Төрөл', 'Ерөнхий ангилал', 'Сегмент']:
            mcol = f"{col}_гар"
            if mcol in classified.columns:
                mask = low_conf & classified[mcol].notna()
                classified.loc[mask, col] = classified.loc[mask, mcol]

        # drop *_гар columns
        classified = classified.drop(columns=[c for c in classified.columns if c.endswith('_гар')], errors='ignore')

    # 8) Join back
    final_result = sales_df.merge(classified, on='Барааны нэр', how='left')

    # 9) Save (openpyxl only)
    job_id = uuid.uuid4().hex[:12]
    out_path = STORAGE_DIR / f"angilsan_{job_id}.xlsx"
    _save_xlsx(final_result, out_path)

    return job_id, final_result
