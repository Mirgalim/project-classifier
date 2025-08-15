import io
import uuid
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import psutil

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import linear_kernel
from functools import partial


STORAGE_DIR = Path(__file__).parent / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

class ClassificationError(Exception):
    pass


def calculate_batch_size() -> int:
    available_memory = psutil.virtual_memory().available
    max_products_in_memory = (available_memory * 0.7) // (100 * 1024 * 1024) * 1000
    return max(500, min(int(max_products_in_memory), 5000))


def process_product_batch(products_batch: np.ndarray, 
                         vectorizer: TfidfVectorizer,
                         cat_vecs: sparse.csr_matrix,
                         category_df: pd.DataFrame) -> pd.DataFrame:
    
    batch_vecs = vectorizer.transform(products_batch)
    
    try:
        nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(cat_vecs)
        distances, indices = nn.kneighbors(batch_vecs, return_distance=True)
        best_idx = indices.reshape(-1).astype(np.int32)
        best_sim = (1.0 - distances.reshape(-1)).astype(np.float32)
        del nn, distances, indices
    except Exception:
        batch_vecs = batch_vecs.tocsr()
        sims = linear_kernel(batch_vecs, cat_vecs)
        best_idx = sims.argmax(axis=1).astype(np.int32)
        best_sim = sims.max(axis=1).astype(np.float32)
        if sparse.issparse(best_sim):
            best_idx = np.asarray(best_idx).ravel()
            best_sim = np.asarray(best_sim).ravel()
        del sims
    
    picked = category_df.iloc[best_idx][['Ангилал', 'Төрөл', 'Ерөнхий ангилал', 'Сегмент']].reset_index(drop=True)
    batch_result = pd.DataFrame({
        'Барааны нэр': products_batch, 
        'Магадлал': best_sim
    })
    batch_result = pd.concat([batch_result, picked], axis=1)
    
    del batch_vecs
    gc.collect()
    
    return batch_result


def _read_excel_required(file_bytes: bytes, label: str) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    except Exception as e:
        raise ClassificationError(f"'{label}' файлыг уншиж чадсангүй: {e}")

def _normalize_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str, copy=False).str.strip().str.lower()

def _ensure_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    for col in required:
        if col not in df.columns:
            df[col] = ""
    return df[required]

def _create_key_text(df: pd.DataFrame) -> pd.Series:
    return (
        df['Төрөл'].astype(str) + " " +
        df['Ерөнхий ангилал'].astype(str) + " " +
        df['Ангилал'].astype(str) + " " +
        df['Тайлбар'].astype(str) + " " +
        df['Бренд'].astype(str)
    )

def _save_xlsx(df: pd.DataFrame, path: Path) -> None:
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')


def run_classification(
    sales_bytes: bytes,
    category_bytes: bytes,
    manual_bytes: Optional[bytes] = None,
    probability_threshold: float = 0.15,
    batch_size: int = None,
    max_workers: int = 3,
    max_features: int = 20000,
) -> Tuple[str, pd.DataFrame]:

    if batch_size is None:
        batch_size = calculate_batch_size()

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

    try:
        cat_xls = pd.ExcelFile(io.BytesIO(category_bytes), engine="openpyxl")
    except Exception as e:
        raise ClassificationError(f"Ангиллын Excel-ийг нээж чадсангүй: {e}")

    needed_cols = ['Ерөнхий ангилал', 'Төрөл', 'Ангилал', 'Тайлбар', 'Бренд', 'Сегмент']
    base_cols = ['Ерөнхий ангилал', 'Төрөл', 'Ангилал', 'Тайлбар', 'Бренд']
    category_dfs = []

    def parse_sheet(sheet_name: str) -> tuple[str, pd.DataFrame]:
        df = cat_xls.parse(sheet_name, dtype=str)
        return sheet_name, df

    with ThreadPoolExecutor(max_workers=min(4, len(cat_xls.sheet_names))) as executor:
        future_to_sheet = {
            executor.submit(partial(parse_sheet, sheet)): sheet
            for sheet in cat_xls.sheet_names
        }
        for future in as_completed(future_to_sheet):
            sheet_name = future_to_sheet[future]
            try:
                _, sheet_df = future.result()
                sheet_df = _ensure_columns(sheet_df, base_cols).fillna('')
                for col in base_cols:
                    sheet_df[col] = sheet_df[col].astype(str).str.strip().str.lower()
                sheet_df['Сегмент'] = sheet_name
                category_dfs.append(sheet_df)
            except Exception:
                continue

    if not category_dfs:
        raise ClassificationError("Ангиллын Excel-д хүчинтэй sheet олдсонгүй.")

    category_df = pd.concat(category_dfs, ignore_index=True)
    del category_dfs, cat_xls
    gc.collect()

    category_df = _ensure_columns(category_df, needed_cols)
    category_df['түлхүүр_текст'] = _create_key_text(category_df)
    cat_texts = category_df['түлхүүр_текст'].values

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

    cat_vecs = vectorizer.fit_transform(cat_texts)
    
    product_batches = []
    for i in range(0, len(unique_products), batch_size):
        batch = unique_products[i:i + batch_size]
        product_batches.append(batch)
    
    all_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(
                process_product_batch, 
                batch, vectorizer, cat_vecs, category_df
            ): i 
            for i, batch in enumerate(product_batches)
        }
        
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_result = future.result()
                all_results.append((batch_idx, batch_result))
            except Exception:
                raise

    all_results.sort(key=lambda x: x[0])
    classified_batches = [result for _, result in all_results]
    classified = pd.concat(classified_batches, ignore_index=True)
    
    del all_results, classified_batches, cat_vecs
    gc.collect()

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

        classified = classified.drop(columns=[c for c in classified.columns if c.endswith('_гар')], errors='ignore')

    final_result = sales_df.merge(classified, on='Барааны нэр', how='left')

    job_id = uuid.uuid4().hex[:12]
    out_path = STORAGE_DIR / f"angilsan_{job_id}.xlsx"
    _save_xlsx(final_result, out_path)

    return job_id, final_result