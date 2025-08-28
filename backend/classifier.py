import io
import uuid
import time
from pathlib import Path
from typing import Optional, Tuple
import gc

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openpyxl


STORAGE_DIR = Path(__file__).parent / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

class ClassificationError(Exception):
    pass

class Timer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"⏱️  {self.name}: {elapsed:.2f}s")


def _read_excel_fast(file_bytes: bytes, label: str) -> pd.DataFrame:
    with Timer(f"Reading {label}"):
        try:
            return pd.read_excel(
                io.BytesIO(file_bytes),
                engine="openpyxl",
                dtype=str,
                na_filter=False
            )
        except Exception as e:
            raise ClassificationError(f"'{label}' файлыг уншиж чадсангүй: {e}")

def _normalize_text_fast(s: pd.Series) -> pd.Series:
    return s.str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)

def _ensure_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    for col in required:
        if col not in df.columns:
            df[col] = ""
    return df[required].copy()

def _create_key_text_fast(df: pd.DataFrame) -> pd.Series:
    # 'Төрөл'-ийг давтан оруулж жин өгнө (анхны "зөв" скрипттэй ижил)
    return (
        df['Төрөл'].astype(str) + " " +
        df['Төрөл'].astype(str) + " " +
        df['Ерөнхий ангилал'].astype(str) + " " +
        df['Ангилал'].astype(str) + " " +
        df['Тайлбар'].astype(str) + " " +
        df['Бренд'].astype(str)
    )

def _save_xlsx_ultra_fast(df: pd.DataFrame, path: Path) -> None:
    with Timer("Saving Excel (ultra fast)"):
        # Сонголтоор Parquet давхар хадгалж болно
        try:
            import pyarrow.parquet as pq
            import pyarrow as pa
            parquet_path = path.with_suffix('.parquet')
            table = pa.Table.from_pandas(df)
            pq.write_table(table, parquet_path)
        except ImportError:
            pass

        # Заавал .xlsx-ээ бичнэ
        try:
            with pd.ExcelWriter(
                path,
                engine='xlsxwriter',
                options={
                    'strings_to_numbers': False,
                    'constant_memory': True,
                    'remove_timezone': True
                }
            ) as writer:
                df.to_excel(
                    writer,
                    index=False,
                    sheet_name='Results',
                    float_format='%.3f'
                )
            return
        except Exception:
            # Том dataframe үед хэсэгчлэн бичих fallback
            chunk_size = 10000
            first_chunk = True
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                if first_chunk:
                    chunk.to_excel(path, index=False, engine='openpyxl')
                    first_chunk = False
                else:
                    with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                        chunk.to_excel(writer, index=False, startrow=i+1, header=False)


def run_classification(
    sales_bytes: bytes,
    category_bytes: bytes,
    manual_bytes: Optional[bytes] = None,
    probability_threshold: float = 0.15,
    batch_size: int = 2000,
    max_features: int = 10000,
) -> Tuple[str, pd.DataFrame]:

    total_start = time.time()

    sales_df = _read_excel_fast(sales_bytes, "sales.xlsx")
    if 'Барааны нэр' not in sales_df.columns:
        raise ClassificationError("sales.xlsx дотор 'Барааны нэр' багана байх ёстой!")

    with Timer("Processing sales data"):
        sales_df['Барааны нэр'] = _normalize_text_fast(sales_df['Барааны нэр'])
        unique_products = sales_df['Барааны нэр'].dropna().unique()
        print(f"📊 Unique products: {len(unique_products)}")

    if unique_products.size == 0:
        job_id = uuid.uuid4().hex[:12]
        out_path = STORAGE_DIR / f"angilsan_{job_id}.xlsx"
        _save_xlsx_ultra_fast(sales_df, out_path)
        return job_id, sales_df

    with Timer("Reading category Excel"):
        try:
            excel_file = pd.ExcelFile(io.BytesIO(category_bytes), engine="openpyxl")
            all_sheets = {}
            for sheet_name in excel_file.sheet_names:
                try:
                    all_sheets[sheet_name] = excel_file.parse(
                        sheet_name,
                        dtype=str,
                        na_filter=False
                    )
                except Exception:
                    continue
        except Exception as e:
            raise ClassificationError(f"Ангиллын Excel-ийг нээж чадсангүй: {e}")

    with Timer("Processing categories"):
        needed_cols = ['Ерөнхий ангилал', 'Төрөл', 'Ангилал', 'Тайлбар', 'Бренд', 'Сегмент']
        base_cols   = ['Ерөнхий ангилал', 'Төрөл', 'Ангилал', 'Тайлбар', 'Бренд']

        category_dfs = []
        for sheet_name, sheet_df in all_sheets.items():
            if sheet_df.empty:
                continue
            try:
                sheet_df = _ensure_columns(sheet_df, base_cols).fillna('')
                for col in base_cols:
                    sheet_df[col] = _normalize_text_fast(sheet_df[col])
                sheet_df['Сегмент'] = sheet_name
                category_dfs.append(sheet_df)
            except Exception:
                continue

        if not category_dfs:
            raise ClassificationError("Ангиллын Excel-д хүчинтэй sheet олдсонгүй.")

        category_df = pd.concat(category_dfs, ignore_index=True)
        category_df = _ensure_columns(category_df, needed_cols)
        print(f"📊 Category entries: {len(category_df)}")

    with Timer("Creating text features"):
        category_df['түлхүүр_текст'] = _create_key_text_fast(category_df)
        cat_texts = category_df['түлхүүр_текст'].values

    if cat_texts.size == 0:
        raise ClassificationError("Ангиллын текст хоосон байна.")

    with Timer("TF-IDF vectorization"):
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            dtype=np.float32,
            lowercase=False,      # бид аль хэдийн lowercase хийсэн
            norm='l2',
            ngram_range=(1, 1),
            token_pattern=r'\b\w+\b'
        )

        # ✅ ЗӨВ: products + categories нийлүүлж fit хийнэ
        combined_texts = np.concatenate([unique_products, cat_texts], axis=0)
        tfidf_all = vectorizer.fit_transform(combined_texts)

        prod_vecs = tfidf_all[:len(unique_products)]
        cat_vecs  = tfidf_all[len(unique_products):]

        print(f"📊 Vector shapes: categories {cat_vecs.shape}, products {prod_vecs.shape}")

    with Timer("Computing similarities"):
        # Жижиг хэмжээтэй үед cosine_similarity, их үед linear_kernel
        if len(unique_products) < 5000 and len(category_df) < 10000:
            similarities = cosine_similarity(prod_vecs, cat_vecs)
            best_idx = similarities.argmax(axis=1).astype(np.int32)
            best_sim = similarities.max(axis=1).astype(np.float32)
        else:
            from sklearn.metrics.pairwise import linear_kernel
            similarities = linear_kernel(prod_vecs, cat_vecs)
            best_idx = similarities.argmax(axis=1).astype(np.int32)
            best_sim = similarities.max(axis=1).astype(np.float32)

        del similarities, prod_vecs, cat_vecs
        gc.collect()

    with Timer("Building results"):
        picked = category_df.iloc[best_idx][['Ангилал', 'Төрөл', 'Ерөнхий ангилал', 'Сегмент']].reset_index(drop=True)
        classified = pd.DataFrame({
            'Барааны нэр': unique_products,
            'Магадлал': best_sim
        })
        classified = pd.concat([classified, picked], axis=1)

    if manual_bytes is not None:
        with Timer("Applying manual overrides"):
            manual_df = _read_excel_fast(manual_bytes, "manual_fix.xlsx")
            if 'Барааны нэр' not in manual_df.columns:
                raise ClassificationError("manual_fix.xlsx дотор 'Барааны нэр' багана байх ёстой!")
            manual_df['Барааны нэр'] = _normalize_text_fast(manual_df['Барааны нэр'])

            classified = classified.merge(manual_df, on='Барааны нэр', how='left', suffixes=('', '_гар'))

            low_conf = classified['Магадлал'] < float(probability_threshold)

            def non_empty(series: pd.Series) -> pd.Series:
                # "" болон зөвхөн whitespace-ийг үл тооно
                return series.astype(str).str.strip().ne('')

            for col in ['Ангилал', 'Төрөл', 'Ерөнхий ангилал', 'Сегмент']:
                mcol = f"{col}_гар"
                if mcol in classified.columns:
                    mask = low_conf & non_empty(classified[mcol])
                    classified.loc[mask, col] = classified.loc[mask, mcol]

            classified = classified.drop(columns=[c for c in classified.columns if c.endswith('_гар')], errors='ignore')

    with Timer("Final merge"):
        final_result = sales_df.merge(classified, on='Барааны нэр', how='left')

    job_id = uuid.uuid4().hex[:12]

    total_time = time.time() - total_start
    print(f"🎉 Total processing time: {total_time:.2f} seconds")
    print(f"📊 Processed {len(unique_products)} products with {len(category_df)} categories")

    return job_id, final_result


def save_excel_file(df: pd.DataFrame, job_id: str) -> Path:
    out_path = STORAGE_DIR / f"angilsan_{job_id}.xlsx"
    _save_xlsx_ultra_fast(df, out_path)
    return out_path
