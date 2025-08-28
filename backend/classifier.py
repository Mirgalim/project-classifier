import io
import re
import uuid
import time
from pathlib import Path
from typing import Optional, Tuple
import gc

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ====== Storage ======
STORAGE_DIR = Path(__file__).parent / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


# ====== Errors & Utils ======
class ClassificationError(Exception):
    pass


class Timer:
    def __init__(self, name: str):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        print(f"⏱️  {self.name}: {time.time() - self.start:.2f}s")


# ====== Heuristic regex (MN/RU mix) ======
CONSUMABLE_PAT = re.compile(
    r"(?:\b|\s)(?:г|гр|грам|ml|мл|kg|кг|шт|ширхэг|уут|пакет|sachet|stick|стик|"
    r"3в1|3-in-1|mix|капучино|латте|эспрессо|espresso|instant|classic|gold|"
    r"jacobs|nescafe|maccoffee|tea|чай|пакетик)(?:\b|\s)"
)
APPLIANCE_PAT = re.compile(
    r"(?:кофе\s?чанагч|кофе\s?машин|кофемашин|электро|цахилгаан|гал тогооны цахилгаан|"
    r"чайник|kettle|машин|ватт|w\b|Вт\b)"
)

# Quick keyword hints for UNCLASSIFIED fallback
KW = {
    "coffee": re.compile(r"\b(кофе|капучино|латте|эспрессо|espresso|3в1|3-in-1)\b", re.I),
    "tea":    re.compile(r"\b(цай|tea|chai|масала)\b", re.I),
    "beer":   re.compile(r"\b(пиво|beer|айраг)\b", re.I),
    "water":  re.compile(r"\b(ус|water)\b", re.I),
    "juice":  re.compile(r"\b(жүүс|juice)\b", re.I),
    "choco":  re.compile(r"\b(шоколад|choco|kinder|nestle|snickers|mars)\b", re.I),
    "candy":  re.compile(r"\b(чихэр|lollipop|trolli|candy|sour)\b", re.I),
}


# ====== Helpers ======
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
    return (
        s.astype(str)
         .str.lower()
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
    )


def _ensure_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    for col in required:
        if col not in df.columns:
            df[col] = ""
    return df[required].copy()


def _create_key_text_fast(df: pd.DataFrame) -> pd.Series:
    # 'Төрөл'-ийг хоёр давтаж жин өгнө (анхны логиктой ижил)
    return (
        df["Төрөл"].astype(str) + " " +
        df["Төрөл"].astype(str) + " " +
        df["Ерөнхий ангилал"].astype(str) + " " +
        df["Ангилал"].astype(str) + " " +
        df["Тайлбар"].astype(str) + " " +
        df["Бренд"].astype(str)
    )


def _word_count(series: pd.Series) -> pd.Series:
    return series.str.replace(r"[^\w\s]+", " ", regex=True).str.split().map(
        lambda x: len(x) if isinstance(x, list) else 0
    )


def _category_hint_tags(row: pd.Series) -> tuple[str, bool, bool]:
    text = " ".join([
        str(row.get("Төрөл", "")),
        str(row.get("Ерөнхий ангилал", "")),
        str(row.get("Ангилал", "")),
        str(row.get("Сегмент", "")),
    ]).lower()
    is_appliance = bool(APPLIANCE_PAT.search(text))
    is_consumable = ("кофе" in text) or ("найруулдаг" in text) or ("уух" in text) or ("beverage" in text)
    tags = []
    if is_appliance:
        tags.append("tag_appliance electronics kitchen_appliance")
    if is_consumable:
        tags.append("tag_consumable beverage coffee instant")
    return (" ".join(tags), is_consumable, is_appliance)


def _product_hint_tags(name: str) -> tuple[str, bool, bool]:
    is_consumable = bool(CONSUMABLE_PAT.search(name))
    is_appliance  = bool(APPLIANCE_PAT.search(name))
    tags = []
    if is_consumable:
        tags.append("tag_consumable beverage coffee instant пакет уут гр мл")
    if is_appliance:
        tags.append("tag_appliance electronics kitchen_appliance")
    return (" ".join(tags), is_consumable, is_appliance)


def _fallback_guess(name: str) -> Optional[tuple[str, str, str]]:
    if KW["coffee"].search(name):
        return ("Найруулдаг кофе", "Кофе", "боловсруулсан хүнс")
    if KW["tea"].search(name):
        return ("Цай", "Цай", "боловсруулсан хүнс")
    if KW["beer"].search(name):
        return ("Пиво", "Согтууруулах ундаа", "Шингэн хүнс")
    if KW["water"].search(name):
        return ("Ус", "Ундаа", "Шингэн хүнс")
    if KW["juice"].search(name):
        return ("Жүүс", "Ундаа", "Шингэн хүнс")
    if KW["choco"].search(name):
        return ("Шоколад", "Чихэр", "Амттан")
    if KW["candy"].search(name):
        return ("Шийтэн/Лоллипоп", "Чихэр", "Амттан")
    return None


def _save_xlsx_ultra_fast(df: pd.DataFrame, path: Path) -> None:
    with Timer("Saving Excel"):
        # Сонголтоор Parquet хадгална
        try:
            import pyarrow.parquet as pq
            import pyarrow as pa
            pq.write_table(pa.Table.from_pandas(df), path.with_suffix(".parquet"))
        except Exception:
            pass

        # Заавал .xlsx бичих
        try:
            with pd.ExcelWriter(
                path, engine="xlsxwriter",
                options={"strings_to_numbers": False, "constant_memory": True, "remove_timezone": True}
            ) as w:
                df.to_excel(w, index=False, sheet_name="Results", float_format="%.3f")
            return
        except Exception:
            # Том DF үед хэсэгчлэн бичих fallback
            chunk = 10000
            first = True
            for i in range(0, len(df), chunk):
                part = df.iloc[i:i+chunk]
                if first:
                    part.to_excel(path, index=False, engine="openpyxl")
                    first = False
                else:
                    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as w:
                        part.to_excel(w, index=False, startrow=i+1, header=False)


# ====== Main ======
def run_classification(
    sales_bytes: bytes,
    category_bytes: bytes,
    manual_bytes: Optional[bytes] = None,
    probability_threshold: float = 0.15,
    batch_size: int = 2000,     # reserved; not used in this variant
    max_features: int = 10000,
) -> Tuple[str, pd.DataFrame]:

    total_start = time.time()

    # --- Sales
    sales_df = _read_excel_fast(sales_bytes, "sales.xlsx")
    if "Барааны нэр" not in sales_df.columns:
        raise ClassificationError("sales.xlsx дотор 'Барааны нэр' багана байх ёстой!")

    with Timer("Processing sales data"):
        sales_df["Барааны нэр"] = _normalize_text_fast(sales_df["Барааны нэр"])
        unique_products = sales_df["Барааны нэр"].dropna().unique()
        print(f"📊 Unique products: {len(unique_products)}")

    if unique_products.size == 0:
        job_id = uuid.uuid4().hex[:12]
        _save_xlsx_ultra_fast(sales_df, STORAGE_DIR / f"angilsan_{job_id}.xlsx")
        return job_id, sales_df

    # --- Categories
    with Timer("Reading category Excel"):
        try:
            excel_file = pd.ExcelFile(io.BytesIO(category_bytes), engine="openpyxl")
            all_sheets = {}
            for sheet_name in excel_file.sheet_names:
                try:
                    all_sheets[sheet_name] = excel_file.parse(sheet_name, dtype=str, na_filter=False)
                except Exception:
                    continue
        except Exception as e:
            raise ClassificationError(f"Ангиллын Excel-ийг нээж чадсангүй: {e}")

    with Timer("Processing categories"):
        needed = ["Ерөнхий ангилал", "Төрөл", "Ангилал", "Тайлбар", "Бренд", "Сегмент"]
        base   = ["Ерөнхий ангилал", "Төрөл", "Ангилал", "Тайлбар", "Бренд"]

        cat_parts = []
        for sheet_name, sheet_df in all_sheets.items():
            if sheet_df.empty:
                continue
            try:
                sheet_df = _ensure_columns(sheet_df, base).fillna("")
                for col in base:
                    sheet_df[col] = _normalize_text_fast(sheet_df[col])
                sheet_df["Сегмент"] = sheet_name
                cat_parts.append(sheet_df)
            except Exception:
                continue

        if not cat_parts:
            raise ClassificationError("Ангиллын Excel-д хүчинтэй sheet олдсонгүй.")

        category_df = pd.concat(cat_parts, ignore_index=True)
        category_df = _ensure_columns(category_df, needed)
        print(f"📊 Category entries (raw): {len(category_df)}")

    # --- Build texts + drop weak/empty rows
    with Timer("Creating text features"):
        base_text = _create_key_text_fast(category_df).str.strip()
        category_df["түлхүүр_текст"] = base_text

        # Хэт сул/хоосон мөрүүдийг drop (>= 2 үгтэйг үлдээнэ)
        valid_mask = _word_count(category_df["түлхүүр_текст"]) >= 2
        dropped = (~valid_mask).sum()
        if dropped:
            print(f"🧹 Dropped empty/weak category rows: {dropped}")
        category_df = category_df.loc[valid_mask].reset_index(drop=True)

        # Хэрвээ бүгд унавал зогсооно
        if category_df.empty:
            raise ClassificationError("Ангиллын текстүүд хоосон байна (бусад sheet-үүдийг шалгана уу).")

        # Domain hint flags for categories
        tags, cat_is_cons, cat_is_appl = [], [], []
        for _, row in category_df.iterrows():
            t, ccons, cappl = _category_hint_tags(row)
            tags.append(t)
            cat_is_cons.append(ccons)
            cat_is_appl.append(cappl)
        category_df["__hint_tags"] = tags
        category_df["__is_consumable"] = np.array(cat_is_cons, dtype=bool)
        category_df["__is_appliance"]  = np.array(cat_is_appl, dtype=bool)

        # Final category text
        cat_texts = (category_df["түлхүүр_текст"] + " " + category_df["__hint_tags"].fillna("")).values

        # Product augmented texts + flags
        prod_aug = []
        prod_is_cons = np.zeros(len(unique_products), dtype=bool)
        prod_is_appl = np.zeros(len(unique_products), dtype=bool)
        for i, name in enumerate(unique_products):
            t, ccons, cappl = _product_hint_tags(name)
            prod_aug.append(f"{name} {t}".strip())
            prod_is_cons[i] = ccons
            prod_is_appl[i] = cappl

    # --- TF-IDF
    with Timer("TF-IDF vectorization"):
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            dtype=np.float32,
            lowercase=False,
            norm="l2",
            ngram_range=(1, 2),            # 1–2 грам: 'кофе чанагч' гэх мэт нийлмэл хэллэг барина
            token_pattern=r"\b\w+\b",
        )
        combined_texts = np.concatenate([np.array(prod_aug, dtype=object), cat_texts], axis=0)
        tfidf_all = vectorizer.fit_transform(combined_texts)
        prod_vecs = tfidf_all[:len(unique_products)]
        cat_vecs  = tfidf_all[len(unique_products):]
        print(f"📊 Vector shapes: categories {cat_vecs.shape}, products {prod_vecs.shape}")

    # --- Similarity (+ domain penalty)
    with Timer("Computing similarities"):
        similarities = cosine_similarity(prod_vecs, cat_vecs)

        # Consumable product ⟷ Appliance category = penalty (40% down)
        rows = np.where(prod_is_cons)[0]
        cols = np.where(category_df["__is_appliance"].values)[0]
        if rows.size and cols.size:
            similarities[np.ix_(rows, cols)] *= 0.6

        # (optionally) Appliance product ⟷ Consumable category penalty
        rows2 = np.where(prod_is_appl)[0]
        cols2 = np.where(category_df["__is_consumable"].values)[0]
        if rows2.size and cols2.size:
            similarities[np.ix_(rows2, cols2)] *= 0.6

        best_idx = similarities.argmax(axis=1).astype(np.int32)
        best_sim = similarities.max(axis=1).astype(np.float32)

        del similarities, prod_vecs, cat_vecs
        gc.collect()

    # --- Build results
    with Timer("Building results"):
        picked = category_df.iloc[best_idx][["Ангилал", "Төрөл", "Ерөнхий ангилал", "Сегмент"]].reset_index(drop=True)
        classified = pd.DataFrame({
            "Барааны нэр": unique_products,
            "Магадлал": best_sim
        })
        classified = pd.concat([classified, picked], axis=1)

        # 0-ижилттэй (эсвэл маш ойр) → UNCLASSIFIED + optional keyword guess
        zero_mask = best_sim <= 1e-8
        if zero_mask.any():
            print(f"⚠️ No-match products: {zero_mask.sum()}")
            classified.loc[zero_mask, ["Ангилал", "Төрөл", "Ерөнхий ангилал", "Сегмент"]] = [
                "UNCLASSIFIED", "UNCLASSIFIED", "UNCLASSIFIED", ""
            ]
            # Жижиг дүрмийн fallback (сонголтоор)
            for i in np.where(zero_mask)[0]:
                name = unique_products[i]
                guess = _fallback_guess(name)
                if guess:
                    a, t, e = guess
                    classified.loc[i, ["Ангилал", "Төрөл", "Ерөнхий ангилал"]] = [a, t, e]

    # --- Manual overrides
    if manual_bytes is not None:
        with Timer("Applying manual overrides"):
            manual_df = _read_excel_fast(manual_bytes, "manual_fix.xlsx")
            if "Барааны нэр" not in manual_df.columns:
                raise ClassificationError("manual_fix.xlsx дотор 'Барааны нэр' багана байх ёстой!")
            manual_df["Барааны нэр"] = _normalize_text_fast(manual_df["Барааны нэр"])

            classified = classified.merge(manual_df, on="Барааны нэр", how="left", suffixes=("", "_гар"))

            low_conf = classified["Магадлал"] < float(probability_threshold)

            def non_empty(s: pd.Series) -> pd.Series:
                return s.astype(str).str.strip().ne("")

            for col in ["Ангилал", "Төрөл", "Ерөнхий ангилал", "Сегмент"]:
                mcol = f"{col}_гар"
                if mcol in classified.columns:
                    mask = low_conf & non_empty(classified[mcol])
                    classified.loc[mask, col] = classified.loc[mask, mcol]

            classified = classified.drop(columns=[c for c in classified.columns if c.endswith("_гар")], errors="ignore")

    # --- Final merge
    with Timer("Final merge"):
        final_result = sales_df.merge(classified, on="Барааны нэр", how="left")

    job_id = uuid.uuid4().hex[:12]
    print(f"🎉 Total processing time: {time.time() - total_start:.2f}s")
    print(f"📊 Processed {len(unique_products)} products with {len(category_df)} categories")

    return job_id, final_result


def save_excel_file(df: pd.DataFrame, job_id: str) -> Path:
    out_path = STORAGE_DIR / f"angilsan_{job_id}.xlsx"
    _save_xlsx_ultra_fast(df, out_path)
    return out_path
