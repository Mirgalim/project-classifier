import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === 1. Борлуулалтын датаг унших ===
sales_file = "sales.xlsx"
sales_df = pd.read_excel(sales_file)

# lowercase + цэвэрлэгээ (илүү олон whitespace-ийг нэг болгоно)
sales_df['Барааны нэр'] = (
    sales_df['Барааны нэр']
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r'\s+', ' ', regex=True)
)

unique_products = sales_df['Барааны нэр'].unique()

# === 2. Ангиллын файлын бүх sheet-үүдийг нэгтгэх ===
category_file = "Nomin_ba3.xlsx"
category_xls = pd.ExcelFile(category_file)
category_df = pd.DataFrame()

for sheet in category_xls.sheet_names:
    sheet_df = category_xls.parse(sheet)
    sheet_df['Сегмент'] = sheet
    category_df = pd.concat([category_df, sheet_df], ignore_index=True)

# === 3. Багануудыг цэвэрлэх ===
keep_cols = ['Ерөнхий ангилал', 'Төрөл', 'Ангилал', 'Тайлбар', 'Бренд', 'Сегмент']
category_df = category_df[keep_cols]
category_df = category_df.fillna('')

# бүх текстийг lowercase + цэвэрлэгээ
category_df = category_df.applymap(lambda x: str(x).lower().strip())
category_df = category_df.applymap(lambda x: x.replace('\n', ' '))
category_df = category_df.applymap(lambda x: pd.Series(x).str.replace(r'\s+', ' ', regex=True).iloc[0])

# === 4. Түлхүүр текст үүсгэх (Төрөл-ийг 2 удаа давтаж жин өгнө) ===
category_df['түлхүүр_текст'] = (
    category_df['Төрөл'] + " " +
    category_df['Төрөл'] + " " +
    category_df['Ерөнхий ангилал'] + " " +
    category_df['Ангилал'] + " " +
    category_df['Тайлбар'] + " " +
    category_df['Бренд']
)

# === 5. Векторжуулалт ба cosine similarity ===
# products + categories-ийг нийлүүлж fit_transform хийнэ
texts = list(unique_products) + list(category_df['түлхүүр_текст'])
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

product_vecs = tfidf_matrix[:len(unique_products)]
category_vecs = tfidf_matrix[len(unique_products):]
similarity_matrix = cosine_similarity(product_vecs, category_vecs)

# === 6. Хамгийн тохирох ангиллыг оноох ===
best_match_indices = similarity_matrix.argmax(axis=1)
best_match_scores = similarity_matrix.max(axis=1)

classified = pd.DataFrame({
    'Барааны нэр': unique_products,
    'Ангилал': category_df.iloc[best_match_indices]['Ангилал'].values,
    'Төрөл': category_df.iloc[best_match_indices]['Төрөл'].values,
    'Ерөнхий ангилал': category_df.iloc[best_match_indices]['Ерөнхий ангилал'].values,
    'Сегмент': category_df.iloc[best_match_indices]['Сегмент'].values,
    'Магадлал': best_match_scores
})

# === ✅ 6.1 Гар аргаар зассан файл оруулж нэгтгэх ===
manual_fix = pd.read_excel("manual_fix.xlsx")
manual_fix['Барааны нэр'] = (
    manual_fix['Барааны нэр']
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r'\s+', ' ', regex=True)
)

# Магадлал бага байвал гар аргын өгөгдлөөр дарна (хоосон мөрөөр дарж цоолчихоос сэргийлнэ)
classified = classified.merge(manual_fix, on='Барааны нэр', how='left', suffixes=('', '_гар'))

def _use_manual(row, col, thr=0.15):
    mcol = f"{col}_гар"
    val = row.get(mcol, None)
    # pd.notna ба хоосон биш эсэхийг шалгана
    if row['Магадлал'] < thr and pd.notna(val) and str(val).strip() != "":
        return val
    return row[col]

for col in ['Ангилал', 'Төрөл', 'Ерөнхий ангилал', 'Сегмент']:
    classified[col] = classified.apply(lambda r, c=col: _use_manual(r, c, 0.15), axis=1)

classified = classified.drop(columns=[col for col in classified.columns if col.endswith('_гар')])

# === 7. Ангиллыг бүх мөрт буцааж тараах ===
final_result = sales_df.merge(classified, on='Барааны нэр', how='left')

# === 8. Excel файлд хадгалах ===
out_name = "angilsan_baraa_torol_priority13.xlsx"
final_result.to_excel(out_name, index=False)
print(f"✔ Амжилттай хадгаллаа: {out_name}")
