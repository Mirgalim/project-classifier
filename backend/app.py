import io
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from classifier import run_classification, save_excel_file, STORAGE_DIR, ClassificationError

app = FastAPI(title="TF-IDF Classifier API", version="2.0.0")

RESULTS_CACHE = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/classify")
async def classify(
    sales: UploadFile = File(..., description="sales.xlsx"),
    category: UploadFile = File(..., description="Nomin_ba3.xlsx"),
    manual: Optional[UploadFile] = File(None, description="manual_fix.xlsx (optional)"),
    threshold: float = Form(0.15),
    engine: str = Form("script")   # логик адил; параметр хадгалж үлдээв
):
    try:
        sales_bytes = await sales.read()
        cat_bytes = await category.read()
        manual_bytes = await manual.read() if manual is not None else None

        job_id, df = run_classification(
            sales_bytes=sales_bytes,
            category_bytes=cat_bytes,
            manual_bytes=manual_bytes,
            probability_threshold=threshold,
            engine=engine,
        )

        RESULTS_CACHE[job_id] = df  # df аль хэдийн NaN-гүй

        # Preview-д хоосон барааны нэртэй мөрийг нуух
        preview_df = df[df["Барааны нэр"].astype(str).str.strip().ne("")]
        preview = preview_df.head(100).to_dict(orient="records")

        return JSONResponse({
            "status": "success",
            "engine": engine,
            "job_id": job_id,
            "rows": int(len(df)),
            "preview": preview,
            "download_url": f"/api/download/{job_id}",
        })
    except ClassificationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Алдаа гарлаа: {e}")

@app.get("/api/download/{job_id}")
async def download(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in RESULTS_CACHE:
        raise HTTPException(404, "Results not found. Please run classification again.")

    df = RESULTS_CACHE[job_id]
    file_path = STORAGE_DIR / f"angilsan_{job_id}.xlsx"
    try:
        save_excel_file(df, job_id)
    except Exception as e:
        raise HTTPException(500, f"Excel файл үүсгэхэд алдаа гарлаа: {e}")

    background_tasks.add_task(lambda: RESULTS_CACHE.pop(job_id, None))

    return StreamingResponse(
        open(file_path, "rb"),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=angilsan_{job_id}.xlsx"},
    )

@app.get("/api/results/{job_id}")
async def get_results(job_id: str, page: int = 0, size: int = 100):
    if job_id not in RESULTS_CACHE:
        raise HTTPException(404, "Results not found")

    df = RESULTS_CACHE[job_id]
    start, end = page * size, page * size + size
    data = df.iloc[start:end].to_dict(orient="records")
    return {
        "job_id": job_id,
        "page": page,
        "size": size,
        "total_rows": int(len(df)),
        "total_pages": int((len(df) + size - 1) // size),
        "data": data,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
