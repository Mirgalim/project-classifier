import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from classifier import run_classification, save_excel_file, STORAGE_DIR, ClassificationError

app = FastAPI(title="TF-IDF Classifier API", version="1.1.0")

RESULTS_CACHE = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
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
    max_features: int = Form(10000),
    engine: str = Form("smart")  # "smart" | "script"
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
            max_features=max_features,
            engine=engine
        )

        RESULTS_CACHE[job_id] = df

        preview = df.head(100).fillna("").to_dict(orient='records')
        return JSONResponse({
            "status": "success",
            "engine": engine,
            "job_id": job_id,
            "rows": len(df),
            "preview": preview,
            "download_url": f"/api/download/{job_id}",
            "processing_time": "Fast! No file saved yet."
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
    
    if not file_path.exists():
        try:
            save_excel_file(df, job_id)
        except Exception as e:
            raise HTTPException(500, f"Excel файл үүсгэхэд алдаа гарлаа: {e}")
    
    background_tasks.add_task(cleanup_cache, job_id)
    
    return StreamingResponse(
        open(file_path, 'rb'),
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={
            "Content-Disposition": f"attachment; filename=angilsan_{job_id}.xlsx"
        }
    )

def cleanup_cache(job_id: str):
    if job_id in RESULTS_CACHE:
        del RESULTS_CACHE[job_id]

@app.get("/api/results/{job_id}")
async def get_results(job_id: str, page: int = 0, size: int = 100):
    if job_id not in RESULTS_CACHE:
        raise HTTPException(404, "Results not found")
    
    df = RESULTS_CACHE[job_id]
    start_idx = page * size
    end_idx = start_idx + size
    
    page_data = df.iloc[start_idx:end_idx].fillna("").to_dict(orient='records')
    
    return {
        "job_id": job_id,
        "page": page,
        "size": size,
        "total_rows": len(df),
        "total_pages": (len(df) + size - 1) // size,
        "data": page_data
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
