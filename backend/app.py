import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from classifier import run_classification, STORAGE_DIR, ClassificationError

app = FastAPI(title="TF-IDF Classifier API", version="1.0.0")

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
    max_workers: int = Form(3),
    batch_size: Optional[int] = Form(None)
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
            max_workers=max_workers,
            batch_size=batch_size
        )

        preview = df.head(100).fillna("").to_dict(orient='records')
        return JSONResponse({
            "status": "success",
            "job_id": job_id,
            "rows": len(df),
            "preview": preview,
            "download_url": f"/api/download/{job_id}"
        })
    except ClassificationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Алдаа гарлаа: {e}")

@app.get("/api/download/{job_id}")
async def download(job_id: str):
    file_path = STORAGE_DIR / f"angilsan_{job_id}.xlsx"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл олдсонгүй")
    return StreamingResponse(
        open(file_path, 'rb'),
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={
            "Content-Disposition": f"attachment; filename=angilsan_{job_id}.xlsx"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)