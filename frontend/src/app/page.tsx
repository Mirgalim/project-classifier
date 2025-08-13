"use client";
import { useState } from "react";
import UploadCard from "../components/UploadCard";
import ResultsTable from "../components/ResultTable";
import { API, uploadAndClassify } from "../lib/api";

export default function Page() {
  const [sales, setSales] = useState<File>();
  const [category, setCategory] = useState<File>();
  const [manual, setManual] = useState<File>();
  const [threshold, setThreshold] = useState(0.15);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<null | { job_id: string; rows: number; preview: any[]; download_url: string; }>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    if (!sales || !category) { setErr("sales.xlsx ба Nomin_ba3.xlsx заавал!"); return; }
    setLoading(true);
    try {
      const res = await uploadAndClassify({ sales, category, manual, threshold });
      setResult(res);
    } catch (e: any) {
      setErr(e.message || "Алдаа гарлаа");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container py-12">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl md:text-4xl font-extrabold leading-tight"><span className="heading-gradient">🧹 Product Classifier</span></h1>
          <p className="mt-2 text-sm text-zinc-400">Файл байршуулах → TF‑IDF-ээр ангилах → цэвэр Excel-ийг урьдчилан үзэх, татаж авах.</p>
        </div>
        {result && (
          <a className="btn-secondary h-10 px-5" href={`${API}${result.download_url}`}>⬇️ Татах</a>
        )}
      </header>

      <form onSubmit={onSubmit} className="grid grid-cols-1 lg:grid-cols-4 gap-4 mt-8">
        <UploadCard label="sales.xlsx" required accept=".xlsx" onChange={setSales} />
        <UploadCard label="Nomin_ba3.xlsx (categories)" required accept=".xlsx" onChange={setCategory} />
        <UploadCard label="manual_fix.xlsx (optional)" accept=".xlsx" onChange={setManual} />

        <div className="card p-5 min-w-[280px] flex flex-col justify-between">
          <div>
            <label className="block mb-2 text-sm font-semibold text-zinc-300">Threshold (магадлал)</label>
            <input type="number" step="0.01" min={0} max={1} value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value || '0.15'))}
              className="input w-44" />
            <p className="mt-2 text-xs text-zinc-400">Default: 0.15 — бага байх тусам manual override илүү хурдан үйлчилнэ</p>
          </div>
          <button type="submit" disabled={loading} className="btn mt-4 h-10">
            {loading ? (
              <span className="inline-flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeWidth="4" className="opacity-25"/><path d="M4 12a8 8 0 018-8" fill="none" stroke="currentColor" strokeWidth="4" className="opacity-75"/></svg>
                Ажиллаж байна....
              </span>
            ) : "Ажиллуулах"}
          </button>
        </div>
      </form>

      {err && (
        <div className="card mt-6 p-5 border border-red-900/50">
          <strong className="text-red-400">Алдаа:</strong> <span className="text-red-200">{err}</span>
        </div>
      )}

      {result && (
        <>
          <section className="mt-6">
            <div className="card p-5 flex items-center justify-between">
              <div>
                <h3 className="m-0 text-lg font-semibold">ID: <span className="badge ml-2">{result.job_id}</span></h3>
                <p className="text-xs text-zinc-400">Нийт мөр: {result.rows.toLocaleString()}</p>
              </div>
              <a className="btn-secondary h-10 px-5" href={`${API}${result.download_url}`}>⬇️ Татах Excel</a>
            </div>
          </section>
          <ResultsTable data={result.preview} />
        </>
      )}
    </main>
  );
}