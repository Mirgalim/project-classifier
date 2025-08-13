export const API = process.env.NEXT_PUBLIC_API || "https://project-classifier.onrender.com";

export async function uploadAndClassify({
  sales,
  category,
  manual,
  threshold,
}: {
  sales: File;
  category: File;
  manual?: File;
  threshold: number;
}) {
  const fd = new FormData();
  fd.append("sales", sales);
  fd.append("category", category);
  if (manual) fd.append("manual", manual);
  fd.append("threshold", String(threshold));

  const res = await fetch(`${API}/api/classify`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json() as Promise<{
    status: string;
    job_id: string;
    rows: number;
    preview: Record<string, any>[];
    download_url: string;
  }>;
}