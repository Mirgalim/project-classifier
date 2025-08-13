"use client";
import { useId, useRef, useState } from "react";

type Props = {
  label: string;
  required?: boolean;
  accept?: string;
  onChange: (f: File | undefined) => void;
};

export default function UploadCard({ label, required, accept, onChange }: Props) {
  const id = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [hover, setHover] = useState(false);
  const [fileName, setFileName] = useState<string>("");

  function handleFile(f?: File) {
    setFileName(f?.name || "");
    onChange(f);
  }

  return (
    <div
      className={`card p-4 min-w-[280px] flex-1 transition ring-1 ring-transparent ${hover ? "ring-emerald-500/30" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setHover(true); }}
      onDragLeave={() => setHover(false)}
      onDrop={(e) => { e.preventDefault(); setHover(false); const f = e.dataTransfer.files?.[0]; if (f) handleFile(f); }}
    >
      <label htmlFor={id} className="block mb-2 text-sm font-semibold text-zinc-300">
        {label} {required && <span className="text-emerald-400">*</span>}
      </label>
      <div
        className="border-2 border-dashed border-border/80 rounded-xl p-4 text-center cursor-pointer hover:border-emerald-500/40"
        onClick={() => inputRef.current?.click()}
      >
        <p className="text-sm text-zinc-400">
          {fileName ? <span className="text-zinc-200">{fileName}</span> : "Drag & drop or click to choose .xlsx"}
        </p>
        <p className="text-[11px] text-zinc-500 mt-1">.xlsx файлууд</p>
      </div>
      <input
        id={id}
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
    </div>
  );
}