"use client";
import { useId } from "react";

type Props = {
  label: string;
  required?: boolean;
  accept?: string;
  onChange: (f: File | undefined) => void;
};

export default function FilePicker({ label, required, accept, onChange }: Props) {
  const id = useId();
  return (
    <div className="card p-4 flex-1 min-w-[280px]">
      <label htmlFor={id} className="block mb-2 text-sm font-semibold text-zinc-300">
        {label} {required && <span className="text-emerald-400">*</span>}
      </label>
      <input
        id={id}
        type="file"
        accept={accept}
        onChange={(e) => onChange(e.target.files?.[0])}
        className="input border-dashed"
      />
      <p className="mt-2 text-xs text-zinc-400">.xlsx файлууд</p>
    </div>
  );
}