"use client";

type Props = { data: Record<string, any>[] };

export default function ResultsTable({ data }: Props) {
  if (!data?.length) return null;
  const cols = Object.keys(data[0]);
  return (
    <div className="card mt-4 p-4">
      <div className="flex items-center justify-between">
        <h3 className="m-0 text-lg font-semibold">Preview <span className="badge ml-2">top {data.length}</span></h3>
      </div>
      <div className="overflow-x-auto mt-3">
        <table className="table">
          <thead>
            <tr>
              {cols.map((c) => (
                <th key={c}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={i}>
                {cols.map((c) => (
                  <td key={c}>{String(row[c] ?? "")}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}