import type { LayerData } from "../types";

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename.replace(/[\\/\u0000-\u001f]/g, "_");
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 30000);
}

export function exportCanvasImage(
  canvas: HTMLCanvasElement,
  filename = "neuralis-visualization.png",
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (!canvas.width || !canvas.height) {
      reject(new Error("Canvas has no pixels to export."));
      return;
    }
    try {
      canvas.toBlob((blob) => {
        if (!blob) {
          reject(new Error("The renderer could not encode this image."));
          return;
        }
        downloadBlob(blob, filename);
        resolve();
      }, "image/png");
    } catch {
      reject(new Error("Canvas image export was blocked by browser security."));
    }
  });
}

export function exportTensorCSV(
  matrix: readonly (readonly number[])[],
  filename = "tensor.csv",
): void {
  const csv = matrix
    .map((row) =>
      row
        .map((value) => (Number.isFinite(value) ? String(value) : ""))
        .join(","),
    )
    .join("\n");
  downloadBlob(new Blob([csv], { type: "text/csv;charset=utf-8" }), filename);
}

export function exportLayersJSON(
  layers: LayerData[],
  filename = "neuralis-tensors.json",
): void {
  const body = JSON.stringify(
    {
      format: "neuralis.layers",
      version: 1,
      capturedAt: new Date().toISOString(),
      layers,
    },
    null,
    2,
  );
  downloadBlob(new Blob([body], { type: "application/json" }), filename);
}

export function exportSVG(
  svg: SVGSVGElement,
  filename = "neuralis-chart.svg",
): void {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const body = new XMLSerializer().serializeToString(clone);
  downloadBlob(
    new Blob([body], { type: "image/svg+xml;charset=utf-8" }),
    filename,
  );
}
