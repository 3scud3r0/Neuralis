import type { ReactNode } from 'react';
import { AlertCircle, Download, LoaderCircle } from 'lucide-react';
import { downloadText } from '../../app/storage';
import { BackendError } from '../../integrations/backendClient';
import type { ExtensionCapabilities, ExtensionClient } from './extensionClient';
export interface ExtensionContext { client: ExtensionClient | null; capabilities: ExtensionCapabilities | null; }
export function PanelTitle({ eyebrow, title, children }: { eyebrow: string; title: string; children: ReactNode }) {
  return <div className="rw-panel-title"><span className="rw-eyebrow">{eyebrow}</span><h2>{title}</h2><p>{children}</p></div>;
}
export function Stat({ label, value, note }: { label: string; value: ReactNode; note?: string }) {
  return <div className="rw-stat"><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</div>;
}
export function ErrorNotice({ error }: { error: string | null }) { return error ? <div className="rw-error" role="alert"><AlertCircle size={17}/>{error}</div> : null; }
export function errorMessage(failure: unknown): string { return failure instanceof BackendError ? `${failure.message}${failure.action ? ` ${failure.action}` : ''}` : failure instanceof Error ? failure.message : 'Operation failed.'; }
export function DownloadButton({ filename, data, label = 'Export JSON', mime }: { filename: string; data: unknown; label?: string; mime?: string }) {
  return <button onClick={() => downloadText(filename, typeof data === 'string' ? data : JSON.stringify(data, null, 2), mime)}><Download size={15}/>{label}</button>;
}
export function BackendButton({ available, busy, onClick, children }: { available: boolean; busy: boolean; onClick: () => void; children: ReactNode }) {
  return <button className="rw-secondary" disabled={!available || busy} onClick={onClick}>{busy ? <LoaderCircle className="rw-spin" size={15}/> : null}{children}</button>;
}
export function ResultJson({ value, label = 'Response evidence' }: { value: unknown; label?: string }) { return <details className="rw-details"><summary>{label}</summary><pre>{JSON.stringify(value, null, 2)}</pre></details>; }
export function HorizontalBars({ values, labels, signed = true }: { values: number[]; labels?: string[]; signed?: boolean }) {
  const maximum = Math.max(1e-12, ...values.map(Math.abs));
  return <div className="rw-bars">{values.map((value, index) => <div className="rw-bar-row" key={index}><span>{labels?.[index] ?? `Feature ${index + 1}`}</span><div className={`rw-bar-track ${signed ? 'is-signed' : ''}`}><i className={value < 0 ? 'negative' : ''} style={{ width: `${Math.abs(value) / maximum * (signed ? 50 : 100)}%`, left: `${signed ? value < 0 ? 50 - Math.abs(value) / maximum * 50 : 50 : 0}%` }}/></div><code>{value.toFixed(4)}</code></div>)}</div>;
}
