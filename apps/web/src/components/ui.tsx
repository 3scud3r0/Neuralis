import { useId, type ReactNode, type ButtonHTMLAttributes } from 'react';
import { ArrowUpRight, ChevronDown, Info } from 'lucide-react';
import { Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { MetricPoint } from '../types';

export function Button({ children, variant = 'secondary', className = '', ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'secondary' | 'ghost' | 'danger' }) {
  return <button className={`button button-${variant} ${className}`} {...props}>{children}</button>;
}
export function Badge({ children, tone = 'neutral', dot = false }: { children: ReactNode; tone?: 'neutral' | 'cyan' | 'violet' | 'green' | 'amber' | 'red'; dot?: boolean }) {
  return <span className={`badge badge-${tone}`}>{dot && <span className="status-dot" />}{children}</span>;
}
export function Panel({ children, title, subtitle, action, className = '', id }: { children: ReactNode; title?: ReactNode; subtitle?: ReactNode; action?: ReactNode; className?: string; id?: string }) {
  return <section id={id} className={`panel ${className}`}>{(title || action) && <header className="panel-header"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</header>}{children}</section>;
}
export function PageHeader({ eyebrow, title, description, actions }: { eyebrow?: string; title: string; description?: string; actions?: ReactNode }) {
  return <div className="page-header"><div>{eyebrow && <div className="eyebrow">{eyebrow}</div>}<h1>{title}</h1>{description && <p>{description}</p>}</div>{actions && <div className="page-actions">{actions}</div>}</div>;
}
export function EmptyState({ title, description, icon, action }: { title: string; description: string; icon?: ReactNode; action?: ReactNode }) {
  return <div className="empty-state">{icon && <div className="empty-icon">{icon}</div>}<h3>{title}</h3><p>{description}</p>{action}</div>;
}
export function MetricCard({ label, value, detail, color = 'cyan', chart, icon }: { label: string; value: string | number; detail?: string; color?: string; chart?: number[]; icon?: ReactNode }) {
  const id = useId().replace(/:/g, '');
  const data = chart?.map((value, index) => ({ value, index }));
  return <div className={`metric-card metric-${color}`}><div className="metric-label">{label}{icon}</div><div className="metric-value">{value}</div>{detail && <div className="metric-detail">{detail}</div>}{data && data.length > 1 && <div className="metric-sparkline"><ResponsiveContainer width="100%" height="100%"><AreaChart data={data}><defs><linearGradient id={id} x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor={`var(--${color})`} stopOpacity={0.3} /><stop offset="100%" stopColor={`var(--${color})`} stopOpacity={0} /></linearGradient></defs><Area type="monotone" dataKey="value" stroke={`var(--${color})`} fill={`url(#${id})`} strokeWidth={1.6} isAnimationActive={false} /></AreaChart></ResponsiveContainer></div>}</div>;
}
export function MetricsChart({ metrics, height = 230, showAccuracy = true, lossOnly = false }: { metrics: MetricPoint[]; height?: number; showAccuracy?: boolean; lossOnly?: boolean }) {
  if (metrics.length < 2) return <div className="chart-empty" style={{ height }}><div className="chart-empty-grid" /><span>Start training to capture real metrics</span></div>;
  return <div className="metrics-chart" style={{ height }}><ResponsiveContainer width="100%" height="100%"><LineChart data={metrics} margin={{ top: 10, right: 12, left: -20, bottom: 0 }}><CartesianGrid stroke="#203047" strokeDasharray="3 5" vertical={false} /><XAxis dataKey="step" tick={{ fill: '#8393ad', fontSize: 10 }} stroke="#273449" tickLine={false} /><YAxis yAxisId="loss" tick={{ fill: '#8393ad', fontSize: 10 }} stroke="transparent" tickLine={false} /><Tooltip contentStyle={{ background: '#101c2d', border: '1px solid #2b3e56', borderRadius: 8, color: '#e8edf5' }} labelFormatter={label => `Step ${label}`} formatter={(value: number) => value.toFixed(4)} /><Line yAxisId="loss" dataKey="loss" stroke="#51c8ff" strokeWidth={2} dot={false} isAnimationActive={false} />{showAccuracy && !lossOnly && <><YAxis yAxisId="accuracy" orientation="right" domain={[0, 1]} hide /><Line yAxisId="accuracy" dataKey="accuracy" stroke="#a68bff" strokeWidth={2} dot={false} isAnimationActive={false} /></>}</LineChart></ResponsiveContainer></div>;
}
export function Field({ label, children, hint, className = '' }: { label: string; children: ReactNode; hint?: string; className?: string }) {
  return <label className={`field ${className}`}><span className="field-label">{label}</span>{children}{hint && <span className="field-hint">{hint}</span>}</label>;
}
export function Select({ value, onChange, options, label, disabled = false }: { value: string; onChange: (value: string) => void; options: { value: string; label: string }[]; label: string; disabled?: boolean }) {
  return <div className="select-wrap"><select aria-label={label} value={value} disabled={disabled} onChange={event => onChange(event.target.value)}>{options.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}</select><ChevronDown size={13} /></div>;
}
export function Toggle({ checked, onChange, label, detail }: { checked: boolean; onChange: (value: boolean) => void; label: string; detail?: string }) {
  return <label className="toggle-row"><span><span>{label}</span>{detail && <small>{detail}</small>}</span><input type="checkbox" checked={checked} onChange={event => onChange(event.target.checked)} /><span className="toggle-track" aria-hidden="true"><span /></span></label>;
}
export function Callout({ children, tone = 'info' }: { children: ReactNode; tone?: 'info' | 'warning' | 'success' }) {
  return <div className={`callout callout-${tone}`}><Info size={16} /><div>{children}</div></div>;
}
export function KeyValue({ label, value, mono = false }: { label: string; value: ReactNode; mono?: boolean }) {
  return <div className="key-value"><span>{label}</span><strong className={mono ? 'mono' : ''}>{value}</strong></div>;
}
export function Progress({ value, color = 'cyan' }: { value: number; color?: string }) {
  return <div className="progress-track"><div style={{ width: `${Math.max(0, Math.min(100, value))}%`, background: `var(--${color})` }} /></div>;
}
export function ExternalLink({ href, children }: { href: string; children: ReactNode }) {
  return <a href={href} target="_blank" rel="noreferrer" className="external-link">{children}<ArrowUpRight size={14} /></a>;
}
export const number = (value: number, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '—';
export const percentage = (value: number) => `${(value * 100).toFixed(1)}%`;
export const duration = (milliseconds: number) => {
  const seconds = Math.floor(milliseconds / 1000);
  return `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
};
