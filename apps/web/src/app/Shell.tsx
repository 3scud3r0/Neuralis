import { useEffect, useRef, useState, type ReactNode } from 'react';
import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import { Search, Command, ChevronRight, ArrowUpRight, PanelLeftClose, Menu, X, Cpu, CircleHelp, Sparkles, CheckCircle2, AlertTriangle } from 'lucide-react';
import { navigation } from './navigation';
import { useStudio } from './StudioContext';
import { DEMOS } from '../engine';
import { Badge, Button } from '../components/ui';
import { RunStatus } from '../components/RunControls';

function Logo() {
  return <div className="brand"><svg className="brand-symbol" viewBox="0 0 40 40" fill="none" aria-hidden="true"><path d="M20 3 35 12v17L20 38 5 29V12L20 3Z" stroke="currentColor" strokeWidth="1.5" /><path d="m20 3 0 35M5 12l30 17M35 12 5 29M5 12h30M5 29h30" stroke="currentColor" strokeWidth=".7" opacity=".6" />{[[20,3],[35,12],[35,29],[20,38],[5,29],[5,12],[20,20]].map(([x,y],i) => <circle key={i} cx={x} cy={y} r="2.4" fill="currentColor" />)}</svg><div><strong>Neuralis<span>Studio</span></strong><small>Research, made visible.</small></div></div>;
}
function CommandPalette({ onClose }: { onClose: () => void }) {
  const [query, setQuery] = useState('');
  const [activeIndex, setActiveIndex] = useState(0);
  const dialog = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const { selectDemo } = useStudio();
  const pages = navigation.filter(item => `${item.label} ${item.description}`.toLowerCase().includes(query.toLowerCase()));
  const demos = query ? DEMOS.filter(item => `${item.title} ${item.tags.join(' ')}`.toLowerCase().includes(query.toLowerCase())).slice(0, 5) : [];
  useEffect(() => { input.current?.focus(); }, []);
  const resultCount = pages.length + demos.length;
  const choose = (index: number) => {
    if (index < pages.length) navigate(pages[index].path);
    else if (demos[index - pages.length]) { const demo = demos[index - pages.length]; if (demo.id === 'super-mario') navigate('/demos'); else if (demo.id === 'custom-data') navigate('/datasets'); else { selectDemo(demo.id); navigate('/training'); } }
    onClose();
  };
  return <div className="modal-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}><div role="dialog" aria-modal="true" aria-label="Search workspace" className="command-palette" ref={dialog} onKeyDown={event => {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); setActiveIndex(index => resultCount ? (index + (event.key === 'ArrowDown' ? 1 : -1) + resultCount) % resultCount : 0); }
    if (event.key === 'Enter' && event.target === input.current && resultCount) { event.preventDefault(); choose(Math.min(activeIndex, resultCount - 1)); }
    if (event.key === 'Tab') { const focusable = dialog.current?.querySelectorAll<HTMLElement>('input,button'); if (focusable?.length) { const first = focusable[0]; const last = focusable[focusable.length - 1]; if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); } else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); } } }
  }}><div className="command-input"><Search size={19} /><input ref={input} value={query} onChange={event => { setQuery(event.target.value); setActiveIndex(0); }} role="combobox" aria-expanded="true" aria-controls="command-results" aria-activedescendant={resultCount ? `command-result-${Math.min(activeIndex, resultCount - 1)}` : undefined} aria-label="Search workspace pages and demos" placeholder="Search pages and demo models…" onKeyDown={event => { if (event.key === 'Escape') onClose(); }} /><button className="icon-button" onClick={onClose} aria-label="Close search"><X size={17} /></button></div><div className="command-results" id="command-results" role="listbox"><span className="section-label">WORKSPACE</span>{pages.map((item, index) => <button key={item.path} id={`command-result-${index}`} role="option" aria-selected={activeIndex === index} className={activeIndex === index ? 'keyboard-active' : ''} onMouseEnter={() => setActiveIndex(index)} onClick={() => choose(index)}><item.icon size={18} /><span><strong>{item.label}</strong><small>{item.description}</small></span><ChevronRight size={14} /></button>)}{demos.length > 0 && <span className="section-label">RUNNABLE DEMOS</span>}{demos.map((demo, index) => <button key={demo.id} id={`command-result-${pages.length + index}`} role="option" aria-selected={activeIndex === pages.length + index} className={activeIndex === pages.length + index ? 'keyboard-active' : ''} onMouseEnter={() => setActiveIndex(pages.length + index)} onClick={() => choose(pages.length + index)}><Sparkles size={18} /><span><strong>{demo.title}</strong><small>{demo.category}</small></span><ChevronRight size={14} /></button>)}{pages.length === 0 && demos.length === 0 && <p className="muted command-no-results">No matching pages or demos.</p>}</div><div className="command-footer"><kbd>esc</kbd> to close <span>All computation stays on your device</span></div></div></div>;
}

export function Shell({ children }: { children: ReactNode }) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();
  const current = navigation.find(item => item.path === location.pathname) ?? navigation[0];
  const { snapshot, notifications, dismiss } = useStudio();
  const demo = DEMOS.find(item => item.id === snapshot.demoId);
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
        event.preventDefault();
        setSearchOpen(value => !value);
      }
      if (event.key === 'Escape') { setSearchOpen(false); setMobileOpen(false); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);
  useEffect(() => { setMobileOpen(false); }, [location.pathname]);
  return <div className={`studio-shell ${collapsed ? 'sidebar-collapsed' : ''}`}>
    <a className="skip-link" href="#main-content">Skip to content</a>
    {mobileOpen && <button className="sidebar-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
    <aside className={`sidebar ${mobileOpen ? 'mobile-open' : ''}`}><NavLink to="/" className="brand-link" aria-label="Neuralis Studio home"><Logo /></NavLink><div className="workspace-switch"><span className="workspace-avatar">N</span><span><strong>Local workspace</strong><small>Personal research lab</small></span><Badge tone="cyan">LOCAL</Badge></div><nav aria-label="Main navigation">{['Workspace', 'Research', 'Laboratory'].map(group => <div className="nav-group" key={group}><span className="nav-group-label">{group}</span>{navigation.filter(item => item.group === group).map(item => <NavLink key={item.path} to={item.path} end={item.path === '/'} title={collapsed ? item.label : undefined} className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}><item.icon size={17} /><span>{item.label}</span>{item.path === '/demos' && <em>{DEMOS.length}</em>}</NavLink>)}</div>)}</nav><div className="sidebar-bottom"><div className="local-status"><span className="status-dot" /><span>Browser engine ready</span><Cpu size={14} /></div><NavLink to="/lab" className="backend-link">Connect a research backend<ArrowUpRight size={13} /></NavLink><div className="user-row"><span className="user-avatar">R</span><div><strong>Researcher</strong><small>Local session</small></div><button className="icon-button collapse-button" onClick={() => setCollapsed(value => !value)} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}><PanelLeftClose size={16} /></button></div></div></aside>
    <div className="studio-content"><header className="topbar"><div className="breadcrumb"><button className="icon-button mobile-menu" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu size={19} /></button><span>Workspace</span><ChevronRight size={12} /><strong>{current.label}</strong></div><button className="search-trigger" onClick={() => setSearchOpen(true)}><Search size={14} /><span>Search workspace…</span><kbd><Command size={10} /> K</kbd></button><div className="topbar-right"><Badge tone="cyan"><Cpu size={11} />CPU / Browser</Badge><button className="icon-button help-button" aria-label="Open documentation" onClick={() => window.open('https://github.com/3scud3r0/Neuralis#readme', '_blank', 'noopener')}><CircleHelp size={17} /></button><span className="topbar-avatar">R</span></div></header>
    <div className="active-run-strip"><div><span className="run-indicator" /><span className="strip-label">ACTIVE MODEL</span><strong>{demo?.title ?? snapshot.demoId}</strong><span className="strip-architecture">{demo?.architecture.join(' → ')}</span></div><div><span className="strip-step">Step {snapshot.step.toLocaleString()}</span><RunStatus /></div></div>
    <main id="main-content" className="page-content">{children}</main><footer className="app-footer"><span><span className="status-dot" /> Local computation · Your data stays on this device</span><span>Neuralis Studio <span className="mono">v0.1</span></span></footer></div>
    {searchOpen && <CommandPalette onClose={() => setSearchOpen(false)} />}
    <div className="toast-stack" role="status" aria-live="polite">{notifications.map(item => <div className={`toast toast-${item.tone}`} key={item.id}>{item.tone === 'error' ? <AlertTriangle size={18} /> : <CheckCircle2 size={18} />}<span>{item.message}</span><button className="icon-button" aria-label="Dismiss notification" onClick={() => dismiss(item.id)}><X size={14} /></button></div>)}</div>
  </div>;
}
