import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, ArrowUpRight, ArrowRight, Play, Code2, Cpu, Gamepad2, Binary, Workflow, BookOpen, Filter } from 'lucide-react';
import { DEMOS } from '../engine';
import { useStudio } from '../app/StudioContext';
import { PageHeader, Badge, Button, ExternalLink, Callout, EmptyState } from '../components/ui';

function DemoArt({ architecture, accent, id }: { architecture: number[]; accent: string; id: string }) {
  const sizes = architecture.slice(0, 6).map(value => Math.min(7, value));
  const width = 320;
  const height = 120;
  const x = (layer: number) => 35 + layer / Math.max(1, sizes.length - 1) * 250;
  const y = (layer: number, node: number) => height / 2 + (node - (sizes[layer] - 1) / 2) * 15;
  return <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${architecture.join(' to ')} network architecture`}><defs><radialGradient id={`glow-${id}`}><stop offset="0%" stopColor={accent} stopOpacity=".22" /><stop offset="100%" stopColor={accent} stopOpacity="0" /></radialGradient></defs><ellipse cx="160" cy="60" rx="150" ry="70" fill={`url(#glow-${id})`} />{sizes.slice(1).flatMap((count, layer) => Array.from({ length: sizes[layer] }, (_, a) => Array.from({ length: count }, (_, b) => <line key={`${layer}-${a}-${b}`} x1={x(layer)} y1={y(layer, a)} x2={x(layer + 1)} y2={y(layer + 1, b)} stroke={accent} strokeWidth=".6" opacity=".2" />))).flat()}{sizes.flatMap((count, layer) => Array.from({ length: count }, (_, node) => <g key={`${layer}-${node}`}><circle cx={x(layer)} cy={y(layer, node)} r="5" fill={accent} opacity=".12" /><circle cx={x(layer)} cy={y(layer, node)} r="2.7" fill={accent} opacity={layer === 0 || layer === sizes.length - 1 ? 1 : .7} /></g>))}</svg>;
}
export function DemoLibrary() {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('All experiments');
  const { selectDemo, snapshot } = useStudio();
  const navigate = useNavigate();
  const browserDemos = DEMOS.filter(demo => demo.id !== 'super-mario');
  const categories = ['All experiments', ...new Set(browserDemos.map(demo => demo.category))];
  const demos = useMemo(() => DEMOS.filter(demo => demo.id !== 'super-mario' && (category === 'All experiments' || category === demo.category) && `${demo.title} ${demo.description} ${demo.tags.join(' ')}`.toLowerCase().includes(query.toLowerCase())), [query, category]);
  const launch = (id: string) => { if (id === 'custom-data') { navigate('/datasets'); return; } selectDemo(id); navigate('/training'); };
  return <>
    <PageHeader eyebrow="THE DEMO LIBRARY" title="Small models. Big discoveries." description="Real, reproducible experiments you can run directly in your browser." actions={<Badge tone="cyan"><Cpu size={12} />{browserDemos.length} browser experiments</Badge>} />
    <div className="library-intro"><div><span className="eyebrow">LEARN BY EXPERIMENTING</span><h2>Choose a question.<br /><span>Train an answer.</span></h2><p>Each demo connects a learning task with live metrics, inspectable tensors, and a spatial model view.</p></div><div className="library-path"><span><Binary size={22} /><strong>Choose a dataset</strong><small>Seeded, generated examples</small></span><ArrowRight size={19} /><span><Workflow size={22} /><strong>Watch it learn</strong><small>Real optimizer steps</small></span><ArrowRight size={19} /><span><BookOpen size={22} /><strong>Explain the result</strong><small>Inspect and intervene</small></span></div></div>
    <div className="library-toolbar"><div className="filter-tabs" role="group" aria-label="Filter demos by category">{categories.map(value => <button key={value} className={value === category ? 'active' : ''} onClick={() => setCategory(value)}>{value}</button>)}</div><div className="library-search"><Search size={15} /><input aria-label="Search demos" value={query} onChange={event => setQuery(event.target.value)} placeholder="Search experiments…" /></div></div>
    <div className="demo-grid">{demos.map((demo, index) => <article className="demo-card" key={demo.id} style={{ '--demo-accent': demo.accent } as React.CSSProperties}><div className="demo-art"><DemoArt architecture={demo.architecture} accent={demo.accent} id={demo.id} /><span className="demo-number">{(index + 1).toString().padStart(2, '0')}</span>{snapshot.demoId === demo.id && <Badge tone="green" dot>Active model</Badge>}</div><div className="demo-content"><div className="demo-meta"><Badge>{demo.category}</Badge><span>{demo.difficulty}</span></div><h3>{demo.title}</h3><p>{demo.description}</p><div className="demo-tags">{demo.tags.slice(0, 3).map(tag => <span key={tag}>{tag}</span>)}</div><div className="demo-footer"><span className="mono">{demo.architecture.join(' → ')}</span><Button onClick={() => launch(demo.id)}><Play size={13} />Launch</Button></div></div></article>)}</div>
    {demos.length === 0 && <EmptyState title="No matching experiments" description="Try another category or search term." icon={<Filter size={25} />} action={<Button onClick={() => { setQuery(''); setCategory('All experiments'); }}>Clear filters</Button>} />}
    <div className="mario-banner"><div className="mario-pixel-art" aria-hidden="true"><Gamepad2 size={46} /></div><div><Badge tone="amber">LOCAL PYTHON DEMO</Badge><h3>Super Mario reinforcement learning</h3><p>Train and evaluate an agent using a NES environment on your own machine. Includes checkpointing, video capture, and measured evaluation.</p></div><div><ExternalLink href="https://github.com/3scud3r0/Neuralis/tree/main/demos/super-mario"><Code2 size={15} />Open Mario code</ExternalLink><ExternalLink href={`${import.meta.env.BASE_URL}demos/super-mario/index.html`}>Open training report</ExternalLink><small>Requires Python + legal game assets</small></div></div>
    <Callout>Browser experiments are intentionally bounded to small CPU workloads. Connect the optional Python backend or desktop client for your own larger models. The Mario trainer runs locally; it is not embedded into GitHub Pages.</Callout>
  </>;
}
