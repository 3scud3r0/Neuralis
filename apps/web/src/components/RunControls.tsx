import { Play, Pause, Square, SkipForward, RotateCcw, Save, Download } from 'lucide-react';
import { useStudio } from '../app/StudioContext';
import { Button, Badge } from './ui';

export function RunControls({ compact = false, checkpoints = false }: { compact?: boolean; checkpoints?: boolean }) {
  const { snapshot, start, pause, step, stop, resetEngine, saveCheckpoint, exportRun } = useStudio();
  const running = snapshot.state === 'running';
  return <div className={`run-controls ${compact ? 'compact' : ''}`}>
    <Button variant="primary" onClick={running ? pause : start} aria-label={running ? 'Pause training' : snapshot.state === 'paused' ? 'Resume training' : 'Start training'}>{running ? <Pause size={15} /> : <Play size={15} />}{running ? 'Pause' : snapshot.state === 'paused' ? 'Resume' : 'Start training'}</Button>
    <Button onClick={step} disabled={running} title="Train a single optimizer step"><SkipForward size={15} />{!compact && 'Step'}</Button>
    <Button onClick={stop} disabled={snapshot.state === 'idle' || ['completed', 'stopped', 'failed'].includes(snapshot.state)} title="Stop training"><Square size={13} />{!compact && 'Stop'}</Button>
    <Button variant="ghost" onClick={resetEngine} title="Reset model to seeded initialization"><RotateCcw size={15} /></Button>
    {checkpoints && <><Button onClick={() => saveCheckpoint()}><Save size={15} />Checkpoint</Button><Button variant="ghost" onClick={exportRun} title="Export run"><Download size={15} /></Button></>}
  </div>;
}
export function RunStatus() {
  const { snapshot } = useStudio();
  const tone = snapshot.state === 'running' || snapshot.state === 'completed' ? 'green' : snapshot.state === 'failed' ? 'red' : snapshot.state === 'paused' ? 'amber' : 'neutral';
  return <Badge tone={tone} dot>{snapshot.state === 'idle' ? 'Ready to train' : snapshot.state}</Badge>;
}
