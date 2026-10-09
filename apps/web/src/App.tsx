import { Component, type ErrorInfo, type ReactNode } from 'react';
import { HashRouter, Route, Routes } from 'react-router-dom';
import { StudioProvider } from './app/StudioContext';
import { Shell } from './app/Shell';
import { Overview } from './features/Overview';
import { NetworkExplorer } from './features/NetworkExplorer';
import { Training } from './features/Training';
import { Tensors, Attention, DataExplorer } from './features/Inspectors';
import { Interventions } from './features/Interventions';
import { DemoLibrary } from './features/DemoLibrary';
import { Experiments, CompareRuns } from './features/Experiments';
import { ExperimentWizard } from './features/ExperimentWizard';
import { Artifacts } from './features/Artifacts';
import { Readiness } from './features/Readiness';
import { RepresentationExplorer } from './features/RepresentationExplorer';
import { DatasetWorkbench } from './research/DatasetWorkbench';
import { Laboratory, Settings } from './features/Workspace';
import { AdvancedResearch } from './features/AdvancedResearch';
import { ResearchWorkbench } from './features/ResearchWorkbench';
import { InferenceStream } from './features/InferenceStream';
import { ComputationalGraphCanvas } from './features/GraphEditor';

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('Neuralis rendering error', error, info.componentStack); }
  render() {
    if (this.state.error) return <div className="error-screen"><h1>The workspace could not render.</h1><p>{this.state.error.message}</p><p>Your saved checkpoints remain in local browser storage.</p><button className="button button-primary" onClick={() => window.location.reload()}>Reload workspace</button></div>;
    return this.props.children;
  }
}
export default function App() {
  return <ErrorBoundary><HashRouter><StudioProvider><Shell><Routes>
    <Route path="/" element={<Overview />} />
    <Route path="/visualization" element={<NetworkExplorer />} />
    <Route path="/training" element={<Training />} />
    <Route path="/tensors" element={<Tensors />} />
    <Route path="/attention" element={<Attention />} />
    <Route path="/interventions" element={<Interventions />} />
    <Route path="/experiments" element={<Experiments />} />
    <Route path="/compare" element={<CompareRuns />} />
    <Route path="/data" element={<DataExplorer />} />
    <Route path="/demos" element={<DemoLibrary />} />
    <Route path="/lab" element={<Laboratory />} />
    <Route path="/new-experiment" element={<ExperimentWizard />} />
    <Route path="/artifacts" element={<Artifacts />} />
    <Route path="/readiness" element={<Readiness />} />
    <Route path="/representations" element={<RepresentationExplorer />} />
    <Route path="/datasets" element={<DatasetWorkbench />} />
    <Route path="/settings" element={<Settings />} />
    <Route path="/research" element={<ResearchWorkbench />} />
    <Route path="/advanced" element={<AdvancedResearch />} />
    <Route path="/inference" element={<InferenceStream />} />
    <Route path="/graph" element={<ComputationalGraphCanvas />} />
    <Route path="*" element={<Overview />} />
  </Routes></Shell></StudioProvider></HashRouter></ErrorBoundary>;
}
