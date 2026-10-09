/** Browser-safe bridge: use global Tauri IPC only inside the installed app. */
export interface NativePreferences {
  schema_version: number;
  mode: 'browser_only' | 'local' | 'remote';
  python_executable: string;
  repository_root: string | null;
  remote_endpoint: string | null;
  data_directory?: string | null;
  startup_timeout_seconds: number;
  capture_budget_bytes: number;
}
export interface NativeStatus {
  phase: 'browser_only' | 'starting' | 'ready' | 'failed' | 'stopped';
  endpoint: string | null;
  managed: boolean;
  process_id: number | null;
  message: string;
}
export interface NativeArtifact {
  filename: string; kind: 'checkpoint' | 'experiment' | 'metrics' | 'dataset' | 'capture';
  bytes: number; sha256: string; created_at_ms: number;
}
export interface NativeProject {
  schema_version: number; id: string; name: string; description: string;
  created_at_ms: number; updated_at_ms: number; tags: string[]; artifacts: NativeArtifact[];
}
export interface FileReceipt { path: string; bytes: number; sha256: string }
interface TauriGlobal { core: { invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> } }
function host(): TauriGlobal | undefined {
  return (globalThis as typeof globalThis & { __TAURI__?: TauriGlobal }).__TAURI__;
}
export function isDesktop(): boolean { return Boolean(host()); }
async function invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  const native = host();
  if (!native) throw new Error('This operation requires the Neuralis desktop application.');
  return native.core.invoke<T>(command, args);
}
export const desktop = {
  preferences: () => invoke<NativePreferences>('get_preferences'),
  savePreferences: (preferences: NativePreferences) => invoke<void>('save_preferences', { preferences }),
  start: (token?: string) => invoke<NativeStatus>('start_backend', { token: token ?? null }),
  stop: () => invoke<void>('stop_backend'),
  status: () => invoke<NativeStatus>('backend_status'),
  probePython: () => invoke<{ executable: string; ready: boolean; missing: string[]; recommendations: string[]; runtime: { python: string; platform: string; cuda_available: boolean; cuda_device_count: number; packages: Record<string, { available: boolean; version: string | null }> } }>("probe_python_runtime"),
  diagnostics: () => invoke<string[]>('backend_diagnostics'),
  request: <T>(method: 'GET' | 'POST' | 'PUT' | 'DELETE', path: string, body?: unknown) => invoke<T>('backend_request', { method, path, body: body ?? null }),
  selectExportFolder: () => invoke<string | null>('choose_export_directory'),
  selectCheckpoint: () => invoke<string | null>('choose_checkpoint'),
  exportCheckpoint: (directory: string, filename: string, content: string) => invoke<FileReceipt>('export_checkpoint', { directory, filename, content }),
  selectWorkspace: () => invoke<string | null>('choose_workspace'),
  listProjects: () => invoke<NativeProject[]>('list_projects'),
  createProject: (name: string, description: string, tags: string[]) => invoke<NativeProject>('create_project', { name, description, tags }),
  updateProject: (id: string, name: string, description: string, tags: string[]) => invoke<NativeProject>('update_project', { id, name, description, tags }),
  addArtifact: (id: string, filename: string, kind: NativeArtifact['kind'], content: string) => invoke<NativeArtifact>('add_project_artifact', { id, filename, kind, content }),
  readArtifact: (id: string, filename: string) => invoke<string>('read_project_artifact', { id, filename }),
  exportProject: (id: string, directory: string, filename: string) => invoke<FileReceipt>('export_project_bundle', { id, directory, filename }),
  importProject: (path: string) => invoke<NativeProject>('import_project_bundle', { path }),
  workspaceUsage: () => invoke<number>('workspace_usage'),
  importCheckpoint: (path: string) => invoke<[string, FileReceipt]>('import_checkpoint', { path }),
};
