#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use neuralis_desktop::{
    config::Preferences,
    files::{FileGrants, FileReceipt},
    runtime::{Runtime, RuntimeStatus},
    workspace::{ArtifactKind, ArtifactManifest, ProjectManifest, Workspace},
};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tauri::{Manager, State};
use tauri_plugin_dialog::DialogExt;

struct AppState {
    runtime: Arc<Mutex<Runtime>>,
    grants: Mutex<FileGrants>,
    preferences: Mutex<Preferences>,
    preferences_path: PathBuf,
    workspace: Mutex<Option<Workspace>>,
}

fn message<T, E: std::fmt::Display>(value: Result<T, E>) -> Result<T, String> {
    value.map_err(|error| error.to_string())
}

#[tauri::command]
fn get_preferences(state: State<AppState>) -> Result<Preferences, String> {
    Ok(message(state.preferences.lock())?.clone())
}

#[tauri::command]
fn save_preferences(preferences: Preferences, state: State<AppState>) -> Result<(), String> {
    message(preferences.save(&state.preferences_path))?;
    *message(state.preferences.lock())? = preferences;
    Ok(())
}

#[tauri::command]
async fn start_backend(
    token: Option<String>,
    state: State<'_, AppState>,
) -> Result<RuntimeStatus, String> {
    // The selected mode and executable come from native preferences, never a server response.
    let mut preferences = message(state.preferences.lock())?.clone();
    if preferences.data_directory.is_none() {
        preferences.data_directory = Some(
            state
                .preferences_path
                .parent()
                .ok_or("Configuration directory is unavailable")?
                .join("data"),
        );
    }
    let runtime = state.runtime.clone();
    tauri::async_runtime::spawn_blocking(move || {
        message(message(runtime.lock())?.start(&preferences, token))
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
fn stop_backend(state: State<AppState>) -> Result<(), String> {
    message(message(state.runtime.lock())?.stop())
}

#[tauri::command]
fn backend_status(state: State<AppState>) -> Result<RuntimeStatus, String> {
    Ok(message(state.runtime.lock())?.status())
}

#[tauri::command]
fn backend_diagnostics(state: State<AppState>) -> Result<Vec<String>, String> {
    Ok(message(state.runtime.lock())?.diagnostics())
}

#[tauri::command]
async fn backend_request(
    method: String,
    path: String,
    body: Option<serde_json::Value>,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let runtime = state.runtime.clone();
    tauri::async_runtime::spawn_blocking(move || {
        message(message(runtime.lock())?.request(&method, &path, body))
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
async fn choose_export_directory(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<Option<String>, String> {
    let selected = app
        .dialog()
        .file()
        .set_title("Choose a Neuralis export folder")
        .blocking_pick_folder();
    let Some(selected) = selected else {
        return Ok(None);
    };
    let path = message(selected.into_path())?;
    let canonical = message(message(state.grants.lock())?.grant_export_directory(&path))?;
    Ok(Some(canonical.to_string_lossy().into_owned()))
}

#[tauri::command]
async fn choose_checkpoint(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<Option<String>, String> {
    let selected = app
        .dialog()
        .file()
        .set_title("Import a Neuralis JSON checkpoint")
        .add_filter("JSON checkpoint", &["json"])
        .blocking_pick_file();
    let Some(selected) = selected else {
        return Ok(None);
    };
    let path = message(selected.into_path())?;
    let canonical = message(message(state.grants.lock())?.grant_import_file(&path))?;
    Ok(Some(canonical.to_string_lossy().into_owned()))
}

#[tauri::command]
fn export_checkpoint(
    directory: String,
    filename: String,
    content: String,
    state: State<AppState>,
) -> Result<FileReceipt, String> {
    message(message(state.grants.lock())?.export_json(
        &PathBuf::from(directory),
        &filename,
        &content,
    ))
}

#[tauri::command]
fn import_checkpoint(
    path: String,
    state: State<AppState>,
) -> Result<(String, FileReceipt), String> {
    message(message(state.grants.lock())?.import_json(&PathBuf::from(path)))
}

#[tauri::command]
async fn choose_workspace(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<Option<String>, String> {
    let selected = app
        .dialog()
        .file()
        .set_title("Choose a Neuralis project workspace")
        .blocking_pick_folder();
    let Some(selected) = selected else {
        return Ok(None);
    };
    let path = message(selected.into_path())?;
    let workspace = message(Workspace::open(&path, 10 * 1024 * 1024 * 1024))?;
    let root = workspace.root().to_string_lossy().into_owned();
    *message(state.workspace.lock())? = Some(workspace);
    Ok(Some(root))
}

#[tauri::command]
fn list_projects(state: State<AppState>) -> Result<Vec<ProjectManifest>, String> {
    let guard = message(state.workspace.lock())?;
    let workspace = guard.as_ref().ok_or("Choose a workspace folder first")?;
    message(workspace.list())
}

#[tauri::command]
fn create_project(
    name: String,
    description: String,
    tags: Vec<String>,
    state: State<AppState>,
) -> Result<ProjectManifest, String> {
    let guard = message(state.workspace.lock())?;
    let workspace = guard.as_ref().ok_or("Choose a workspace folder first")?;
    message(workspace.create(name, description, tags))
}

#[tauri::command]
fn update_project(
    id: String,
    name: String,
    description: String,
    tags: Vec<String>,
    state: State<AppState>,
) -> Result<ProjectManifest, String> {
    let guard = message(state.workspace.lock())?;
    let workspace = guard.as_ref().ok_or("Choose a workspace folder first")?;
    message(workspace.update_metadata(&id, name, description, tags))
}

#[tauri::command]
fn add_project_artifact(
    id: String,
    filename: String,
    kind: ArtifactKind,
    content: String,
    state: State<AppState>,
) -> Result<ArtifactManifest, String> {
    let guard = message(state.workspace.lock())?;
    let workspace = guard.as_ref().ok_or("Choose a workspace folder first")?;
    message(workspace.add_artifact(&id, &filename, kind, &content))
}

#[tauri::command]
fn read_project_artifact(
    id: String,
    filename: String,
    state: State<AppState>,
) -> Result<String, String> {
    let guard = message(state.workspace.lock())?;
    let workspace = guard.as_ref().ok_or("Choose a workspace folder first")?;
    message(workspace.read_artifact(&id, &filename))
}

#[tauri::command]
fn export_project_bundle(
    id: String,
    directory: String,
    filename: String,
    state: State<AppState>,
) -> Result<FileReceipt, String> {
    let guard = message(state.workspace.lock())?;
    let workspace = guard.as_ref().ok_or("Choose a workspace folder first")?;
    let content = message(workspace.bundle(&id))?;
    message(message(state.grants.lock())?.export_json(
        &PathBuf::from(directory),
        &filename,
        &content,
    ))
}

#[tauri::command]
fn import_project_bundle(path: String, state: State<AppState>) -> Result<ProjectManifest, String> {
    let (text, _) = message(message(state.grants.lock())?.import_json(&PathBuf::from(path)))?;
    let guard = message(state.workspace.lock())?;
    let workspace = guard.as_ref().ok_or("Choose a workspace folder first")?;
    message(workspace.import_bundle(&text))
}

#[tauri::command]
fn workspace_usage(state: State<AppState>) -> Result<u64, String> {
    let guard = message(state.workspace.lock())?;
    let workspace = guard.as_ref().ok_or("Choose a workspace folder first")?;
    message(workspace.usage())
}

#[tauri::command]
async fn probe_python_runtime(
    state: State<'_, AppState>,
) -> Result<neuralis_desktop::diagnostics::RuntimeReadiness, String> {
    let executable = message(state.preferences.lock())?.python_executable.clone();
    tauri::async_runtime::spawn_blocking(move || {
        message(neuralis_desktop::diagnostics::probe_python(
            &executable,
            std::time::Duration::from_secs(15),
        ))
    })
    .await
    .map_err(|error| error.to_string())?
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let path = app.path().app_config_dir()?.join("preferences.json");
            let preferences = Preferences::load(&path).unwrap_or_default();
            app.manage(AppState {
                runtime: Arc::new(Mutex::new(Runtime::new()?)),
                grants: Mutex::new(FileGrants::default()),
                preferences: Mutex::new(preferences),
                preferences_path: path,
                workspace: Mutex::new(None),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_preferences,
            save_preferences,
            start_backend,
            stop_backend,
            backend_status,
            backend_diagnostics,
            backend_request,
            choose_export_directory,
            choose_checkpoint,
            export_checkpoint,
            import_checkpoint,
            choose_workspace,
            list_projects,
            create_project,
            update_project,
            add_project_artifact,
            read_project_artifact,
            export_project_bundle,
            import_project_bundle,
            workspace_usage,
            probe_python_runtime
        ])
        .build(tauri::generate_context!())
        .expect("Could not initialize Neuralis Studio")
        .run(|app, event| {
            if matches!(event, tauri::RunEvent::Exit) {
                if let Some(state) = app.try_state::<AppState>() {
                    if let Ok(mut runtime) = state.runtime.lock() {
                        let _ = runtime.stop();
                    }
                }
            }
        });
}
