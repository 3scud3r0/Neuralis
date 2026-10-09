use neuralis_desktop::{
    config::{BackendMode, Preferences},
    runtime::Runtime,
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let python = PathBuf::from(
        arguments
            .next()
            .ok_or("Usage: runtime_smoke PYTHON REPOSITORY_ROOT")?,
    );
    let root = PathBuf::from(arguments.next().ok_or("Repository path is required")?);
    let directory = std::env::temp_dir().join(format!(
        "neuralis-desktop-smoke-{}",
        neuralis_desktop::security::session_token()
    ));
    let preferences = Preferences {
        mode: BackendMode::Local,
        python_executable: python,
        repository_root: Some(root),
        data_directory: Some(directory.clone()),
        ..Preferences::default()
    };
    let mut runtime = Runtime::new()?;
    let outcome = (|| -> Result<(), Box<dyn std::error::Error>> {
        let status = runtime.start(&preferences, None)?;
        assert!(status.managed);
        assert!(status
            .endpoint
            .as_ref()
            .is_some_and(|value| value.starts_with("http://127.0.0.1:")));
        let capabilities = runtime.request("GET", "/api/v1/capabilities", None)?;
        assert!(capabilities.is_object());
        println!("Local Python startup, authenticated readiness and native API request succeeded");
        runtime.stop()?;
        assert!(!runtime.status().managed);
        println!("Managed Python shutdown succeeded");
        Ok(())
    })();
    let _ = runtime.stop();
    if directory.exists() {
        std::fs::remove_dir_all(directory)?;
    }
    outcome
}
