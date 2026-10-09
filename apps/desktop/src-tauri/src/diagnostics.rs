use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    io::Read,
    path::Path,
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

const PROBE: &str = r#"
import json,sys,platform,importlib.util,importlib.metadata
names=['fastapi','uvicorn','numpy','torch','safetensors','psutil']
modules={}
for name in names:
    available=importlib.util.find_spec(name) is not None
    version=None
    if available:
        try: version=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    modules[name]={'available':available,'version':version}
result={'python':platform.python_version(),'major':sys.version_info.major,'minor':sys.version_info.minor,'platform':platform.system(),'architecture':platform.machine(),'packages':modules,'cuda_available':False,'cuda_device_count':0}
if modules['torch']['available']:
    try:
        import torch
        result['cuda_available']=torch.cuda.is_available()
        result['cuda_device_count']=torch.cuda.device_count() if result['cuda_available'] else 0
    except Exception:
        result['packages']['torch']['available']=False
print(json.dumps(result,allow_nan=False))
"#;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackageReadiness {
    pub available: bool,
    pub version: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PythonReadiness {
    pub python: String,
    pub major: u32,
    pub minor: u32,
    pub platform: String,
    pub architecture: String,
    pub packages: HashMap<String, PackageReadiness>,
    pub cuda_available: bool,
    pub cuda_device_count: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeReadiness {
    pub executable: String,
    pub runtime: PythonReadiness,
    pub ready: bool,
    pub missing: Vec<String>,
    pub recommendations: Vec<String>,
}

pub fn probe_python(executable: &Path, timeout: Duration) -> Result<RuntimeReadiness> {
    if executable.as_os_str().is_empty() || timeout.is_zero() || timeout > Duration::from_secs(60) {
        return Err(Error::Config(
            "Python probe requires an executable and a deadline of at most 60 seconds".into(),
        ));
    }
    let mut command = Command::new(executable);
    command
        .args(["-I", "-c", PROBE])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    // The probe never needs credentials or injected Python startup hooks.
    for name in [
        "NEURALIS_TOKEN",
        "NEURALIS_WORKER_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "PYTHONSTARTUP",
        "PYTHONPATH",
    ] {
        command.env_remove(name);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut child = command.spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| Error::Runtime("Python probe output is unavailable".into()))?;
    let output = Arc::new(Mutex::new(Vec::new()));
    let copy = output.clone();
    let reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stdout.take(65537).read_to_end(&mut bytes);
        if let Ok(mut value) = copy.lock() {
            *value = bytes;
        }
        result
    });
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill()?;
            child.wait()?;
            let _ = reader.join();
            return Err(Error::Runtime(
                "Python readiness probe exceeded its deadline".into(),
            ));
        }
        thread::sleep(Duration::from_millis(25));
    };
    reader
        .join()
        .map_err(|_| Error::Runtime("Python probe reader failed".into()))??;
    if !status.success() {
        return Err(Error::Runtime(format!(
            "Python readiness probe exited with {status}"
        )));
    }
    let bytes = output
        .lock()
        .map_err(|_| Error::Runtime("Probe output lock failed".into()))?;
    if bytes.len() > 65536 {
        return Err(Error::Runtime(
            "Python probe exceeded its output budget".into(),
        ));
    }
    let runtime: PythonReadiness = serde_json::from_slice(&bytes)?;
    let mut missing: Vec<String> = [
        "fastapi",
        "uvicorn",
        "numpy",
        "torch",
        "safetensors",
        "psutil",
    ]
    .iter()
    .filter(|name| {
        !runtime
            .packages
            .get(**name)
            .is_some_and(|value| value.available)
    })
    .map(|name| (*name).to_string())
    .collect();
    if runtime.major != 3 || !(12..14).contains(&runtime.minor) {
        missing.push("Python 3.12 or 3.13".into());
    }
    let mut recommendations = Vec::new();
    if !missing.is_empty() {
        recommendations.push(
            "Install the API's frozen uv lockfile and select its virtual-environment interpreter"
                .into(),
        );
    }
    if !runtime.cuda_available {
        recommendations.push("CPU mode is available; CUDA acceleration requires compatible hardware and a separate supported runtime".into());
    }
    Ok(RuntimeReadiness {
        executable: executable.to_string_lossy().into_owned(),
        ready: missing.is_empty(),
        runtime,
        missing,
        recommendations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_interpreter_is_an_explicit_error() {
        let path = std::env::temp_dir().join("neuralis-interpreter-that-does-not-exist");
        assert!(probe_python(&path, Duration::from_secs(1)).is_err());
    }

    #[test]
    fn diagnostic_deadlines_are_bounded() {
        assert!(probe_python(Path::new("python3"), Duration::ZERO).is_err());
        assert!(probe_python(Path::new("python3"), Duration::from_secs(120)).is_err());
    }

    #[test]
    fn valid_probe_schema_reports_real_dependency_gaps() {
        let json = r#"{"python":"3.12.1","major":3,"minor":12,"platform":"Linux","architecture":"x86_64","packages":{"torch":{"available":false,"version":null}},"cuda_available":false,"cuda_device_count":0}"#;
        let parsed: PythonReadiness = serde_json::from_str(json).unwrap();
        assert!(!parsed.packages["torch"].available);
        assert_eq!(parsed.cuda_device_count, 0);
    }
}
