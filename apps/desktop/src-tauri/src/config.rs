use crate::{security::validate_endpoint, Error, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BackendMode {
    BrowserOnly,
    Local,
    Remote,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preferences {
    pub schema_version: u32,
    pub mode: BackendMode,
    pub python_executable: PathBuf,
    pub repository_root: Option<PathBuf>,
    pub remote_endpoint: Option<String>,
    pub data_directory: Option<PathBuf>,
    pub startup_timeout_seconds: u64,
    pub capture_budget_bytes: usize,
}

impl Default for Preferences {
    fn default() -> Self {
        Self {
            schema_version: 1,
            mode: BackendMode::BrowserOnly,
            python_executable: PathBuf::from(if cfg!(windows) {
                "python.exe"
            } else {
                "python3"
            }),
            repository_root: None,
            remote_endpoint: None,
            data_directory: None,
            startup_timeout_seconds: 30,
            capture_budget_bytes: 16 * 1024 * 1024,
        }
    }
}

impl Preferences {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(Error::Config("Unsupported preferences schema".into()));
        }
        if !(1..=120).contains(&self.startup_timeout_seconds) {
            return Err(Error::Config(
                "Startup timeout must be between 1 and 120 seconds".into(),
            ));
        }
        if !(1024..=64 * 1024 * 1024).contains(&self.capture_budget_bytes) {
            return Err(Error::Config(
                "Capture budget must be between 1 KiB and 64 MiB".into(),
            ));
        }
        if self.python_executable.as_os_str().is_empty() {
            return Err(Error::Config("Python executable is empty".into()));
        }
        if self
            .data_directory
            .as_ref()
            .is_some_and(|path| !path.is_absolute())
        {
            return Err(Error::Config(
                "Backend data directory must be an absolute path".into(),
            ));
        }
        if self.mode == BackendMode::Local {
            let root = self.repository_root.as_ref().ok_or_else(|| {
                Error::Config("Select the Neuralis repository before launching Python".into())
            })?;
            if !root.is_absolute() || !root.join("services/api").is_dir() {
                return Err(Error::Config(
                    "Local repository must contain services/api".into(),
                ));
            }
        }
        if let Some(endpoint) = &self.remote_endpoint {
            validate_endpoint(endpoint)?;
        }
        if self.mode == BackendMode::Remote && self.remote_endpoint.is_none() {
            return Err(Error::Config("Remote mode needs an endpoint".into()));
        }
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = fs::read(path)?;
        if bytes.len() > 64 * 1024 {
            return Err(Error::Config("Preferences file is too large".into()));
        }
        let value: Self = serde_json::from_slice(&bytes)?;
        value.validate()?;
        Ok(value)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let parent = path
            .parent()
            .ok_or_else(|| Error::Config("Preferences path has no parent".into()))?;
        fs::create_dir_all(parent)?;
        let temp = parent.join(format!(
            "preferences.{}.tmp",
            crate::security::session_token()
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let outcome = (|| -> Result<()> {
            let mut file = options.open(&temp)?;
            file.write_all(&serde_json::to_vec_pretty(self)?)?;
            file.sync_all()?;
            // Windows rename does not replace an existing target. A backup permits rollback.
            let backup = parent.join("preferences.previous.json");
            if cfg!(windows) && path.exists() {
                fs::rename(path, &backup)?;
            }
            if let Err(error) = fs::rename(&temp, path) {
                if cfg!(windows) && backup.exists() {
                    let _ = fs::rename(&backup, path);
                }
                return Err(error.into());
            }
            if backup.exists() {
                fs::remove_file(backup)?;
            }
            Ok(())
        })();
        if temp.exists() {
            let _ = fs::remove_file(temp);
        }
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_unknown_secrets_and_versions() {
        let json = r#"{"schema_version":1,"mode":"browser_only","python_executable":"python3","repository_root":null,"remote_endpoint":null,"startup_timeout_seconds":30,"capture_budget_bytes":1024,"token":"secret"}"#;
        assert!(serde_json::from_str::<Preferences>(json).is_err());
        let mut preferences = Preferences::default();
        preferences.schema_version = 2;
        assert!(preferences.validate().is_err());
    }

    #[test]
    fn configuration_round_trip_has_no_credentials() {
        let directory = std::env::temp_dir().join(format!(
            "neuralis-config-{}",
            crate::security::session_token()
        ));
        let path = directory.join("preferences.json");
        let preferences = Preferences::default();
        preferences.save(&path).unwrap();
        assert_eq!(
            Preferences::load(&path).unwrap().mode,
            BackendMode::BrowserOnly
        );
        assert!(!fs::read_to_string(&path).unwrap().contains("token"));
        fs::remove_dir_all(directory).unwrap();
    }
}
