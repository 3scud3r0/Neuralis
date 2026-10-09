use crate::{
    security::{validate_name, MAX_CHECKPOINT_BYTES},
    Error, Result,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

#[derive(Default)]
pub struct FileGrants {
    exports: HashSet<PathBuf>,
    imports: HashSet<PathBuf>,
}

#[derive(Serialize)]
pub struct FileReceipt {
    pub path: String,
    pub bytes: usize,
    pub sha256: String,
}

impl FileGrants {
    pub fn grant_export_directory(&mut self, selected: &Path) -> Result<PathBuf> {
        let canonical = selected.canonicalize()?;
        if !canonical.is_dir() {
            return Err(Error::Permission(
                "Export destination must be a directory".into(),
            ));
        }
        self.exports.insert(canonical.clone());
        Ok(canonical)
    }

    pub fn grant_import_file(&mut self, selected: &Path) -> Result<PathBuf> {
        let canonical = selected.canonicalize()?;
        if !canonical.is_file() {
            return Err(Error::Permission(
                "Checkpoint selection must be a regular file".into(),
            ));
        }
        self.imports.insert(canonical.clone());
        Ok(canonical)
    }

    pub fn export_json(
        &self,
        directory: &Path,
        filename: &str,
        value: &str,
    ) -> Result<FileReceipt> {
        validate_name(filename)?;
        let canonical = directory.canonicalize()?;
        if !self.exports.contains(&canonical) {
            return Err(Error::Permission(
                "Choose the export folder with the native dialog first".into(),
            ));
        }
        if value.len() > MAX_CHECKPOINT_BYTES {
            return Err(Error::Permission("Export exceeds 64 MiB".into()));
        }
        let parsed: serde_json::Value = serde_json::from_str(value)?;
        if !parsed.is_object() {
            return Err(Error::Config("Checkpoint must be a JSON object".into()));
        }
        let destination = canonical.join(filename);
        let mut options = OpenOptions::new();
        // Create-new refuses overwrite, existing symbolic links and accidental clobbering.
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&destination)?;
        if let Err(error) = file
            .write_all(value.as_bytes())
            .and_then(|_| file.sync_all())
        {
            drop(file);
            let _ = fs::remove_file(&destination);
            return Err(error.into());
        }
        Ok(receipt(&destination, value.as_bytes()))
    }

    pub fn import_json(&self, selected: &Path) -> Result<(String, FileReceipt)> {
        let canonical = selected.canonicalize()?;
        if !self.imports.contains(&canonical) {
            return Err(Error::Permission(
                "Choose the checkpoint with the native dialog first".into(),
            ));
        }
        let file = fs::File::open(&canonical)?;
        if !file.metadata()?.is_file() || file.metadata()?.len() > MAX_CHECKPOINT_BYTES as u64 {
            return Err(Error::Permission(
                "Checkpoint must be a regular JSON file smaller than 64 MiB".into(),
            ));
        }
        let mut bytes = Vec::new();
        file.take((MAX_CHECKPOINT_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_CHECKPOINT_BYTES {
            return Err(Error::Permission(
                "Checkpoint exceeded the read budget".into(),
            ));
        }
        let parsed: serde_json::Value = serde_json::from_slice(&bytes)?;
        if !parsed.is_object() {
            return Err(Error::Config("Checkpoint must be a JSON object".into()));
        }
        let text = String::from_utf8(bytes.clone())
            .map_err(|_| Error::Config("Checkpoint is not UTF-8".into()))?;
        let summary = receipt(&canonical, &bytes);
        Ok((text, summary))
    }

    pub fn clear(&mut self) {
        self.exports.clear();
        self.imports.clear();
    }
}

fn receipt(path: &Path, bytes: &[u8]) -> FileReceipt {
    let hash = Sha256::digest(bytes);
    FileReceipt {
        path: path.to_string_lossy().into_owned(),
        bytes: bytes.len(),
        sha256: hash.iter().map(|byte| format!("{byte:02x}")).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialog_grants_are_required_and_overwrite_is_refused() {
        let directory = std::env::temp_dir().join(format!(
            "neuralis-files-{}",
            crate::security::session_token()
        ));
        fs::create_dir(&directory).unwrap();
        let mut grants = FileGrants::default();
        assert!(grants
            .export_json(&directory, "checkpoint.json", "{}")
            .is_err());
        grants.grant_export_directory(&directory).unwrap();
        let receipt = grants
            .export_json(&directory, "checkpoint.json", "{}")
            .unwrap();
        assert_eq!(receipt.bytes, 2);
        assert_eq!(receipt.sha256.len(), 64);
        assert!(grants
            .export_json(&directory, "checkpoint.json", "{}")
            .is_err());
        assert!(grants
            .export_json(&directory, "../outside.json", "{}")
            .is_err());
        let path = directory.join("checkpoint.json");
        assert!(grants.import_json(&path).is_err());
        grants.grant_import_file(&path).unwrap();
        assert_eq!(grants.import_json(&path).unwrap().0, "{}");
        grants.clear();
        assert!(grants.import_json(&path).is_err());
        fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn cannot_overwrite_through_symbolic_link() {
        let directory = std::env::temp_dir().join(format!(
            "neuralis-links-{}",
            crate::security::session_token()
        ));
        fs::create_dir(&directory).unwrap();
        fs::write(directory.join("actual.json"), "{}").unwrap();
        std::os::unix::fs::symlink(directory.join("actual.json"), directory.join("linked.json"))
            .unwrap();
        let mut grants = FileGrants::default();
        grants.grant_export_directory(&directory).unwrap();
        assert!(grants
            .export_json(&directory, "linked.json", "{\"modified\":true}")
            .is_err());
        assert_eq!(
            fs::read_to_string(directory.join("actual.json")).unwrap(),
            "{}"
        );
        fs::remove_dir_all(directory).unwrap();
    }
}
