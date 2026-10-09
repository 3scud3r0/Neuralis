use crate::{
    security::{session_token, validate_name, MAX_CHECKPOINT_BYTES},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

const MAX_MANIFEST_BYTES: usize = 512 * 1024;
const MAX_PROJECTS: usize = 10_000;
const MAX_ARTIFACTS: usize = 500;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectManifest {
    pub schema_version: u32,
    pub id: String,
    pub name: String,
    pub description: String,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub tags: Vec<String>,
    pub artifacts: Vec<ArtifactManifest>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactManifest {
    pub filename: String,
    pub kind: ArtifactKind,
    pub bytes: usize,
    pub sha256: String,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Checkpoint,
    Experiment,
    Metrics,
    Dataset,
    Capture,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectBundle {
    pub format: String,
    pub version: u32,
    pub project: ProjectManifest,
    pub artifacts: HashMap<String, String>,
}

pub struct Workspace {
    root: PathBuf,
    quota_bytes: u64,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

impl ProjectManifest {
    pub fn validate(&self) -> Result<()> {
        validate_name(&self.id)?;
        if self.schema_version != 1 || !self.id.starts_with("project-") {
            return Err(Error::Config("Unsupported project manifest".into()));
        }
        if self.name.trim().is_empty() || self.name.len() > 200 || self.description.len() > 10_000 {
            return Err(Error::Config(
                "Project name and description exceed allowed bounds".into(),
            ));
        }
        if self.tags.len() > 32 || self.tags.iter().any(|tag| tag.is_empty() || tag.len() > 64) {
            return Err(Error::Config(
                "Projects support at most 32 tags of 64 bytes each".into(),
            ));
        }
        if self.artifacts.len() > MAX_ARTIFACTS {
            return Err(Error::Config(
                "Project exceeds its artifact count budget".into(),
            ));
        }
        let mut names = std::collections::HashSet::new();
        for artifact in &self.artifacts {
            validate_name(&artifact.filename)?;
            if !artifact.filename.ends_with(".json") || artifact.bytes > MAX_CHECKPOINT_BYTES {
                return Err(Error::Config("Artifacts must be bounded JSON files".into()));
            }
            if artifact.sha256.len() != 64
                || !artifact
                    .sha256
                    .chars()
                    .all(|value| value.is_ascii_hexdigit())
            {
                return Err(Error::Config("Artifact checksum is invalid".into()));
            }
            if !names.insert(&artifact.filename) {
                return Err(Error::Config(
                    "Project has duplicate artifact filenames".into(),
                ));
            }
        }
        Ok(())
    }
}

impl Workspace {
    // Called only after native folder selection; no remote API chooses this root.
    pub fn open(selected: &Path, quota_bytes: u64) -> Result<Self> {
        if !(1024 * 1024..=1024 * 1024 * 1024 * 1024).contains(&quota_bytes) {
            return Err(Error::Config(
                "Workspace quota must be between 1 MiB and 1 TiB".into(),
            ));
        }
        let root = selected.canonicalize()?;
        if !root.is_dir() {
            return Err(Error::Config("Workspace root must be a directory".into()));
        }
        let projects = root.join("neuralis-projects");
        if projects.exists() {
            ensure_regular_directory(&projects)?;
        } else {
            fs::create_dir(&projects)?;
        }
        Ok(Self { root, quota_bytes })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn projects(&self) -> PathBuf {
        self.root.join("neuralis-projects")
    }

    fn project_path(&self, id: &str) -> Result<PathBuf> {
        validate_name(id)?;
        if !id.starts_with("project-") {
            return Err(Error::Permission("Invalid project identifier".into()));
        }
        let path = self.projects().join(id);
        if path.exists() {
            ensure_regular_directory(&path)?;
        }
        Ok(path)
    }

    pub fn list(&self) -> Result<Vec<ProjectManifest>> {
        ensure_regular_directory(&self.projects())?;
        let mut values = Vec::new();
        for entry in fs::read_dir(self.projects())? {
            let entry = entry?;
            if !entry.file_name().to_string_lossy().starts_with("project-") {
                continue;
            }
            if values.len() >= MAX_PROJECTS {
                return Err(Error::Config("Workspace has too many projects".into()));
            }
            let id = entry.file_name().to_string_lossy().into_owned();
            values.push(self.load(&id)?);
        }
        values.sort_by_key(|value| std::cmp::Reverse(value.updated_at_ms));
        Ok(values)
    }

    pub fn create(
        &self,
        name: String,
        description: String,
        tags: Vec<String>,
    ) -> Result<ProjectManifest> {
        if self.list()?.len() >= MAX_PROJECTS {
            return Err(Error::Config("Workspace project limit reached".into()));
        }
        let time = now();
        let project = ProjectManifest {
            schema_version: 1,
            id: format!("project-{}", &session_token()[..24]),
            name,
            description,
            created_at_ms: time,
            updated_at_ms: time,
            tags,
            artifacts: Vec::new(),
        };
        project.validate()?;
        self.check_quota(MAX_MANIFEST_BYTES as u64)?;
        let path = self.project_path(&project.id)?;
        fs::create_dir(&path)?;
        if let Err(error) = self.save_manifest(&project) {
            let _ = fs::remove_dir(&path);
            return Err(error);
        }
        fs::create_dir(path.join("artifacts"))?;
        Ok(project)
    }

    pub fn load(&self, id: &str) -> Result<ProjectManifest> {
        let path = self.project_path(id)?;
        let bytes = bounded_read(&path.join("project.json"), MAX_MANIFEST_BYTES)?;
        let value: ProjectManifest = serde_json::from_slice(&bytes)?;
        value.validate()?;
        if value.id != id {
            return Err(Error::Config(
                "Project identifier does not match its directory".into(),
            ));
        }
        Ok(value)
    }

    pub fn update_metadata(
        &self,
        id: &str,
        name: String,
        description: String,
        tags: Vec<String>,
    ) -> Result<ProjectManifest> {
        let mut project = self.load(id)?;
        project.name = name;
        project.description = description;
        project.tags = tags;
        project.updated_at_ms = now();
        project.validate()?;
        self.save_manifest(&project)?;
        Ok(project)
    }

    pub fn add_artifact(
        &self,
        id: &str,
        filename: &str,
        kind: ArtifactKind,
        text: &str,
    ) -> Result<ArtifactManifest> {
        validate_name(filename)?;
        if !filename.ends_with(".json") || text.len() > MAX_CHECKPOINT_BYTES {
            return Err(Error::Config(
                "Artifact must be a JSON file smaller than 64 MiB".into(),
            ));
        }
        let value: serde_json::Value = serde_json::from_str(text)?;
        if !value.is_object() && !value.is_array() {
            return Err(Error::Config(
                "Artifact must contain an object or array".into(),
            ));
        }
        let mut project = self.load(id)?;
        if project.artifacts.len() >= MAX_ARTIFACTS
            || project
                .artifacts
                .iter()
                .any(|item| item.filename == filename)
        {
            return Err(Error::Config(
                "Artifact already exists or project artifact limit is reached".into(),
            ));
        }
        self.check_quota((text.len() + MAX_MANIFEST_BYTES) as u64)?;
        let artifact_directory = self.project_path(id)?.join("artifacts");
        ensure_regular_directory(&artifact_directory)?;
        let artifact_path = artifact_directory.join(filename);
        write_new(&artifact_path, text.as_bytes())?;
        let artifact = ArtifactManifest {
            filename: filename.into(),
            kind,
            bytes: text.len(),
            sha256: digest(text.as_bytes()),
            created_at_ms: now(),
        };
        project.artifacts.push(artifact.clone());
        project.updated_at_ms = now();
        if let Err(error) = self.save_manifest(&project) {
            let _ = fs::remove_file(artifact_path);
            return Err(error);
        }
        Ok(artifact)
    }

    pub fn read_artifact(&self, id: &str, filename: &str) -> Result<String> {
        validate_name(filename)?;
        let project = self.load(id)?;
        let expected = project
            .artifacts
            .iter()
            .find(|value| value.filename == filename)
            .ok_or_else(|| Error::Config("Artifact is not indexed by the project".into()))?;
        let directory = self.project_path(id)?.join("artifacts");
        ensure_regular_directory(&directory)?;
        let bytes = bounded_read(&directory.join(filename), MAX_CHECKPOINT_BYTES)?;
        if bytes.len() != expected.bytes || digest(&bytes) != expected.sha256 {
            return Err(Error::Config(
                "Artifact checksum does not match the manifest".into(),
            ));
        }
        serde_json::from_slice::<serde_json::Value>(&bytes)?;
        String::from_utf8(bytes).map_err(|_| Error::Config("Artifact is not UTF-8".into()))
    }

    pub fn bundle(&self, id: &str) -> Result<String> {
        let project = self.load(id)?;
        let mut artifacts = HashMap::new();
        let mut total = 0;
        for artifact in &project.artifacts {
            let text = self.read_artifact(id, &artifact.filename)?;
            total += text.len();
            if total > MAX_CHECKPOINT_BYTES {
                return Err(Error::Config(
                    "Project exceeds the 64 MiB portable bundle budget".into(),
                ));
            }
            artifacts.insert(artifact.filename.clone(), text);
        }
        let encoded = serde_json::to_string(&ProjectBundle {
            format: "neuralis-project".into(),
            version: 1,
            project,
            artifacts,
        })?;
        if encoded.len() > MAX_CHECKPOINT_BYTES {
            return Err(Error::Config("Bundle exceeds 64 MiB".into()));
        }
        Ok(encoded)
    }

    pub fn import_bundle(&self, text: &str) -> Result<ProjectManifest> {
        if text.len() > MAX_CHECKPOINT_BYTES {
            return Err(Error::Config("Bundle exceeds 64 MiB".into()));
        }
        let bundle: ProjectBundle = serde_json::from_str(text)?;
        if bundle.format != "neuralis-project" || bundle.version != 1 {
            return Err(Error::Config("Unsupported portable project format".into()));
        }
        bundle.project.validate()?;
        if bundle.artifacts.len() != bundle.project.artifacts.len() {
            return Err(Error::Config(
                "Bundle manifest and artifact count differ".into(),
            ));
        }
        // Preserve original JSON bytes so manifest checksums verify portable
        // bundles before any new project files are written.
        let mut pending = Vec::new();
        for artifact in &bundle.project.artifacts {
            let value = bundle
                .artifacts
                .get(&artifact.filename)
                .ok_or_else(|| Error::Config("Bundle is missing an indexed artifact".into()))?;
            if value.len() != artifact.bytes || digest(value.as_bytes()) != artifact.sha256 {
                return Err(Error::Config(
                    "Bundled artifact checksum does not match its manifest".into(),
                ));
            }
            let parsed: serde_json::Value = serde_json::from_str(value)?;
            if !parsed.is_object() && !parsed.is_array() {
                return Err(Error::Config(
                    "Bundled artifact must contain an object or array".into(),
                ));
            }
            pending.push((
                artifact.filename.clone(),
                artifact.kind.clone(),
                value.clone(),
            ));
        }
        let total: usize = pending.iter().map(|(_, _, value)| value.len()).sum();
        self.check_quota((total + MAX_MANIFEST_BYTES) as u64)?;
        let project = self.create(
            bundle.project.name,
            bundle.project.description,
            bundle.project.tags,
        )?;
        let outcome = (|| -> Result<ProjectManifest> {
            for (filename, kind, content) in pending {
                self.add_artifact(&project.id, &filename, kind, &content)?;
            }
            self.load(&project.id)
        })();
        if outcome.is_err() {
            // Only this newly created, validated directory can be cleaned up.
            let _ = fs::remove_dir_all(self.project_path(&project.id)?);
        }
        outcome
    }

    pub fn usage(&self) -> Result<u64> {
        directory_size(&self.projects(), 0)
    }

    fn check_quota(&self, additional: u64) -> Result<()> {
        if self.usage()?.saturating_add(additional) > self.quota_bytes {
            return Err(Error::Config(
                "Workspace storage quota would be exceeded".into(),
            ));
        }
        Ok(())
    }

    fn save_manifest(&self, project: &ProjectManifest) -> Result<()> {
        project.validate()?;
        let bytes = serde_json::to_vec_pretty(project)?;
        if bytes.len() > MAX_MANIFEST_BYTES {
            return Err(Error::Config("Project manifest exceeds its budget".into()));
        }
        let directory = self.project_path(&project.id)?;
        ensure_regular_directory(&directory)?;
        let path = directory.join("project.json");
        if path.exists() && fs::symlink_metadata(&path)?.file_type().is_symlink() {
            return Err(Error::Permission(
                "Symbolic-link manifests are refused".into(),
            ));
        }
        let temp = directory.join(format!("project.{}.tmp", session_token()));
        write_new(&temp, &bytes)?;
        let backup = directory.join("project.previous.json");
        if cfg!(windows) && path.exists() {
            fs::rename(&path, &backup)?;
        }
        if let Err(error) = fs::rename(&temp, &path) {
            if cfg!(windows) && backup.exists() {
                let _ = fs::rename(&backup, &path);
            }
            let _ = fs::remove_file(temp);
            return Err(error.into());
        }
        if backup.exists() {
            fs::remove_file(backup)?;
        }
        Ok(())
    }
}

fn ensure_regular_directory(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(Error::Permission(
            "Workspace directories cannot be symbolic links".into(),
        ));
    }
    Ok(())
}

fn bounded_read(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > limit as u64 {
        return Err(Error::Permission(
            "Workspace file is not a bounded regular file".into(),
        ));
    }
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take((limit + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(Error::Permission(
            "Workspace file exceeded its read budget".into(),
        ));
    }
    Ok(bytes)
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    if let Err(error) = file.write_all(bytes).and_then(|_| file.sync_all()) {
        drop(file);
        let _ = fs::remove_file(path);
        return Err(error.into());
    }
    Ok(())
}

fn directory_size(path: &Path, depth: u32) -> Result<u64> {
    if depth > 4 {
        return Err(Error::Config(
            "Workspace contains unexpected directory nesting".into(),
        ));
    }
    ensure_regular_directory(path)?;
    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = fs::symlink_metadata(entry.path())?;
        if metadata.file_type().is_symlink() {
            return Err(Error::Permission(
                "Symbolic links are refused in workspace storage".into(),
            ));
        }
        total = total.saturating_add(if metadata.is_dir() {
            directory_size(&entry.path(), depth + 1)?
        } else {
            metadata.len()
        });
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn directory() -> PathBuf {
        let path = std::env::temp_dir().join(format!("neuralis-workspace-{}", session_token()));
        fs::create_dir(&path).unwrap();
        path
    }

    #[test]
    fn projects_round_trip_with_checked_artifacts() {
        let path = directory();
        let workspace = Workspace::open(&path, 10 * 1024 * 1024).unwrap();
        let project = workspace
            .create(
                "Vision experiment".into(),
                "CPU baseline".into(),
                vec!["vision".into()],
            )
            .unwrap();
        workspace
            .add_artifact(
                &project.id,
                "checkpoint.json",
                ArtifactKind::Checkpoint,
                "{\"weights\":[1,2]}",
            )
            .unwrap();
        assert_eq!(workspace.list().unwrap().len(), 1);
        assert_eq!(
            workspace
                .read_artifact(&project.id, "checkpoint.json")
                .unwrap(),
            "{\"weights\":[1,2]}"
        );
        let bundle = workspace.bundle(&project.id).unwrap();
        let imported = workspace.import_bundle(&bundle).unwrap();
        assert_ne!(project.id, imported.id);
        assert_eq!(imported.artifacts.len(), 1);
        assert_eq!(workspace.list().unwrap().len(), 2);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn detects_external_artifact_corruption() {
        let path = directory();
        let workspace = Workspace::open(&path, 10 * 1024 * 1024).unwrap();
        let project = workspace.create("Test".into(), "".into(), vec![]).unwrap();
        workspace
            .add_artifact(&project.id, "capture.json", ArtifactKind::Capture, "{}")
            .unwrap();
        fs::write(
            path.join("neuralis-projects")
                .join(&project.id)
                .join("artifacts/capture.json"),
            "{\"corrupted\":true}",
        )
        .unwrap();
        assert!(workspace
            .read_artifact(&project.id, "capture.json")
            .is_err());
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn rejects_traversal_and_duplicate_artifacts() {
        let path = directory();
        let workspace = Workspace::open(&path, 10 * 1024 * 1024).unwrap();
        let project = workspace.create("Test".into(), "".into(), vec![]).unwrap();
        assert!(workspace
            .add_artifact(&project.id, "../outside.json", ArtifactKind::Dataset, "{}")
            .is_err());
        workspace
            .add_artifact(&project.id, "data.json", ArtifactKind::Dataset, "[]")
            .unwrap();
        assert!(workspace
            .add_artifact(&project.id, "data.json", ArtifactKind::Dataset, "[]")
            .is_err());
        assert!(workspace.load("../../outside").is_err());
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn failed_bundle_import_does_not_create_partial_project() {
        let path = directory();
        let workspace = Workspace::open(&path, 10 * 1024 * 1024).unwrap();
        assert!(workspace.import_bundle("{\"format\":\"wrong\"}").is_err());
        assert_eq!(workspace.list().unwrap().len(), 0);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn bundled_checksum_failure_does_not_create_files() {
        let path = directory();
        let workspace = Workspace::open(&path, 10 * 1024 * 1024).unwrap();
        let project = workspace.create("Test".into(), "".into(), vec![]).unwrap();
        workspace
            .add_artifact(&project.id, "weights.json", ArtifactKind::Checkpoint, "{}")
            .unwrap();
        let mut bundle: ProjectBundle =
            serde_json::from_str(&workspace.bundle(&project.id).unwrap()).unwrap();
        bundle
            .artifacts
            .insert("weights.json".into(), "{\"tampered\":true}".into());
        assert!(workspace
            .import_bundle(&serde_json::to_string(&bundle).unwrap())
            .is_err());
        assert_eq!(workspace.list().unwrap().len(), 1);
        fs::remove_dir_all(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_project_directories() {
        let path = directory();
        let workspace = Workspace::open(&path, 10 * 1024 * 1024).unwrap();
        std::os::unix::fs::symlink("/tmp", path.join("neuralis-projects/project-evil")).unwrap();
        assert!(workspace.load("project-evil").is_err());
        assert!(workspace.usage().is_err());
        fs::remove_dir_all(path).unwrap();
    }
}
