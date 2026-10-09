use crate::{
    config::{BackendMode, Preferences},
    security::{session_token, validate_api_path, validate_endpoint, MAX_HTTP_BYTES},
    Error, Result,
};
use reqwest::blocking::Client;
use serde::Serialize;
use std::{
    collections::VecDeque,
    io::{BufRead, BufReader, Read},
    net::TcpListener,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimePhase {
    BrowserOnly,
    Starting,
    Ready,
    Failed,
    Stopped,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeStatus {
    pub phase: RuntimePhase,
    pub endpoint: Option<String>,
    pub managed: bool,
    pub process_id: Option<u32>,
    pub message: String,
}

pub struct Runtime {
    child: Option<Child>,
    endpoint: Option<String>,
    token: Option<String>,
    phase: RuntimePhase,
    logs: Arc<Mutex<VecDeque<String>>>,
    client: Client,
    message: String,
}

impl Runtime {
    pub fn new() -> Result<Self> {
        Ok(Self {
            child: None,
            endpoint: None,
            token: None,
            phase: RuntimePhase::BrowserOnly,
            logs: Arc::new(Mutex::new(VecDeque::with_capacity(200))),
            client: Client::builder()
                .timeout(Duration::from_secs(10))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            message: "Browser demos are available without Python".into(),
        })
    }

    pub fn start(
        &mut self,
        preferences: &Preferences,
        remote_token: Option<String>,
    ) -> Result<RuntimeStatus> {
        preferences.validate()?;
        self.stop()?;
        if preferences.mode == BackendMode::BrowserOnly {
            self.phase = RuntimePhase::BrowserOnly;
            self.message = "Browser demos are available without Python".into();
            return Ok(self.status());
        }
        self.phase = RuntimePhase::Starting;
        if preferences.mode == BackendMode::Remote {
            self.endpoint = preferences.remote_endpoint.clone();
            self.token = remote_token.filter(|value| !value.is_empty());
            if let Err(error) = self.check_health() {
                self.phase = RuntimePhase::Failed;
                self.message = format!("Remote readiness failed: {error}");
                self.token = None;
                return Err(error);
            }
            self.phase = RuntimePhase::Ready;
            self.message = "Connected to the selected remote backend".into();
            return Ok(self.status());
        }
        let root = preferences.repository_root.as_ref().unwrap();
        let token = session_token();
        // Reserve an OS-selected loopback port; the Python process binds it immediately.
        // If another process wins the bind race, readiness requires the unique bearer token.
        let reservation = TcpListener::bind(("127.0.0.1", 0))?;
        let port = reservation.local_addr()?.port();
        self.endpoint = Some(format!("http://127.0.0.1:{port}"));
        self.token = Some(token.clone());
        let mut command = Command::new(&preferences.python_executable);
        command
            .args(["-m", "neuralis"])
            .current_dir(root.join("services/api"))
            .env("NEURALIS_HOST", "127.0.0.1")
            .env("NEURALIS_PORT", port.to_string())
            .env("NEURALIS_TOKEN", &token)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONPATH", root.join("services/api/src"))
            .env("OMP_NUM_THREADS", "2")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }
        if let Some(directory) = &preferences.data_directory {
            std::fs::create_dir_all(directory)?;
            command.env("NEURALIS_DATA", directory);
        }
        drop(reservation);
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                self.phase = RuntimePhase::Failed;
                self.token = None;
                self.message = "Python could not be started; check the selected interpreter".into();
                return Err(error.into());
            }
        };
        if let Some(stdout) = child.stdout.take() {
            capture_log(stdout, self.logs.clone(), token.clone());
        }
        if let Some(stderr) = child.stderr.take() {
            capture_log(stderr, self.logs.clone(), token);
        }
        self.child = Some(child);
        let deadline = Instant::now() + Duration::from_secs(preferences.startup_timeout_seconds);
        let mut last_error = String::new();
        while Instant::now() < deadline {
            if let Some(code) = self.child.as_mut().unwrap().try_wait()? {
                self.child.take();
                self.phase = RuntimePhase::Failed;
                self.token = None;
                self.message = format!("Python exited before readiness: {code}");
                return Err(Error::Runtime(self.message.clone()));
            }
            match self.check_health() {
                Ok(()) => {
                    self.phase = RuntimePhase::Ready;
                    self.message = "Authenticated local backend is ready".into();
                    return Ok(self.status());
                }
                Err(error) => last_error = error.to_string(),
            }
            thread::sleep(Duration::from_millis(150));
        }
        self.stop()?;
        self.phase = RuntimePhase::Failed;
        self.message = format!("Backend readiness timed out: {last_error}");
        Err(Error::Runtime(self.message.clone()))
    }

    fn check_health(&self) -> Result<()> {
        let value = self.request("GET", "/health", None)?;
        if value.get("status").and_then(|value| value.as_str()) != Some("ok") {
            return Err(Error::Runtime(
                "Health endpoint did not return status=ok".into(),
            ));
        }
        // A protected endpoint proves the local port is our authenticated backend.
        self.request("GET", "/api/v1/capabilities", None)?;
        Ok(())
    }

    pub fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        validate_api_path(path)?;
        let endpoint = self
            .endpoint
            .as_ref()
            .ok_or_else(|| Error::Runtime("No backend is selected".into()))?;
        let parsed = validate_endpoint(endpoint)?;
        let base = parsed.as_str().trim_end_matches('/');
        let url = format!("{base}{path}");
        let method = match method {
            "GET" => reqwest::Method::GET,
            "POST" => reqwest::Method::POST,
            "PUT" => reqwest::Method::PUT,
            "DELETE" => reqwest::Method::DELETE,
            _ => {
                return Err(Error::Permission(
                    "Only GET, POST, PUT and DELETE are supported".into(),
                ))
            }
        };
        let mut request = self.client.request(method, url);
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }
        if let Some(value) = body {
            if serde_json::to_vec(&value)?.len() > MAX_HTTP_BYTES {
                return Err(Error::Permission(
                    "Request exceeds the 16 MiB budget".into(),
                ));
            }
            request = request.json(&value);
        }
        let response = request.send()?;
        let status = response.status();
        let mut bytes = Vec::new();
        response
            .take((MAX_HTTP_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_HTTP_BYTES {
            return Err(Error::Runtime("Response exceeded the 16 MiB budget".into()));
        }
        if !status.is_success() {
            // Do not echo arbitrary error bodies; they can contain credentials or user data.
            return Err(Error::Runtime(format!(
                "Backend returned HTTP {}",
                status.as_u16()
            )));
        }
        if bytes.is_empty() {
            return Ok(serde_json::Value::Null);
        }
        Ok(serde_json::from_slice(&bytes)?)
    }

    pub fn status(&mut self) -> RuntimeStatus {
        if let Some(child) = self.child.as_mut() {
            if let Ok(Some(code)) = child.try_wait() {
                self.child.take();
                self.token = None;
                self.phase = RuntimePhase::Failed;
                self.message = format!("Managed Python process exited: {code}");
            }
        }
        RuntimeStatus {
            phase: self.phase.clone(),
            endpoint: self.endpoint.clone(),
            managed: self.child.is_some(),
            process_id: self.child.as_ref().map(Child::id),
            message: self.message.clone(),
        }
    }

    pub fn diagnostics(&self) -> Vec<String> {
        self.logs
            .lock()
            .map(|lines| lines.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn stop(&mut self) -> Result<()> {
        if let Some(mut child) = self.child.take() {
            if child.try_wait()?.is_none() {
                child.kill()?;
                child.wait()?;
            }
        }
        self.token = None;
        self.endpoint = None;
        self.phase = RuntimePhase::Stopped;
        self.message = "Backend connection stopped".into();
        Ok(())
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn capture_log<R: Read + Send + 'static>(
    reader: R,
    logs: Arc<Mutex<VecDeque<String>>>,
    token: String,
) {
    thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        loop {
            let mut bytes = Vec::new();
            // A malicious or broken subprocess cannot cause unbounded line allocations.
            match reader.by_ref().take(4097).read_until(b'\n', &mut bytes) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let line = String::from_utf8_lossy(&bytes[..bytes.len().min(4096)])
                        .replace(&token, "[redacted]");
                    if let Ok(mut lines) = logs.lock() {
                        if lines.len() >= 200 {
                            lines.pop_front();
                        }
                        lines.push_back(line.trim().to_string());
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_mode_does_not_launch_a_process() {
        let mut runtime = Runtime::new().unwrap();
        let status = runtime.start(&Preferences::default(), None).unwrap();
        assert!(matches!(status.phase, RuntimePhase::BrowserOnly));
        assert!(!status.managed);
        assert!(runtime.request("GET", "/health", None).is_err());
        runtime.stop().unwrap();
        assert!(matches!(runtime.status().phase, RuntimePhase::Stopped));
    }

    #[test]
    fn logs_are_bounded_and_token_is_redacted() {
        let logs = Arc::new(Mutex::new(VecDeque::new()));
        capture_log(
            std::io::Cursor::new(b"my secret token\n".to_vec()),
            logs.clone(),
            "secret".into(),
        );
        for _ in 0..20 {
            if !logs.lock().unwrap().is_empty() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(logs.lock().unwrap()[0], "my [redacted] token");
    }
}
