use crate::{Error, Result};
use rand::RngCore;
use url::Url;

pub const MAX_HTTP_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_CHECKPOINT_BYTES: usize = 64 * 1024 * 1024;

pub fn session_token() -> String {
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn validate_endpoint(value: &str) -> Result<Url> {
    let parsed = Url::parse(value).map_err(|error| Error::Config(error.to_string()))?;
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(Error::Config(
            "Credentials must not be embedded in URLs".into(),
        ));
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(Error::Config(
            "Endpoint cannot have a query or fragment".into(),
        ));
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| Error::Config("Endpoint needs a host".into()))?;
    let loopback = matches!(host, "127.0.0.1" | "localhost" | "[::1]" | "::1");
    if parsed.scheme() != "https" && !(parsed.scheme() == "http" && loopback) {
        return Err(Error::Config(
            "Remote connections require HTTPS; HTTP is allowed only on loopback".into(),
        ));
    }
    Ok(parsed)
}

pub fn validate_api_path(path: &str) -> Result<()> {
    if !path.starts_with('/') || path.starts_with("//") || path.contains('\\') {
        return Err(Error::Permission(
            "API path must be relative to the selected backend".into(),
        ));
    }
    if path.contains('#')
        || path.contains('%')
        || path.split('/').any(|part| part == ".." || part == ".")
    {
        return Err(Error::Permission(
            "Encoded paths and traversal segments are not allowed".into(),
        ));
    }
    if path.len() > 2048 {
        return Err(Error::Permission("API path exceeds 2048 bytes".into()));
    }
    Ok(())
}

pub fn validate_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 120
        || matches!(name, "." | "..")
        || !name
            .chars()
            .all(|char| char.is_ascii_alphanumeric() || matches!(char, '-' | '_' | '.'))
    {
        return Err(Error::Permission(
            "Filename must contain only ASCII letters, digits, dots, dashes and underscores".into(),
        ));
    }
    let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
    if matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && stem
                .chars()
                .last()
                .is_some_and(|char| ('1'..='9').contains(&char)))
    {
        return Err(Error::Permission(
            "Reserved operating-system filename".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_requires_verified_tls() {
        assert!(validate_endpoint("https://lab.example/api/").is_ok());
        assert!(validate_endpoint("http://127.0.0.1:8000").is_ok());
        assert!(validate_endpoint("http://lab.example").is_err());
        assert!(validate_endpoint("https://secret@lab.example").is_err());
        assert!(validate_endpoint("file:///etc/passwd").is_err());
        assert!(validate_endpoint("https://lab.example?token=x").is_err());
    }

    #[test]
    fn prevents_origin_escape_and_traversal() {
        for value in [
            "//evil.example",
            "/../secret",
            "/%2e%2e/secret",
            "/a\\b",
            "https://evil.example",
        ] {
            assert!(validate_api_path(value).is_err(), "{value}");
        }
        assert!(validate_api_path("/runs?id=42").is_ok());
    }

    #[test]
    fn filenames_are_portable_and_confined() {
        for name in ["../../token", "a/b", "NUL.json", "COM1", "", "hello world"] {
            assert!(validate_name(name).is_err(), "{name}");
        }
        assert!(validate_name("run-42.checkpoint.json").is_ok());
    }

    #[test]
    fn tokens_have_entropy_and_are_ephemeral() {
        let first = session_token();
        assert_eq!(first.len(), 64);
        assert_ne!(first, session_token());
    }
}
