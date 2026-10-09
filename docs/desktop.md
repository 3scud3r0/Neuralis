# Desktop application

Neuralis Studio uses a Tauri 2 native shell with the same React/WebGL application as the browser. Browser demos run without Python. Instrumented models require the optional Python API runtime, installed independently from the desktop bundle. No model weights, system Python installation or proprietary emulator ROM is bundled.

## Build prerequisites

Use a current stable Rust toolchain, Node 22 or newer, and the repository's locked npm dependencies. Linux requires GTK 3, WebKitGTK 4.1 and a C compiler. Debian and Ubuntu users can install:

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends \
  build-essential pkg-config libgtk-3-dev libwebkit2gtk-4.1-dev \
  librsvg2-dev libayatana-appindicator3-dev patchelf
```

On macOS install Xcode command-line tools. On Windows install Microsoft C++ Build Tools and WebView2. Build packages on the target operating system; a Linux build does not produce verified macOS or Windows installers.

```bash
npm --prefix apps/web ci
npm --prefix apps/desktop install
npm --prefix apps/desktop run doctor
npm --prefix apps/desktop run dev
npm --prefix apps/desktop run build
```

The desktop configuration builds `apps/web/dist`; it does not fetch an external website. Development uses the existing Vite server on port 5173. The native shell is behind the `desktop` Cargo feature so the security and workspace library can be tested on machines without graphical libraries:

```bash
cargo test --locked --manifest-path apps/desktop/src-tauri/Cargo.toml --no-default-features
cargo check --locked --manifest-path apps/desktop/src-tauri/Cargo.toml --features desktop
```

The package metadata includes Debian/AppImage, DMG and NSIS targets. Release signatures, notarization and Windows Authenticode certificates are deployment requirements supplied by the release owner; they are not configured automatically.

## Python runtime

Install the API using its frozen uv lockfile, then choose its interpreter and the absolute Neuralis repository path in desktop settings. For a source checkout, a typical interpreter is `services/api/.venv/bin/python` on POSIX or `services/api/.venv/Scripts/python.exe` on Windows.

```bash
cd services/api
uv sync --frozen --extra dev
```

The native runtime launches `python -m neuralis` directly, with no shell. It selects an available loopback port, generates a 256-bit session bearer token, and checks `/health` followed by the protected `/api/v1/capabilities` endpoint. Only a successful authenticated readiness check exposes the backend to the interface. A bounded startup deadline and early process-exit check report missing packages and invalid interpreters. Closing the application stops its managed Python child and waits for exit. Remote servers remain independently managed.

The backend inherits the user's normal environment except explicit runtime overrides. User-provided interpreter paths execute local programs; select an interpreter you control. Startup does not download or execute a server-provided binary. The sidecar is a managed Python subprocess from the installed runtime, rather than an embedded PyInstaller executable.

## Local and remote connections

The default mode is `browser_only`. The `local` mode requires a selected interpreter and source repository. The `remote` mode requires an HTTPS endpoint, except loopback HTTP for local development. Embedded URL credentials, query parameters and fragments are refused. Enter a bearer token for the current session; it is retained only in native memory. Preferences save the selected endpoint and interpreter without token values.

Native HTTP requests stay on the selected origin. Redirects are disabled; relative path validation rejects encoded traversal and origin escapes. Responses and request bodies are bounded to 16 MiB. TLS certificate and hostname verification remain enabled. A private CA must be installed in the trust store supported by the deployed HTTP client; do not disable verification to connect.

Preferences use a versioned JSON schema, strict fields and atomic replacement. The native configuration directory follows Tauri's operating-system conventions. A failed preferences parse falls back to safe browser mode. Diagnostics retain at most 200 lines of 4 KiB each and redact the session token.

## File permissions and project workspaces

The renderer receives no unrestricted filesystem or shell plugin. Export folders and import files are selected with native dialogs, and grants last only for the running session. JSON checkpoint import reads at most 64 MiB. Export validates portable filenames and uses create-new semantics: existing files and symbolic links cannot be overwritten. File receipts include byte counts and SHA-256 checksums.

The native project workspace creates `neuralis-projects` inside a folder explicitly selected by the user. Every project has a versioned manifest and an `artifacts` directory. Supported artifact kinds are checkpoints, experiments, metrics, datasets and captures. Files remain JSON; no pickle or executable deserialization is used. Every indexed artifact has an integrity checksum, and reads verify it before returning data. Projects support tags and editable metadata, up to 500 artifacts per project and 10,000 projects per workspace.

The initial storage quota is 10 GiB. Individual artifacts and portable bundles are bounded to 64 MiB. Bundles include JSON artifacts and metadata and import into a fresh project identifier, preserving the source project. The importer validates every artifact before writing. Partial newly created imports are cleaned up on failure. Project directories, manifests and artifacts refuse symbolic links and traversal segments. These checks protect normal local workflows; a separate hostile process with write access to the selected folder can still race filesystem operations, so use a workspace owned by your user.

`apps/desktop/bridge.ts` is a browser-safe typed adapter. `isDesktop()` detects native IPC. Its methods cover runtime controls, preferences, file dialogs and project workspace CRUD/artifacts/bundles. Calling native operations in a browser gives a clear capability error.

## Validation boundaries

Core unit tests cover URL restrictions, token generation, redaction, configuration persistence, dialog grants, symlink refusal, artifact corruption, bundle import and project identity. Worker/backend validation is documented separately. Actual native GUI compilation depends on platform libraries. Packaging metadata alone does not establish a signed installer or successful runtime on every operating system.
