"""Print desktop build prerequisites without reading credential values."""
from __future__ import annotations
import importlib.util
import json
import platform
import shutil
import subprocess
import sys


def version(binary: str, argument: str = "--version") -> str | None:
    path = shutil.which(binary)
    if not path:
        return None
    try:
        result = subprocess.run([path, argument], capture_output=True, text=True, timeout=5)
        return result.stdout.strip().splitlines()[0] if result.returncode == 0 else "installed; version unavailable"
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return "installed; version unavailable"


def main() -> int:
    system = platform.system()
    required = {"node": version("node"), "npm": version("npm"), "rustc": version("rustc"), "cargo": version("cargo")}
    optional = {"python": sys.version.split()[0], "torch": importlib.util.find_spec("torch") is not None}
    missing = [name for name, value in required.items() if value is None]
    linux = {}
    if system == "Linux":
        pkg_config = shutil.which("pkg-config")
        for package in ["gtk+-3.0", "webkit2gtk-4.1", "libsoup-3.0"]:
            found = False
            if pkg_config:
                found = subprocess.run([pkg_config, "--exists", package], capture_output=True).returncode == 0
            linux[package] = found
            if not found:
                missing.append(package)
    print(json.dumps({"platform": system, "architecture": platform.machine(), "required": required,
        "optional": optional, "linux_native_libraries": linux, "missing": missing,
        "python_bundled": False, "code_signing_configured": False}, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
