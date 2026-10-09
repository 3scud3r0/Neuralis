"""Import integrity and filesystem safety; the original 48 tests are untouched."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "import-axislab.py"
SPEC = importlib.util.spec_from_file_location("axislab_bundle_import", SCRIPT)
IMPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER)


def bundle(*, first_name="LICENSE", payload=b"example\n", repeated=False):
    blocks = [b"Arquivos textuais: 28\n\n"]
    for i in range(28):
        name = first_name if i == 0 or repeated else f"file_{i}.py"
        path = f"{IMPORTER.ROOT}/{name}".encode()
        digest = hashlib.sha256(payload).hexdigest().encode()
        blocks.append(b"===== BEGIN FILE: " + path + b" =====\n"
                      b"# SHA256: " + digest + b"\n# BYTE_LENGTH: "
                      + str(len(payload)).encode() + b"\n" + payload
                      + b"\n===== END FILE: " + path + b" =====\n\n")
    return b"".join(blocks)


def test_payload_marker_and_unicode_preserved(tmp_path):
    payload = "UTF-8: São Paulo\n===== END FILE: fake =====\n".encode()
    source = tmp_path / "bundle.txt"
    source.write_bytes(bundle(payload=payload, first_name="README.md"))
    destination = tmp_path / "extracted"
    manifest = IMPORTER.import_bundle(source, destination)
    assert (destination / "README.original.md").read_bytes() == payload
    assert manifest["text_file_count"] == 28
    assert manifest["source_modifications"] == []
    assert not (destination / "README.md").exists()


@pytest.mark.parametrize("name", ["../outside", "/absolute", "nested/../../outside",
                                 "nested//file", "nested/./file", "C:\\outside",
                                 "file\x00.py", "file with spaces.py"])
def test_reject_unsafe_paths(name):
    with pytest.raises(IMPORTER.BundleError, match="path"):
        IMPORTER.parse_bundle(bundle(first_name=name))


@pytest.mark.parametrize("data", [
    bundle(repeated=True),
    bundle().replace(b"example\n", b"tamper!\n", 1),
    bundle().replace(b"# BYTE_LENGTH: 8", b"# BYTE_LENGTH: 9", 1),
    bundle().replace(b"# SHA256:", b"# SHA256bad:", 1),
    bundle().replace(b"===== END FILE:", b"===== WRONG FILE:", 1),
    bundle()[:-100],
    bundle() + b"trailing corruption",
    bundle().replace(b"Arquivos textuais: 28", b"Arquivos textuais: 29", 1),
    bundle(first_name="file_1.py"),
    bundle(first_name="file_1.py/nested.py"),
])
def test_reject_malformed_or_corrupt_data_before_writing(tmp_path, data):
    source, destination = tmp_path / "bundle.txt", tmp_path / "extracted"
    source.write_bytes(data)
    with pytest.raises(IMPORTER.BundleError):
        IMPORTER.import_bundle(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".axislab-import-*"))


def test_check_mode_and_existing_destination(tmp_path):
    source, destination = tmp_path / "bundle.txt", tmp_path / "extracted"
    source.write_bytes(bundle())
    assert IMPORTER.import_bundle(source, destination, check=True)["text_file_count"] == 28
    assert not destination.exists()
    destination.mkdir()
    (destination / "keep.txt").write_text("original")
    with pytest.raises(IMPORTER.BundleError, match="overwrite"):
        IMPORTER.import_bundle(source, destination)
    assert (destination / "keep.txt").read_text() == "original"


def test_destination_symlink_refused(tmp_path):
    source, destination = tmp_path / "bundle.txt", tmp_path / "extracted"
    source.write_bytes(bundle())
    destination.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(IMPORTER.BundleError, match="overwrite"):
        IMPORTER.import_bundle(source, destination)
    assert not (tmp_path / "missing").exists()
