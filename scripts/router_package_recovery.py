"""Back up and restore only one router package and its exact distribution metadata.

Shared-site restoration requires the coordinated consumer hold and authorization.
Preparation must restore only isolated copies. No provider or router import occurs.
"""
import argparse
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat


PACKAGE = "skell_e_router"
DIST_PATTERN = re.compile(r"skell_e_router-([A-Za-z0-9._+!-]+)\.dist-info")


def _plain(path):
    info = path.lstat()
    if path.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("Links and reparse points are not allowed")
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise ValueError("Only ordinary files and directories are allowed")


def _location(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            _plain(item)
    return path.resolve()


def _separate(site, backup):
    if site == backup or site in backup.parents or backup in site.parents:
        raise ValueError("Site and backup directories must not overlap")


def _files(root):
    _plain(root)
    if not root.is_dir():
        raise ValueError("Package roots must be directories")
    found = []
    for child in sorted(root.iterdir()):
        _plain(child)
        if child.is_dir():
            found.extend(_files(child))
        else:
            found.append(child)
    return found


def _metadata(root):
    source = root / "METADATA"
    _plain(source)
    data = Parser().parsestr(source.read_text(encoding="utf-8"))
    version = data.get("Version")
    if data.get("Name", "").lower().replace("_", "-") != "skell-e-router":
        raise ValueError("Unexpected distribution name")
    if not isinstance(version, str) or root.name != f"{PACKAGE}-{version}.dist-info":
        raise ValueError("Distribution directory and version disagree")
    return version


def _roots(site):
    if not site.is_dir():
        raise ValueError("Site directory is missing")
    candidates = [p for p in site.iterdir() if DIST_PATTERN.fullmatch(p.name)]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one router distribution")
    package, dist = site / PACKAGE, candidates[0]
    _files(package)
    _files(dist)
    return [package, dist], _metadata(dist)


def create_backup(site: Path, backup_dir: Path):
    site, backup = _location(site), _location(backup_dir)
    _separate(site, backup)
    roots, version = _roots(site)
    source_files = [p for root in roots for p in _files(root)]
    if backup.exists():
        raise ValueError("Backup directory already exists")
    backup.mkdir(parents=True)
    records = []
    for source in source_files:
        relative = source.relative_to(site).as_posix()
        info = source.stat()
        data = source.read_bytes()
        target = backup / "files" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(),
                        "size": len(data), "mtime_ns": info.st_mtime_ns})
    # Refuse a snapshot taken across concurrent package writes.
    if source_files != [p for root in roots for p in _files(root)]:
        raise ValueError("Source file set changed during backup")
    for record in records:
        current = site / record["path"]
        if (hashlib.sha256(current.read_bytes()).hexdigest() != record["sha256"]
                or current.stat().st_mtime_ns != record["mtime_ns"]):
            raise ValueError("Source changed during backup")
    manifest = {"schema_version": 1, "package_name": "skell-e-router", "version": version,
                "source_site": str(site), "roots": [p.name for p in roots], "files": records}
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return verify_backup(backup)


def verify_backup(backup_dir: Path):
    backup = _location(backup_dir)
    manifest_path = backup / "manifest.json"
    _plain(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("version")
    if (manifest.get("schema_version") != 1 or manifest.get("package_name") != "skell-e-router"
            or not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9._+!-]+", version)):
        raise ValueError("Invalid backup identity")
    roots = [PACKAGE, f"{PACKAGE}-{version}.dist-info"]
    if manifest.get("roots") != roots:
        raise ValueError("Invalid backup roots")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Backup has no files")
    payload = backup / "files"
    _plain(payload)
    paths = set()
    for record in records:
        relative = record.get("path")
        if not isinstance(relative, str) or "\\" in relative or ":" in relative:
            raise ValueError("Invalid backup path")
        path = PurePosixPath(relative)
        if (path.is_absolute() or ".." in path.parts or len(path.parts) < 2
                or path.parts[0] not in roots or str(path) != relative or relative in paths):
            raise ValueError("Invalid or duplicate backup path")
        paths.add(relative)
        target = _location(payload / relative)
        if not target.is_relative_to(payload):
            raise ValueError("Backup file escapes its root")
        try:
            data = target.read_bytes()
        except OSError as exc:
            raise ValueError("Backup file is missing or unreadable") from exc
        if len(data) != record.get("size") or hashlib.sha256(data).hexdigest() != record.get("sha256"):
            raise ValueError("Backup file hash mismatch")
        if type(record.get("mtime_ns")) is not int or record["mtime_ns"] < 0:
            raise ValueError("Invalid backup timestamp")
    actual = {p.relative_to(payload).as_posix() for p in _files(payload)}
    if actual != paths:
        raise ValueError("Backup file set differs from manifest")
    if _metadata(payload / roots[1]) != version:
        raise ValueError("Backup metadata version mismatch")
    return manifest


def restore_backup(site: Path, backup_dir: Path, expected_backup_sha256: str, *, replace_version=None):
    site, backup = _location(site), _location(backup_dir)
    _separate(site, backup)
    _plain(backup / "manifest.json")
    digest = hashlib.sha256((backup / "manifest.json").read_bytes()).hexdigest()
    if not isinstance(expected_backup_sha256, str) or digest != expected_backup_sha256.lower():
        raise ValueError("Backup manifest digest mismatch")
    manifest = verify_backup(backup)
    if replace_version is None:
        current_roots, _ = _roots(site)
    else:
        # A failed installer may leave no metadata or both old/new metadata.
        # Only the explicitly named upgrade version and backed-up version are
        # allowed; unexpected router versions still stop before mutation.
        if not isinstance(replace_version, str) or not re.fullmatch(r"[A-Za-z0-9._+!-]+", replace_version):
            raise ValueError("Invalid replacement version")
        allowed = set(manifest["roots"]) | {f"{PACKAGE}-{replace_version}.dist-info"}
        candidates = [p for p in site.iterdir() if p.name == PACKAGE or DIST_PATTERN.fullmatch(p.name)]
        if any(p.name not in allowed for p in candidates):
            raise ValueError("Unexpected router distribution during recovery")
        current_roots = candidates
        for root in current_roots:
            _files(root)
            if root.name != PACKAGE and (root / "METADATA").exists():
                _metadata(root)
    target_roots = [site / name for name in manifest["roots"]]
    # Reject all unsafe roots before the first filesystem mutation.
    for root in set(current_roots + target_roots):
        if root.parent != site or _location(root).parent != site:
            raise ValueError("Restoration root escapes the selected site")
        if root.exists():
            _files(root)
    for root in current_roots:
        # Only validated router roots are removed, never site-packages.
        if root.resolve().parent != site or root == site:
            raise ValueError("Invalid removal target")
        shutil.rmtree(root)
    for record in manifest["files"]:
        target = site / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(backup / "files" / record["path"], target)
        os.utime(target, ns=(record["mtime_ns"], record["mtime_ns"]))
    restored_roots, version = _roots(site)
    restored = {p.relative_to(site).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for root in restored_roots for p in _files(root)}
    expected = {record["path"]: record["sha256"] for record in manifest["files"]}
    if restored != expected or version != manifest["version"]:
        raise ValueError("Restored package differs from the backup")
    return {"status": "restored", "version": version, "site": str(site),
            "manifest_sha256": digest, "verified_files": len(restored)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["backup", "verify", "restore"])
    parser.add_argument("--site", type=Path)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--replace-version", help="Allow partial state from only this named upgrade")
    args = parser.parse_args()
    if args.action == "verify":
        manifest = verify_backup(args.backup_dir)
        result = {"status": "verified", "version": manifest["version"], "files": len(manifest["files"])}
    else:
        if args.site is None:
            parser.error("--site is required")
        if args.action == "backup":
            manifest = create_backup(args.site, args.backup_dir)
            result = {"status": "backed_up", "version": manifest["version"], "files": len(manifest["files"])}
        else:
            if not args.expected_sha256:
                parser.error("--expected-sha256 is required")
            result = restore_backup(args.site, args.backup_dir, args.expected_sha256,
                                    replace_version=args.replace_version)
    result["manifest_sha256"] = hashlib.sha256((args.backup_dir / "manifest.json").read_bytes()).hexdigest()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
