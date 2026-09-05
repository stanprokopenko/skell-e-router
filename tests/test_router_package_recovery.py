"""Offline recovery checks using disposable, synthetic site-packages trees only."""

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "router_package_recovery.py"
SPEC = importlib.util.spec_from_file_location("router_package_recovery", SCRIPT)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def write_distribution(site, version="3.26.2", name="skell-e-router"):
    """Create realistic metadata without importing or installing any package."""
    package = site / "skell_e_router"
    metadata = site / f"skell_e_router-{version}.dist-info"
    package.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    files = {
        "skell_e_router/__init__.py": f"__version__ = '{version}'\n".encode(),
        "skell_e_router/utils.py": b"ORIGINAL = True\n",
        "skell_e_router/__pycache__/utils.cpython-312.pyc": b"\x00\xfforiginal-bytecode\r\n",
        f"{metadata.name}/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        ).encode(),
        f"{metadata.name}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{metadata.name}/INSTALLER": b"pip\n",
        f"{metadata.name}/direct_url.json": (
            '{"url":"https://example.invalid/router.git","vcs_info":'
            '{"vcs":"git","commit_id":"d8' + "1" * 38 + '",'
            '"requested_revision":"main"}}\n'
        ).encode(),
    }
    if version == "3.26.3":
        files["skell_e_router/errors.py"] = b"class RouterError(Exception): pass\n"
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for relative, data in files.items():
        target = site / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow([relative, "sha256=" + digest, str(len(data))])
    writer.writerow([f"{metadata.name}/RECORD", "", ""])
    (metadata / "RECORD").write_bytes(record.getvalue().encode())
    return metadata


class RouterPackageRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="router-recovery-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.site = self.root / "site-packages"
        self.site.mkdir()
        self.backup = self.root / "backup"
        self.metadata = write_distribution(self.site)
        (self.site / "unrelated.py").write_bytes(b"leave this file alone\x00\xff")
        unrelated = self.site / "unrelated-1.0.dist-info"
        unrelated.mkdir()
        (unrelated / "METADATA").write_bytes(b"Name: unrelated\nVersion: 1.0\n")
        (unrelated / "RECORD").write_bytes(b"unrelated.py,,\n")
        self.original = snapshot(self.site)

    def make_backup(self):
        manifest = recovery.create_backup(self.site, self.backup)
        self.assertIsInstance(manifest, dict)
        self.assertTrue((self.backup / "manifest.json").is_file())
        return sha256(self.backup / "manifest.json")

    def upgrade(self):
        shutil.rmtree(self.site / "skell_e_router")
        shutil.rmtree(self.metadata)
        write_distribution(self.site, "3.26.3")

    def assert_restore_rejected_without_site_changes(self, digest):
        before = snapshot(self.site)
        with self.assertRaises(ValueError):
            recovery.restore_backup(self.site, self.backup, digest)
        self.assertEqual(snapshot(self.site), before)

    def test_round_trip_restores_exact_vcs_metadata_bytecode_and_record(self):
        digest = self.make_backup()
        self.assertEqual(snapshot(self.site), self.original)
        verified = recovery.verify_backup(self.backup)
        self.assertIsInstance(verified, dict)
        self.upgrade()
        unrelated_after_backup = self.site / "installed_later.txt"
        unrelated_after_backup.write_bytes(b"keep new unrelated files too")

        report = recovery.restore_backup(self.site, self.backup, digest)

        self.assertIsInstance(report, dict)
        expected = dict(self.original)
        expected["installed_later.txt"] = unrelated_after_backup.read_bytes()
        self.assertEqual(snapshot(self.site), expected)
        self.assertFalse((self.site / "skell_e_router/errors.py").exists())
        self.assertFalse((self.site / "skell_e_router-3.26.3.dist-info").exists())
        self.assertEqual(sha256(self.backup / "manifest.json"), digest)

    def test_restore_is_repeatable(self):
        digest = self.make_backup()
        self.upgrade()
        recovery.restore_backup(self.site, self.backup, digest)
        recovery.restore_backup(self.site, self.backup, digest)
        self.assertEqual(snapshot(self.site), self.original)

    def test_backup_only_contains_owned_files(self):
        self.make_backup()
        copied = snapshot(self.backup / "files")
        self.assertEqual(copied, {
            path: data for path, data in self.original.items()
            if path.startswith(("skell_e_router/", self.metadata.name + "/"))
        })

    def test_wrong_expected_manifest_hash_rejects_before_mutation(self):
        self.make_backup()
        self.upgrade()
        self.assert_restore_rejected_without_site_changes("0" * 64)

    def test_changed_manifest_rejects_before_mutation(self):
        digest = self.make_backup()
        manifest = self.backup / "manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        self.upgrade()
        self.assert_restore_rejected_without_site_changes(digest)

    def test_changed_backup_file_rejects_verification_and_restore(self):
        digest = self.make_backup()
        (self.backup / "files/skell_e_router/utils.py").write_bytes(b"tampered\n")
        with self.assertRaises(ValueError):
            recovery.verify_backup(self.backup)
        self.upgrade()
        self.assert_restore_rejected_without_site_changes(digest)

    def test_missing_backup_file_rejects_before_mutation(self):
        digest = self.make_backup()
        (self.backup / "files" / self.metadata.name / "direct_url.json").unlink()
        self.upgrade()
        self.assert_restore_rejected_without_site_changes(digest)

    def test_unsafe_manifest_paths_reject_even_with_matching_manifest_hash(self):
        self.make_backup()
        self.upgrade()
        manifest_path = self.backup / "manifest.json"
        original_manifest = manifest_path.read_bytes()
        outside = self.root / "outside.py"
        outside.write_bytes(b"external sentinel")
        for malicious_path in (
            "../outside.py",
            "/outside.py",
            "C:/outside.py",
            "skell_e_router/../../outside.py",
            "skell_e_router\\utils.py",
            "skell_e_router/utils.py:alternate_stream",
            "unrelated.py",
        ):
            with self.subTest(path=malicious_path):
                manifest = json.loads(original_manifest)
                manifest["files"][0]["path"] = malicious_path
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                self.assert_restore_rejected_without_site_changes(sha256(manifest_path))
                self.assertEqual(outside.read_bytes(), b"external sentinel")

    def test_unsafe_manifest_roots_reject_before_mutation(self):
        self.make_backup()
        self.upgrade()
        manifest_path = self.backup / "manifest.json"
        original_manifest = manifest_path.read_bytes()
        for roots in (
            ["skell_e_router", "../outside"],
            ["skell_e_router", "unrelated-1.0.dist-info"],
            ["skell_e_router", "skell_e_router-3.26.3.dist-info"],
            ["skell_e_router", self.metadata.name, "unrelated.py"],
        ):
            with self.subTest(roots=roots):
                manifest = json.loads(original_manifest)
                manifest["roots"] = roots
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                self.assert_restore_rejected_without_site_changes(sha256(manifest_path))

    def test_wrong_distribution_name_rejects_backup_without_site_changes(self):
        metadata = self.metadata / "METADATA"
        metadata.write_bytes(b"Name: another-router\nVersion: 3.26.2\n")
        before = snapshot(self.site)
        with self.assertRaises(ValueError):
            recovery.create_backup(self.site, self.backup)
        self.assertEqual(snapshot(self.site), before)

    def test_multiple_router_distributions_reject_backup(self):
        second = self.site / "skell_e_router-3.25.0.dist-info"
        second.mkdir()
        (second / "METADATA").write_bytes(b"Name: skell-e-router\nVersion: 3.25.0\n")
        with self.assertRaises(ValueError):
            recovery.create_backup(self.site, self.backup)

    def test_multiple_installed_router_distributions_reject_restore(self):
        digest = self.make_backup()
        self.upgrade()
        second = self.site / "skell_e_router-0.0.0.dist-info"
        second.mkdir()
        (second / "METADATA").write_bytes(b"Name: skell-e-router\nVersion: 0.0.0\n")
        self.assert_restore_rejected_without_site_changes(digest)

    def test_wrong_installed_distribution_name_rejects_restore(self):
        digest = self.make_backup()
        self.upgrade()
        metadata = self.site / "skell_e_router-3.26.3.dist-info/METADATA"
        metadata.write_bytes(b"Name: another-router\nVersion: 3.26.3\n")
        self.assert_restore_rejected_without_site_changes(digest)

    def test_overlapping_backup_and_site_roots_are_rejected(self):
        for backup in (self.site, self.site / "backup", self.root):
            with self.subTest(backup=backup):
                before = snapshot(self.site)
                with self.assertRaises(ValueError):
                    recovery.create_backup(self.site, backup)
                self.assertEqual(snapshot(self.site), before)

    def test_restore_overlapping_roots_rejects_before_mutation(self):
        digest = self.make_backup()
        before = snapshot(self.site)
        for backup in (self.site, self.site / "backup", self.root):
            with self.subTest(backup=backup):
                with self.assertRaises(ValueError):
                    recovery.restore_backup(self.site, backup, digest)
                self.assertEqual(snapshot(self.site), before)

    def symlink_or_skip(self, link, target):
        try:
            link.symlink_to(target, target_is_directory=target.is_dir())
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"OS does not permit test symlinks: {error}")

    def test_source_package_symlink_rejects_backup(self):
        outside = self.root / "outside.py"
        outside.write_bytes(b"not router content")
        self.symlink_or_skip(self.site / "skell_e_router/linked.py", outside)
        with self.assertRaises(ValueError):
            recovery.create_backup(self.site, self.backup)
        self.assertEqual(outside.read_bytes(), b"not router content")

    def test_backup_symlink_rejects_restore_without_touching_target(self):
        digest = self.make_backup()
        copied = self.backup / "files/skell_e_router/utils.py"
        outside = self.root / "outside.py"
        outside.write_bytes(copied.read_bytes())
        copied.unlink()
        self.symlink_or_skip(copied, outside)
        self.upgrade()
        self.assert_restore_rejected_without_site_changes(digest)
        self.assertEqual(outside.read_bytes(), b"ORIGINAL = True\n")

    def test_installed_router_symlink_rejects_restore(self):
        digest = self.make_backup()
        self.upgrade()
        outside = self.root / "outside.py"
        outside.write_bytes(b"unrelated external file")
        self.symlink_or_skip(self.site / "skell_e_router/linked.py", outside)
        self.assert_restore_rejected_without_site_changes(digest)
        self.assertEqual(outside.read_bytes(), b"unrelated external file")

    @unittest.skipUnless(os.name == "nt", "Windows junction regression")
    def test_package_junction_rejects_backup_without_touching_target(self):
        import _winapi

        outside = self.root / "external-directory"
        outside.mkdir()
        sentinel = outside / "sentinel.txt"
        sentinel.write_bytes(b"external content")
        _winapi.CreateJunction(str(outside), str(self.site / "skell_e_router/junction"))
        with self.assertRaises(ValueError):
            recovery.create_backup(self.site, self.backup)
        self.assertEqual(sentinel.read_bytes(), b"external content")

    @unittest.skipUnless(os.name == "nt", "Windows junction regression")
    def test_installed_package_junction_rejects_restore_without_mutation(self):
        import _winapi

        digest = self.make_backup()
        self.upgrade()
        outside = self.root / "external-directory"
        outside.mkdir()
        sentinel = outside / "sentinel.txt"
        sentinel.write_bytes(b"external content")
        _winapi.CreateJunction(str(outside), str(self.site / "skell_e_router/junction"))
        self.assert_restore_rejected_without_site_changes(digest)
        self.assertEqual(sentinel.read_bytes(), b"external content")

    def test_recovers_when_installer_removed_both_router_roots(self):
        digest = self.make_backup()
        shutil.rmtree(self.site / "skell_e_router")
        shutil.rmtree(self.metadata)
        recovery.restore_backup(self.site, self.backup, digest, replace_version="3.26.3")
        self.assertEqual(snapshot(self.site), self.original)

    def test_recovers_with_both_named_distribution_directories(self):
        digest = self.make_backup()
        write_distribution(self.site, "3.26.3")
        recovery.restore_backup(self.site, self.backup, digest, replace_version="3.26.3")
        self.assertEqual(snapshot(self.site), self.original)

    def test_recovers_named_upgrade_missing_metadata_file(self):
        digest = self.make_backup()
        self.upgrade()
        (self.site / "skell_e_router-3.26.3.dist-info/METADATA").unlink()
        recovery.restore_backup(self.site, self.backup, digest, replace_version="3.26.3")
        self.assertEqual(snapshot(self.site), self.original)

    def test_partial_recovery_rejects_an_unexpected_third_version(self):
        digest = self.make_backup()
        write_distribution(self.site, "9.9.9")
        before = snapshot(self.site)
        with self.assertRaises(ValueError):
            recovery.restore_backup(self.site, self.backup, digest, replace_version="3.26.3")
        self.assertEqual(snapshot(self.site), before)


if __name__ == "__main__":
    unittest.main()
