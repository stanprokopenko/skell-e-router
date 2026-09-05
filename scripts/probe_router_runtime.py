"""Print credential-free runtime metadata without importing router or a consumer.

For parent attribution, the consumer parent must launch this script through its
normal Python spawn configuration. A shell invocation proves only that shell.
"""
import os
import sys

# Preserve startup's import path, but no later operation needs inherited keys.
os.environ.clear()

import argparse
import ast
from datetime import datetime, timezone
from importlib import metadata
from importlib.machinery import PathFinder
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--entry-directory", type=Path, required=True)
parser.add_argument("--expected-site", type=Path)
parser.add_argument("--expected-version")
parser.add_argument("--expected-parent", type=int)
args = parser.parse_args()
sys.path[0] = str(args.entry_directory.resolve())
spec = PathFinder.find_spec("skell_e_router", sys.path)
result = {
    "observed_at_utc": datetime.now(timezone.utc).isoformat(),
    "parent_pid": os.getppid(),
    "child_pid": os.getpid(),
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "entry_directory": sys.path[0],
    "router_origin": spec.origin if spec else None,
}
if spec and spec.origin:
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets):
            if isinstance(node.value, ast.Constant):
                result["source_version"] = node.value.value
try:
    dist = metadata.distribution("skell-e-router")
    source = json.loads(dist.read_text("direct_url.json") or "{}")
    result.update(
        distribution_version=dist.version,
        distribution_root=str(dist.locate_file("")),
        vcs_commit=source.get("vcs_info", {}).get("commit_id"),
        archive_hashes=source.get("archive_info", {}).get("hashes", {}),
    )
except metadata.PackageNotFoundError:
    result["distribution_version"] = None

result["sdk_distributions"] = {}
for name in ("anthropic", "requests", "openai", "litellm", "google-genai", "tenacity"):
    try:
        sdk = metadata.distribution(name)
        result["sdk_distributions"][name] = {
            "version": sdk.version,
            "distribution_root": str(sdk.locate_file("")),
        }
    except metadata.PackageNotFoundError:
        result["sdk_distributions"][name] = {"version": None}

checks = {
    "package_found": spec is not None,
    "code_matches_metadata": result.get("source_version") == result["distribution_version"],
}
if args.expected_site:
    checks["expected_package_path"] = bool(spec and spec.origin) and (
        Path(spec.origin).resolve() == args.expected_site.resolve() / "skell_e_router" / "__init__.py")
    checks["expected_metadata_path"] = (
        Path(result.get("distribution_root", ".")).resolve() == args.expected_site.resolve())
if args.expected_version:
    checks["expected_version"] = result.get("source_version") == args.expected_version
if args.expected_parent:
    checks["expected_parent"] = result["parent_pid"] == args.expected_parent
result["checks"] = checks
print(json.dumps(result, indent=2))
raise SystemExit(0 if all(checks.values()) else 1)
