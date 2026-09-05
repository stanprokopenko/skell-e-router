"""Check actual session-search errors with the installed router and denied network.

Controlled index boundary: only VECTORS/CHUNKS point to a miniature fixture.
All fixtures remain for inspection except the synthetic credential, removed
between cases. Version 3.26.2 is a compatibility check, not rollout acceptance.
"""
import argparse
import ast
import hashlib
from importlib import metadata
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MARKER = "synthetic-installed-search-negative-only"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", default="3.26.3")
    parser.add_argument("--python", type=Path, default=Path(sys.base_prefix) / "python.exe")
    parser.add_argument("--expected-site", type=Path)
    parser.add_argument("--search", type=Path, default=ROOT.parent / "claude-orchestrator/scripts/session-digest/search.py")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=["credential", "missing"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.python = args.python.resolve()
    args.expected_site = (args.expected_site or args.python.parent / "Lib/site-packages").resolve()
    args.search = args.search.resolve()
    return args


def child_environment(fixture, python):
    return {"SystemRoot": r"C:\Windows", "PATH": str(python.parent),
                       **{name: str(fixture) for name in ("HOME", "USERPROFILE", "TEMP", "TMP")},
                       "APPDATA": str(fixture / "appdata"), "PYTHON_DOTENV_DISABLED": "1",
                       "PYTHONUSERBASE": site.getuserbase(),
                       "LITELLM_LOCAL_MODEL_COST_MAP": "True", "LITELLM_TELEMETRY": "False",
                       "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
                       "ALL_PROXY": "http://127.0.0.1:9", "NO_PROXY": ""}


def child(args):
    # Startup has selected user-site/system SDK precedence. Keep that order,
    # removing only checkout entries; clear credentials before any SDK imports.
    environment = child_environment(args.child, args.python)
    os.environ.clear()
    os.environ.update(environment)
    sys.path[:] = [p for p in sys.path if p and not any(
        Path(p).resolve().is_relative_to(root) for root in (ROOT, args.search.parents[2]))]
    os.chdir(args.child)
    import socket
    import threading
    state = threading.local()
    original_connect, original_pair = socket.socket.connect, socket.socketpair

    def deny(*unused, **kwargs):
        raise OSError("Network disabled for installed search verification")

    def connect(sock, address):
        if getattr(state, "active", False) and isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return original_connect(sock, address)
        return deny()

    def pair(*positional, **kwargs):
        state.active = True
        try:
            return original_pair(*positional, **kwargs)
        finally:
            state.active = False

    socket.socketpair, socket.socket.connect = pair, connect
    socket.socket.connect_ex = socket.socket.sendto = deny
    socket.create_connection = socket.getaddrinfo = deny
    assert Path(sys.executable).resolve() == args.python, "Unexpected interpreter"
    spec = PathFinder.find_spec("skell_e_router", sys.path)
    expected = args.expected_site / "skell_e_router/__init__.py"
    assert spec and Path(spec.origin).resolve() == expected, "Unexpected router path"
    dist = metadata.distribution("skell-e-router")
    assert Path(dist.locate_file("")).resolve() == args.expected_site, "Unexpected metadata path"
    tree = ast.parse(expected.read_text(encoding="utf-8-sig"))
    version = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets))
    assert version == dist.version == args.expected_version, "Router code/metadata version mismatch"
    sdk_origins = {name: PathFinder.find_spec(name, sys.path).origin
                   for name in ("anthropic", "requests", "openai", "litellm")}
    assert all(any(Path(origin).resolve().is_relative_to(Path(root).resolve())
                   for root in (args.expected_site, site.getusersitepackages())) for origin in sdk_origins.values())
    details = {"python": sys.executable, "router_origin": spec.origin, "code_version": version,
               "metadata_version": dist.version, "metadata_root": str(dist.locate_file("")),
               "search_sha256": hashlib.sha256(args.search.read_bytes()).hexdigest(),
               "selected_sdk_metadata_versions": {name: metadata.version(name) for name in ("anthropic", "requests", "openai", "litellm")}}
    import runpy
    import numpy as np
    vectors, chunks = args.child / "vectors.npy", args.child / "chunks.json"
    if not vectors.exists():
        np.save(vectors, np.ones((1, 1536), dtype=np.float32))
        chunks.write_text(json.dumps({"model": "openai-embedding-3-small", "dimensions": 1536,
                                     "chunks": [{"text": "synthetic fixture"}]}), encoding="utf-8")
    module = runpy.run_path(str(args.search), run_name="installed_search_verification")
    # runpy's returned mapping may differ from function globals. Patch those
    # mappings explicitly; leave credential resolution and router calls intact.
    for name in ("cmd_query", "main"):
        module[name].__globals__.update(VECTORS=vectors, CHUNKS=chunks)
    sys.argv = [str(args.search), "query", "-n", "3", "--json", "--", "anything"]
    try:
        module["main"]()
    finally:
        router = sys.modules.get("skell_e_router")
        details["router_imported"] = router is not None
        if args.case == "credential":
            assert router is not None, "Search did not import the actual router"
            assert Path(router.__file__).resolve() == expected, "Imported router path mismatch"
            assert router.__version__ == args.expected_version, "Imported router version mismatch"
            import inspect
            details["get_embedding_signature"] = str(inspect.signature(router.get_embedding))
            assert "config" in inspect.signature(router.get_embedding).parameters
        else:
            assert router is None, "Missing credential unexpectedly imported router"
        details["imported_sdk_origins"] = {name: getattr(sys.modules[name], "__file__", None)
                                           for name in ("anthropic", "requests", "openai", "litellm")
                                           if name in sys.modules}
        assert all(Path(origin).resolve() == Path(sdk_origins[name]).resolve()
                   for name, origin in details["imported_sdk_origins"].items()), "SDK import path mismatch"
        (args.child / f"{args.case}-metadata.json").write_text(json.dumps(details, indent=2), encoding="utf-8")


def main(args):
    assert 1 <= args.timeout <= 90, "Timeout must be between 1 and 90 seconds"
    fixture = Path(tempfile.mkdtemp(prefix="installed-search-negative-"))
    key = fixture / "appdata/session-search/openai-embedding.key"
    key.parent.mkdir(parents=True)
    key.write_text(MARKER, encoding="utf-8")
    before = hashlib.sha256(args.search.read_bytes()).hexdigest()
    revision = subprocess.run(["git", "-C", str(args.search.parents[2]), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    report = {"boundary": "Actual search.py and installed router; only VECTORS/CHUNKS use a synthetic index",
              "acceptance": "compatibility-only" if args.expected_version == "3.26.2" else "installed-version-check",
              "fixture": str(fixture), "search": str(args.search), "search_revision": revision,
              "search_sha256": before, "expected_version": args.expected_version, "cases": []}
    for case, prefix in (("credential", "session-search: embedding provider failed"),
                         ("missing", "session-search: missing embedding credential")):
        if case == "missing":
            assert key.resolve().is_relative_to(fixture.resolve())
            key.unlink()  # Only this run's synthetic key; retain every other fixture.
        command = [str(args.python), "-B", str(Path(__file__).resolve()), "--child", str(fixture),
                   "--case", case, "--python", str(args.python), "--expected-site", str(args.expected_site),
                   "--expected-version", args.expected_version, "--search", str(args.search)]
        try:
            # Public user-base path preserves normal SDK priority without
            # inheriting credentials or a real profile directory at startup.
            result = subprocess.run(command, cwd=fixture, env=child_environment(fixture, args.python),
                                    capture_output=True, timeout=args.timeout,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            lines = result.stderr.decode("utf-8", errors="replace").splitlines()
            passed = (result.returncode == 1 and result.stdout == b"" and len(lines) == 1
                      and lines[0].startswith(prefix) and b"Traceback" not in result.stderr
                      and MARKER.encode() not in result.stdout + result.stderr)
            item = {"case": case, "passed": passed, "exit_code": result.returncode,
                    "stdout_bytes": len(result.stdout), "stderr_lines": len(lines),
                    "diagnostic": lines[0] if passed else "Unexpected output withheld"}
        except subprocess.TimeoutExpired:
            item = {"case": case, "passed": False, "timeout_seconds": args.timeout}
        meta = fixture / f"{case}-metadata.json"
        item["metadata"] = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else None
        item["passed"] = item["passed"] and item["metadata"] is not None
        report["cases"].append(item)
    report["search_unchanged"] = hashlib.sha256(args.search.read_bytes()).hexdigest() == before
    report["passed"] = report["search_unchanged"] and all(c["passed"] for c in report["cases"])
    output = args.output or fixture / "report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(output.resolve()), "fixture": str(fixture)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.child:
        child(arguments)
    else:
        raise SystemExit(main(arguments))
