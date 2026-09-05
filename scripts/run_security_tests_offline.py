"""Run tests with synthetic credentials and blocked sockets before SDK imports.

Use an isolated dependency environment and Python -I. Pass pytest arguments after
this script, or --probe to reproduce the original embedding disclosure.
"""
import os
from pathlib import Path
import socket
import sys
import threading


ROOT = Path(__file__).resolve().parents[1]
os.environ.clear()
os.environ.update({
    "SystemRoot": r"C:\Windows",
    "PATH": str(Path(sys.executable).parent),
    "HOME": str(ROOT / ".security-home"),
    "USERPROFILE": str(ROOT / ".security-home"),
    "APPDATA": str(ROOT / ".security-home" / "appdata"),
    "TEMP": str(ROOT / ".security-home"),
    "TMP": str(ROOT / ".security-home"),
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "LITELLM_TELEMETRY": "False",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
})
Path(os.environ["HOME"]).mkdir(exist_ok=True)


def deny_network(*args, **kwargs):
    raise AssertionError("Network disabled for synthetic security tests")


# Windows asyncio needs a local socket pair for its wakeup pipe. Permit only
# connects made while the standard library constructs that pair on this thread.
_socketpair_state = threading.local()
_original_connect = socket.socket.connect
_original_socketpair = socket.socketpair


def guarded_connect(sock, address):
    if (getattr(_socketpair_state, "active", False) and isinstance(address, tuple)
            and address[0] in {"127.0.0.1", "::1"}):
        return _original_connect(sock, address)
    return deny_network()


def local_socketpair(*args, **kwargs):
    _socketpair_state.active = True
    try:
        return _original_socketpair(*args, **kwargs)
    finally:
        _socketpair_state.active = False


socket.socketpair = local_socketpair
socket.socket.connect = guarded_connect
socket.socket.connect_ex = deny_network
socket.socket.sendto = deny_network
socket.create_connection = deny_network
socket.getaddrinfo = deny_network
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

if sys.argv[1:] == ["--probe"]:
    import contextlib
    import io
    import json
    import traceback
    from unittest.mock import patch
    import h11
    import skell_e_router

    results = []
    for escaped in (False, True):
        marker = "synthetic-router-security-20260905"
        key = marker + ("\x00" if escaped else "")

        def transport(**kwargs):
            assert kwargs["api_key"] == key
            if escaped:
                h11.Request(method=b"GET", target=b"/", headers=[
                    (b"Host", b"synthetic.invalid"),
                    (b"Authorization", ("Bearer " + key).encode()),
                ])
            raise ValueError("Rejected credential " + key)

        output = io.StringIO()
        with patch("skell_e_router.embeddings._perform_embedding", transport):
            with contextlib.redirect_stdout(output):
                try:
                    skell_e_router.get_embedding(
                        "openai-embedding-3-small", "synthetic input",
                        config={"openai_api_key": key}, verbosity="info",
                    )
                except Exception as exc:
                    results.append({
                        "escaped_header": escaped,
                        "final_contains_marker": marker in str(exc),
                        "trace_contains_marker": marker in traceback.format_exc(),
                        "logs_contain_marker": marker in output.getvalue(),
                        "original_cause_attached": exc.__cause__ is not None,
                        "original_context_attached": exc.__context__ is not None,
                    })
        assert results, "Probe transport was not reached"
    print(json.dumps({"version": skell_e_router.__version__,
                      "source": skell_e_router.__file__, "probes": results}, indent=2))
else:
    import pytest
    raise SystemExit(pytest.main(sys.argv[1:] or ["tests", "-q"]))
