# Run with python -I from Houston's environment to check the installed package.
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distribution, version
import io
import json
import os
from pathlib import Path
import re
import runpy
import socket
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("orchestrator_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    helper = args.orchestrator_root / "scripts/houston-lead-tldr.py"
    models = args.orchestrator_root / "relay/src/models.js"
    match = re.search(r"export const TLDR_MODEL\s*=\s*['\"]([^'\"]+)['\"]", models.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError("Cannot resolve the configured TLDR model")
    model = match[1]
    captured = []
    output = io.StringIO()
    original_connect = socket.socket.connect

    def deny_network(*args, **kwargs):
        raise AssertionError("Real networking is forbidden in this verification")

    def local_connect(sock, address):
        # Windows asyncio uses a loopback socket pair internally.
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return original_connect(sock, address)
        return deny_network()

    # Install socket guards before importing the provider stack.
    with patch.object(socket.socket, "connect", local_connect), patch.object(socket.socket, "connect_ex", deny_network), patch.object(socket, "create_connection", deny_network), patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test-output-limit-only",
        "OPENAI_API_BASE": "https://api.openai.com/v1",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }):
        import httpx
        import skell_e_router

        def mock_send(client, request, **kwargs):
            body = json.loads(request.content)
            if request.url.path != "/v1/chat/completions":
                raise AssertionError(f"Unexpected endpoint: {request.url.path}")
            captured.append({
                "url": str(request.url),
                "model": body.get("model"),
                "limits": {key: body[key] for key in ("max_tokens", "max_completion_tokens", "max_output_tokens") if key in body},
            })
            return httpx.Response(200, request=request, json={
                "id": "chatcmpl-offline-tldr",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Offline TLDR verification."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
            })

        payload = {"model": model, "system": "Summarize briefly.", "user": "Offline fixture conversation.", "max_tokens": 600}
        with patch.object(httpx.Client, "send", mock_send), patch.object(sys, "stdin", io.StringIO(json.dumps(payload))), contextlib.redirect_stdout(output):
            runpy.run_path(str(helper), run_name="__main__")

        lines = output.getvalue().splitlines()
        result = json.loads(lines[lines.index("---TLDR-JSON---") + 1])
        assert result.get("ok") is True, result
        assert result.get("text") == "Offline TLDR verification.", result
        assert len(captured) == 1, captured
        assert captured[0]["model"] == model, captured
        assert captured[0]["limits"] == {"max_completion_tokens": 600}, captured
        package_path = Path(skell_e_router.__file__).resolve()
        router_checkout = Path(__file__).resolve().parents[1]
        assert not package_path.is_relative_to(router_checkout), "Verification must import the installed package, not this checkout"
        installed = distribution("skell-e-router")
        assert installed.version == skell_e_router.__version__, "Installed metadata and imported code versions differ"
        evidence = {
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.executable,
            "router_version": installed.version,
            "router_import": str(package_path),
            "router_source": json.loads(installed.read_text("direct_url.json") or "{}"),
            "litellm_version": version("litellm"),
            "openai_version": version("openai"),
            "helper": str(helper.resolve()),
            "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
            "models_sha256": hashlib.sha256(models.read_bytes()).hexdigest(),
            "configured_model": model,
            "captured_requests": captured,
            "helper_ok": result["ok"],
            "external_network_blocked": True,
            "live_api_spend_usd": 0,
        }

    rendered = json.dumps(evidence, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
