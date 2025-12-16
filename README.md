## Install the package

   ```
   pip install git+https://github.com/stanprokopenko/skell-e-router@main
   ```

## Setting Default Python Encoding to UTF-8 on Windows

This project requires Python to use UTF-8 as its default file encoding. If you encounter `UnicodeDecodeError` errors when running Python scripts (especially related to reading configuration or data files), follow these steps:

1.  **Open PowerShell as Administrator:**
    *   Search for "PowerShell" in the Start menu.
    *   Right-click "Windows PowerShell" and select "Run as administrator".

2.  **Run the following command:**
    This command tells Windows to set the `PYTHONUTF8` environment variable to `1` for your user account, making Python default to UTF-8.

    ```powershell
    [System.Environment]::SetEnvironmentVariable('PYTHONUTF8', '1', 'User')
    ```

3.  **Restart Your Terminal/Computer:**
    *   Close and reopen any open PowerShell or Command Prompt windows.
    *   For the change to be fully recognized by all applications, it's sometimes necessary to log out and log back in, or even restart your computer.

After completing these steps, Python should correctly interpret files using UTF-8 encoding by default. 

## Retry policy

The router has an internal retry up to 3 times before sending the response.

- Retries: only on network/timeout errors, HTTP 500/502/503/504, and 429 without quota/billing exhaustion.
- No retry: bad params (4xx), auth/permission errors, not found, quota/billing 429, policy blocks.
- Backoff: exponential with jitter; if `Retry-After` is present on 429/503, it is honored up to a maximum of 120 seconds. If `Retry-After` exceeds 120 seconds, no retry is attempted and the error is returned.

## Verbosity Settings

- **none**: No output
- **response**: Response content
- **info**: Response content + response info/stats (Model, Finish Reason, Cost, Speed, Prompt Tokens, Completion Tokens, Reasoning Tokens, Total Tokens, Tool Calls, Function Call, Provider Specific Fields, Safety ratings if applicable)
- **debug**: Request details (kwargs, messages) + response content + response info/stats + raw response

## Groq Compound (temporary header)

When routing to Groq Compound models (`groq/groq/compound`, `groq/groq/compound-mini`), the router injects a request header `Groq-Model-Version: latest`. This selects the Compound profile that exposes built‑in tools like `visit_website`.

- This is a temporary shim for LiteLLM. Once LiteLLM forwards this header by default for Groq Compound, remove the injection in `skell_e_router/utils.py` in the `_handle_model_specific_params` function and the extra header params in `skell_e_router/model_config.py`.
- If you prefer pinning to a specific profile, change the injected header value from `latest` to the desired version.