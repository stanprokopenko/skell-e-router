"""Controlled provider diagnostics shared by the router's public boundaries."""


_MESSAGES = {
    "authentication": "Provider authentication failed. Check the configured credential.",
    "permission": "Provider permission denied. Check account access.",
    "rate_limit": "Provider rate limit or quota exceeded.",
    "timeout": "Provider request timed out.",
    "connection": "Provider connection failed.",
    "invalid_request": "Provider rejected the request.",
    "unavailable": "Provider service is unavailable.",
    "dependency": "A required provider dependency is unavailable.",
    "provider_error": "Provider request failed.",
}


def _status_code(exc):
    # Provider properties and non-integer values are untrusted too. Never render
    # them, their exception type names, response bodies, headers, or error notes.
    try:
        for obj, field in ((exc, "status_code"), (exc, "code"),
                           (getattr(exc, "response", None), "status_code")):
            value = getattr(obj, field, None)
            if type(value) is int and 100 <= value <= 599:
                return value
    except Exception:
        pass
    return None


def provider_error(exc, error_type, *, code="PROVIDER_ERROR", details=None):
    """Build a fresh error using only fixed categories and an HTTP status number.

    Do not copy the provider exception or call str/repr on it. Replacement of
    known keys cannot cover escaped, truncated, or otherwise encoded secrets.
    """
    status = _status_code(exc)
    name = type(exc).__name__.lower()
    if status == 401 or "authentication" in name:
        category = "authentication"
    elif status == 403 or "permission" in name:
        category = "permission"
    elif status == 429 or "ratelimit" in name:
        category = "rate_limit"
    elif status in (408, 504) or "timeout" in name:
        category = "timeout"
    elif "connection" in name or "connecterror" in name:
        category = "connection"
    elif status is not None and status >= 500:
        category = "unavailable"
    elif status is not None and status >= 400:
        category = "invalid_request"
    elif isinstance(exc, ImportError):
        category = "dependency"
    else:
        category = "provider_error"
    safe_details = dict(details or {})
    safe_details["category"] = category
    if status is not None:
        safe_details["status_code"] = status
    return error_type(code=code, message=_MESSAGES[category], details=safe_details)


def call_provider(call, error_type, *, code="PROVIDER_ERROR", details=None):
    """Raise after leaving except so the raw exception is not even __context__."""
    try:
        return call()
    except Exception as exc:
        error = provider_error(exc, error_type, code=code, details=details)
    raise error from None


def safe_iterator(iterator, error_type, *, code="PROVIDER_ERROR", details=None):
    """Protect failures occurring after a streaming API has returned to its caller."""
    iterator = call_provider(lambda: iter(iterator), error_type, code=code, details=details)
    try:
        while True:
            try:
                item = next(iterator)
            except StopIteration as done:
                return done.value
            except Exception as exc:
                error = provider_error(exc, error_type, code=code, details=details)
            else:
                yield item
                continue
            raise error from None
    finally:
        close = call_provider(lambda: getattr(iterator, "close", None), error_type,
                              code=code, details=details)
        if close is not None:
            call_provider(close, error_type, code=code, details=details)


class SafeStream:
    """Protect Anthropic stream entry, iteration, properties, methods and exit."""

    def __init__(self, stream, error_type, *, details=None):
        self._stream = stream
        self._error_type = error_type
        self._details = details

    def _call(self, call):
        return call_provider(call, self._error_type, details=self._details)

    def __enter__(self):
        stream = self._call(lambda: self._stream.__enter__())
        return SafeStream(stream, self._error_type, details=self._details)

    def __exit__(self, exc_type, exc, tb):
        return self._call(lambda: self._stream.__exit__(exc_type, exc, tb))

    def __iter__(self):
        return safe_iterator(self._stream, self._error_type, details=self._details)

    def __getattr__(self, name):
        value = self._call(lambda: getattr(self._stream, name))
        if name == "text_stream":
            return safe_iterator(value, self._error_type, details=self._details)
        if callable(value):
            return lambda *args, **kwargs: self._call(lambda: value(*args, **kwargs))
        return value


def _redact_keys(message, config):
    """Legacy private helper. Provider error boundaries must not rely on this."""
    for value in (config or {}).values():
        if isinstance(value, str) and value:
            message = message.replace(value, "[REDACTED]")
    return message
