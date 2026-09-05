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


class _SafeIterator:
    """Forward iteration and explicit close without closing on temporary access."""

    def __init__(self, iterator, error_type, code, details):
        self._error_type, self._code, self._details = error_type, code, details
        self._iterator = self._call(lambda: iter(iterator))

    def _call(self, call):
        return call_provider(call, self._error_type, code=self._code, details=self._details)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            raise
        except Exception as exc:
            error = provider_error(exc, self._error_type, code=self._code, details=self._details)
        raise error from None

    def close(self):
        close = self._call(lambda: getattr(self._iterator, "close", None))
        if close is not None:
            return self._call(close)


def safe_iterator(iterator, error_type, *, code="PROVIDER_ERROR", details=None):
    """Protect failures occurring after a streaming API has returned to its caller."""
    return _SafeIterator(iterator, error_type, code, details)


class SafeStream:
    """Protect Anthropic stream entry, iteration, properties, methods and exit."""

    def __init__(self, stream, error_type, *, details=None):
        self._stream = stream
        self._error_type = error_type
        self._details = details
        self._events = None
        self._text = None

    def _call(self, call):
        return call_provider(call, self._error_type, details=self._details)

    def __enter__(self):
        stream = self._call(lambda: self._stream.__enter__())
        return SafeStream(stream, self._error_type, details=self._details)

    def __exit__(self, exc_type, exc, tb):
        return self._call(lambda: self._stream.__exit__(exc_type, exc, tb))

    def __iter__(self):
        if self._events is None:
            self._events = safe_iterator(self._stream, self._error_type, details=self._details)
        return self._events

    def __next__(self):
        return next(iter(self))

    def __getattr__(self, name):
        if name == "text_stream":
            if self._text is None:
                value = self._call(lambda: getattr(self._stream, name))
                self._text = safe_iterator(value, self._error_type, details=self._details)
            return self._text
        value = self._call(lambda: getattr(self._stream, name))
        if callable(value):
            return lambda *args, **kwargs: self._call(lambda: value(*args, **kwargs))
        return value


def _redact_keys(message, config):
    """Legacy private helper. Provider error boundaries must not rely on this."""
    for value in (config or {}).values():
        if isinstance(value, str) and value:
            message = message.replace(value, "[REDACTED]")
    return message
