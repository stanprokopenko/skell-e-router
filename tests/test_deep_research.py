"""Tests for gemini_deep_research.py — data classes, citations, helpers."""

import pytest
from unittest.mock import patch, MagicMock

from skell_e_router.gemini_deep_research import (
    DeepResearchError,
    DeepResearchUsage,
    DeepResearchResult,
    DeepResearchConfig,
    ParsedCitation,
    _check_genai_available,
    _check_api_key,
    _redact_keys,
    _extract_usage,
    _extract_citations,
    _extract_text,
    _build_result,
    _extract_citations_from_text,
    _is_blocked_page_title,
    _rebuild_sources_section,
    _resolve_citation_urls,
    _is_retryable_error,
    process_citations,
    citations_to_dict,
    result_to_dict,
    RETRYABLE_ERRORS,
    DEFAULT_AGENT,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_TIMEOUT,
)

from tests.helpers import make_interaction


# ---------------------------------------------------------------------------
# DeepResearchError
# ---------------------------------------------------------------------------

class TestDeepResearchError:

    def test_creation(self):
        err = DeepResearchError(code="TIMEOUT", message="timed out")
        assert err.code == "TIMEOUT"
        assert err.message == "timed out"
        assert err.details == {}

    def test_with_details(self):
        err = DeepResearchError(code="ERR", message="msg", details={"id": "123"})
        assert err.details == {"id": "123"}

    def test_str_format(self):
        err = DeepResearchError(code="CODE", message="description")
        assert "CODE" in str(err)
        assert "description" in str(err)

    def test_is_exception(self):
        assert issubclass(DeepResearchError, Exception)


# ---------------------------------------------------------------------------
# DeepResearchUsage
# ---------------------------------------------------------------------------

class TestDeepResearchUsage:

    def test_defaults(self):
        u = DeepResearchUsage()
        assert u.prompt_tokens is None
        assert u.completion_tokens is None
        assert u.total_tokens is None
        assert u.prompt_cost is None
        assert u.completion_cost is None
        assert u.total_cost is None
        assert u.raw_usage == {}

    def test_full_creation(self):
        u = DeepResearchUsage(
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            prompt_cost=0.01,
            completion_cost=0.02,
            total_cost=0.03,
            raw_usage={"custom": "data"},
        )
        assert u.total_tokens == 300
        assert u.total_cost == 0.03
        assert u.raw_usage["custom"] == "data"


# ---------------------------------------------------------------------------
# ParsedCitation
# ---------------------------------------------------------------------------

class TestParsedCitation:

    def test_creation(self):
        c = ParsedCitation(number=1, domain="example.com")
        assert c.number == 1
        assert c.domain == "example.com"
        assert c.url is None
        assert c.title is None
        assert c.redirect_url is None

    def test_full_creation(self):
        c = ParsedCitation(
            number=3,
            domain="reddit.com",
            url="https://reddit.com/r/test",
            title="Test Post",
            redirect_url="https://vertexaisearch.example/redirect",
        )
        assert c.url == "https://reddit.com/r/test"
        assert c.title == "Test Post"

    def test_to_dict(self):
        c = ParsedCitation(number=1, domain="example.com", url="https://example.com", title="Example")
        d = c.to_dict()
        assert d["number"] == 1
        assert d["domain"] == "example.com"
        assert d["url"] == "https://example.com"
        assert d["title"] == "Example"
        assert d["redirect_url"] is None


# ---------------------------------------------------------------------------
# DeepResearchResult
# ---------------------------------------------------------------------------

class TestDeepResearchResult:

    def test_defaults(self):
        r = DeepResearchResult(id="test-id", status="completed")
        assert r.id == "test-id"
        assert r.status == "completed"
        assert r.text is None
        assert r.text_without_sources is None
        assert r.error is None
        assert r.outputs == []
        assert r.citations == []
        assert r.parsed_citations == []
        assert r.usage is None
        assert r.duration_seconds is None
        assert r.raw_interaction is None

    def test_to_dict_minimal(self):
        r = DeepResearchResult(id="abc", status="completed")
        d = r.to_dict()
        assert d["id"] == "abc"
        assert d["status"] == "completed"
        assert d["text"] is None
        assert d["citations"] == []
        assert d["usage"] is None

    def test_to_dict_with_citations(self):
        r = DeepResearchResult(
            id="abc",
            status="completed",
            text="Report text",
            parsed_citations=[
                ParsedCitation(number=1, domain="example.com", url="https://example.com"),
            ],
            usage=DeepResearchUsage(prompt_tokens=10, total_tokens=50),
            duration_seconds=30.5,
        )
        d = r.to_dict()
        assert len(d["citations"]) == 1
        assert d["citations"][0]["domain"] == "example.com"
        assert d["usage"]["prompt_tokens"] == 10
        assert d["duration_seconds"] == 30.5

    def test_to_dict_with_usage(self):
        r = DeepResearchResult(
            id="x",
            status="completed",
            usage=DeepResearchUsage(
                prompt_tokens=100,
                completion_tokens=200,
                total_tokens=300,
                total_cost=0.05,
            ),
        )
        d = r.to_dict()
        assert d["usage"]["total_tokens"] == 300
        assert d["usage"]["total_cost"] == 0.05


# ---------------------------------------------------------------------------
# DeepResearchConfig
# ---------------------------------------------------------------------------

class TestDeepResearchConfig:

    def test_defaults(self):
        c = DeepResearchConfig()
        assert c.agent == "deep-research-pro-preview-12-2025"
        assert c.background is True
        assert c.stream is False
        assert c.thinking_summaries == "auto"
        assert c.tools is None

    def test_custom_config(self):
        c = DeepResearchConfig(agent="custom-agent", stream=True, tools=[{"type": "file_search"}])
        assert c.agent == "custom-agent"
        assert c.stream is True
        assert c.tools == [{"type": "file_search"}]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:

    def test_default_agent(self):
        assert "deep-research" in DEFAULT_AGENT

    def test_default_poll_interval(self):
        assert DEFAULT_POLL_INTERVAL == 10.0

    def test_default_timeout(self):
        assert DEFAULT_TIMEOUT == 3600.0

    def test_retryable_errors_not_empty(self):
        assert len(RETRYABLE_ERRORS) > 0
        assert "gateway_timeout" in RETRYABLE_ERRORS
        assert "timeout" in RETRYABLE_ERRORS


# ---------------------------------------------------------------------------
# _check_genai_available
# ---------------------------------------------------------------------------

class TestCheckGenaiAvailable:

    @patch("skell_e_router.gemini_deep_research.GENAI_AVAILABLE", True)
    def test_passes_when_available(self):
        _check_genai_available()  # should not raise

    @patch("skell_e_router.gemini_deep_research.GENAI_AVAILABLE", False)
    def test_raises_when_unavailable(self):
        with pytest.raises(DeepResearchError) as exc_info:
            _check_genai_available()
        assert exc_info.value.code == "MISSING_DEPENDENCY"


# ---------------------------------------------------------------------------
# _check_api_key
# ---------------------------------------------------------------------------

class TestCheckApiKey:

    def test_passes_with_config(self):
        _check_api_key(config={"gemini_api_key": "key"})

    @patch.dict("os.environ", {"GEMINI_API_KEY": "key"}, clear=False)
    def test_passes_with_env(self):
        _check_api_key(config=None)

    def test_raises_when_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(DeepResearchError) as exc_info:
                _check_api_key(config=None)
            assert exc_info.value.code == "MISSING_API_KEY"


# ---------------------------------------------------------------------------
# _redact_keys (deep research version)
# ---------------------------------------------------------------------------

class TestDeepResearchRedactKeys:

    def test_redacts(self):
        result = _redact_keys("Error with key123", {"gemini_api_key": "key123"})
        assert "key123" not in result
        assert "[REDACTED]" in result

    def test_noop_none_config(self):
        assert _redact_keys("hello", None) == "hello"

    def test_noop_empty_config(self):
        assert _redact_keys("hello", {}) == "hello"


# ---------------------------------------------------------------------------
# _extract_usage
# ---------------------------------------------------------------------------

class TestExtractUsage:

    def test_extracts_from_usage_metadata(self):
        usage_meta = MagicMock(spec=[])
        usage_meta.prompt_token_count = 100
        usage_meta.candidates_token_count = 200
        usage_meta.total_token_count = 300
        interaction = MagicMock()
        interaction.usage_metadata = usage_meta

        usage = _extract_usage(interaction)
        assert usage is not None
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 200
        assert usage.total_tokens == 300

    def test_returns_none_when_no_usage(self):
        interaction = MagicMock()
        interaction.usage_metadata = None
        interaction.usage = None
        assert _extract_usage(interaction) is None

    def test_fallback_to_usage_attr(self):
        usage_data = MagicMock(spec=[])
        usage_data.prompt_token_count = None
        usage_data.prompt_tokens = 50
        usage_data.candidates_token_count = None
        usage_data.completion_tokens = 75
        usage_data.total_token_count = None
        usage_data.total_tokens = 125
        interaction = MagicMock()
        interaction.usage_metadata = None
        interaction.usage = usage_data

        usage = _extract_usage(interaction)
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 75


# ---------------------------------------------------------------------------
# _extract_citations
# ---------------------------------------------------------------------------

class TestExtractCitations:

    def test_from_citations_attr(self):
        interaction = MagicMock()
        interaction.citations = ["cit1", "cit2"]
        assert _extract_citations(interaction) == ["cit1", "cit2"]

    def test_from_outputs(self):
        output = MagicMock()
        output.citations = ["cit_from_output"]
        interaction = MagicMock()
        interaction.citations = None
        interaction.outputs = [output]
        assert _extract_citations(interaction) == ["cit_from_output"]

    def test_empty_when_none(self):
        interaction = MagicMock()
        interaction.citations = None
        interaction.outputs = []
        assert _extract_citations(interaction) == []


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------

class TestExtractText:

    def test_extracts_text_from_last_output(self):
        output = MagicMock()
        output.text = "Final report"
        interaction = MagicMock()
        interaction.outputs = [output]
        assert _extract_text(interaction) == "Final report"

    def test_extracts_content_fallback(self):
        output = MagicMock(spec=[])  # no 'text' attr
        output.content = "Content fallback"
        interaction = MagicMock()
        interaction.outputs = [output]
        assert _extract_text(interaction) == "Content fallback"

    def test_returns_none_when_no_outputs(self):
        interaction = MagicMock()
        interaction.outputs = []
        assert _extract_text(interaction) is None


# ---------------------------------------------------------------------------
# _build_result
# ---------------------------------------------------------------------------

class TestBuildResult:

    def test_builds_from_interaction(self):
        output = MagicMock()
        output.text = "Report text"
        interaction = make_interaction(
            interaction_id="id-456",
            status="completed",
            outputs=[output],
        )
        result = _build_result(interaction, duration_seconds=45.0)
        assert result.id == "id-456"
        assert result.status == "completed"
        assert result.text == "Report text"
        assert result.duration_seconds == 45.0

    def test_builds_failed_result(self):
        interaction = make_interaction(
            interaction_id="id-err",
            status="failed",
            error="Something went wrong",
        )
        result = _build_result(interaction)
        assert result.status == "failed"
        assert result.error == "Something went wrong"


# ---------------------------------------------------------------------------
# _extract_citations_from_text
# ---------------------------------------------------------------------------

class TestExtractCitationsFromText:

    def test_extracts_standard_sources(self):
        text = (
            "This is the report body.\n\n"
            "**Sources:**\n"
            "1. [reddit.com](https://vertexaisearch.example/redirect1)\n"
            "2. [wikipedia.org](https://vertexaisearch.example/redirect2)\n"
        )
        clean_text, citations = _extract_citations_from_text(text)
        assert "Sources" not in clean_text
        assert clean_text.strip() == "This is the report body."
        assert len(citations) == 2
        assert citations[0].number == 1
        assert citations[0].domain == "reddit.com"
        assert citations[0].redirect_url == "https://vertexaisearch.example/redirect1"
        assert citations[1].number == 2
        assert citations[1].domain == "wikipedia.org"

    def test_h2_sources_section(self):
        text = "Body\n\n## Sources\n1. [example.com](https://example.com)\n"
        clean_text, citations = _extract_citations_from_text(text)
        assert len(citations) == 1

    def test_h3_sources_section(self):
        text = "Body\n\n### Sources\n1. [example.com](https://example.com)\n"
        clean_text, citations = _extract_citations_from_text(text)
        assert len(citations) == 1

    def test_plain_sources_section(self):
        text = "Body\n\nSources:\n1. [example.com](https://example.com)\n"
        clean_text, citations = _extract_citations_from_text(text)
        assert len(citations) == 1

    def test_no_sources_section(self):
        text = "Just a report with no sources."
        clean_text, citations = _extract_citations_from_text(text)
        assert clean_text == text
        assert citations == []

    def test_empty_text(self):
        clean_text, citations = _extract_citations_from_text("")
        assert clean_text == ""
        assert citations == []

    def test_none_text(self):
        clean_text, citations = _extract_citations_from_text(None)
        assert clean_text is None
        assert citations == []

    def test_multiple_citations(self):
        text = (
            "Report.\n\n"
            "**Sources:**\n"
            "1. [a.com](https://a.com/1)\n"
            "2. [b.com](https://b.com/2)\n"
            "3. [c.com](https://c.com/3)\n"
            "4. [d.com](https://d.com/4)\n"
            "5. [e.com](https://e.com/5)\n"
        )
        _, citations = _extract_citations_from_text(text)
        assert len(citations) == 5
        assert citations[4].number == 5
        assert citations[4].domain == "e.com"


# ---------------------------------------------------------------------------
# _is_blocked_page_title
# ---------------------------------------------------------------------------

class TestIsBlockedPageTitle:

    @pytest.mark.parametrize("title", [
        "Attention Required! | Cloudflare",
        "Just a moment...",
        "Access Denied",
        "403 Forbidden",
        "404 Not Found",
        "Checking your browser before accessing",
        "Security Check Required",
        "Not Acceptable!",
    ])
    def test_blocked_titles(self, title):
        assert _is_blocked_page_title(title) is True

    @pytest.mark.parametrize("title", [
        "How to Build a Robot - Reddit",
        "Machine Learning - Wikipedia",
        "arxiv.org - Deep Learning Paper",
    ])
    def test_valid_titles(self, title):
        assert _is_blocked_page_title(title) is False

    def test_none_title(self):
        assert _is_blocked_page_title(None) is True

    def test_empty_title(self):
        assert _is_blocked_page_title("") is True


# ---------------------------------------------------------------------------
# _rebuild_sources_section
# ---------------------------------------------------------------------------

class TestRebuildSourcesSection:

    def test_rebuilds_with_resolved_urls(self):
        citations = [
            ParsedCitation(number=1, domain="example.com", url="https://example.com/page"),
            ParsedCitation(number=2, domain="test.org", url="https://test.org/article"),
        ]
        result = _rebuild_sources_section(citations)
        assert "**Sources:**" in result
        assert "1. [example.com](https://example.com/page)" in result
        assert "2. [test.org](https://test.org/article)" in result

    def test_uses_redirect_url_as_fallback(self):
        citations = [
            ParsedCitation(number=1, domain="example.com", redirect_url="https://redirect.example/1"),
        ]
        result = _rebuild_sources_section(citations)
        assert "https://redirect.example/1" in result

    def test_uses_domain_as_last_resort(self):
        citations = [
            ParsedCitation(number=1, domain="example.com"),
        ]
        result = _rebuild_sources_section(citations)
        assert "https://example.com" in result

    def test_empty_citations(self):
        assert _rebuild_sources_section([]) == ""

    def test_sorts_by_number(self):
        citations = [
            ParsedCitation(number=3, domain="c.com", url="https://c.com"),
            ParsedCitation(number=1, domain="a.com", url="https://a.com"),
            ParsedCitation(number=2, domain="b.com", url="https://b.com"),
        ]
        result = _rebuild_sources_section(citations)
        lines = result.strip().split("\n")
        # First line is header, then citations in order
        assert "1. [a.com]" in lines[1]
        assert "2. [b.com]" in lines[2]
        assert "3. [c.com]" in lines[3]


# ---------------------------------------------------------------------------
# _resolve_citation_urls
# ---------------------------------------------------------------------------

class TestResolveCitationUrls:

    @patch("skell_e_router.gemini_deep_research._resolve_redirect_url")
    def test_resolves_vertexaisearch_urls(self, mock_resolve):
        mock_resolve.return_value = ("https://real-url.com/page", "Real Page Title")
        citations = [
            ParsedCitation(number=1, domain="real-url.com", redirect_url="https://vertexaisearch.example/r1"),
        ]
        result = _resolve_citation_urls(citations)
        assert result[0].url == "https://real-url.com/page"
        assert result[0].title == "Real Page Title"

    @patch("skell_e_router.gemini_deep_research._resolve_redirect_url")
    def test_blocked_page_uses_domain_title(self, mock_resolve):
        mock_resolve.return_value = ("https://blocked.com", "Attention Required! | Cloudflare")
        citations = [
            ParsedCitation(number=1, domain="blocked.com", redirect_url="https://vertexaisearch.example/r2"),
        ]
        result = _resolve_citation_urls(citations)
        assert result[0].title == "blocked.com"

    @patch("skell_e_router.gemini_deep_research._resolve_redirect_url")
    def test_failed_resolution_uses_domain_url(self, mock_resolve):
        mock_resolve.return_value = (None, None)
        citations = [
            ParsedCitation(number=1, domain="example.com", redirect_url="https://vertexaisearch.example/r3"),
        ]
        result = _resolve_citation_urls(citations)
        assert result[0].url == "https://example.com"

    def test_skips_non_vertexaisearch_urls(self):
        citations = [
            ParsedCitation(number=1, domain="example.com", redirect_url="https://example.com/direct"),
        ]
        result = _resolve_citation_urls(citations)
        assert result[0].url is None  # Not modified


# ---------------------------------------------------------------------------
# _is_retryable_error
# ---------------------------------------------------------------------------

class TestIsRetryableError:

    @pytest.mark.parametrize("msg", [
        "gateway_timeout: request timed out",
        "Error: deadline_expired",
        "connection_reset by peer",
        "Server unavailable, try later",
        "resource_exhausted",
        "TIMEOUT occurred",
    ])
    def test_retryable_messages(self, msg):
        assert _is_retryable_error(msg) is True

    @pytest.mark.parametrize("msg", [
        "invalid_argument: bad query",
        "authentication_error",
        "permission_denied",
        "not_found",
    ])
    def test_non_retryable_messages(self, msg):
        assert _is_retryable_error(msg) is False


# ---------------------------------------------------------------------------
# process_citations
# ---------------------------------------------------------------------------

class TestProcessCitations:

    def test_processes_and_populates_result(self):
        result = DeepResearchResult(
            id="test",
            status="completed",
            text=(
                "Report body.\n\n"
                "**Sources:**\n"
                "1. [example.com](https://example.com/page)\n"
            ),
        )
        with patch("skell_e_router.gemini_deep_research._resolve_citation_urls") as mock_resolve:
            # Just return citations as-is (no actual HTTP calls)
            mock_resolve.side_effect = lambda c, timeout=5.0: c
            processed = process_citations(result, resolve_urls=True)

        assert processed.text_without_sources == "Report body."
        assert len(processed.parsed_citations) == 1
        assert processed.parsed_citations[0].domain == "example.com"

    def test_skip_url_resolution(self):
        result = DeepResearchResult(
            id="test",
            status="completed",
            text="Body.\n\n**Sources:**\n1. [x.com](https://x.com)\n",
        )
        processed = process_citations(result, resolve_urls=False)
        assert len(processed.parsed_citations) == 1
        # URL should still be None since we didn't resolve
        assert processed.parsed_citations[0].url is None

    def test_no_text_returns_unchanged(self):
        result = DeepResearchResult(id="test", status="completed", text=None)
        processed = process_citations(result)
        assert processed.text is None
        assert processed.parsed_citations == []

    def test_no_sources_section(self):
        result = DeepResearchResult(id="test", status="completed", text="Just text, no sources.")
        processed = process_citations(result, resolve_urls=False)
        assert processed.text_without_sources == "Just text, no sources."
        assert processed.parsed_citations == []


# ---------------------------------------------------------------------------
# citations_to_dict / result_to_dict
# ---------------------------------------------------------------------------

class TestSerializationHelpers:

    def test_citations_to_dict(self):
        citations = [
            ParsedCitation(number=1, domain="a.com", url="https://a.com"),
            ParsedCitation(number=2, domain="b.com"),
        ]
        dicts = citations_to_dict(citations)
        assert len(dicts) == 2
        assert dicts[0]["url"] == "https://a.com"
        assert dicts[1]["url"] is None

    def test_citations_to_dict_empty(self):
        assert citations_to_dict([]) == []

    def test_result_to_dict(self):
        result = DeepResearchResult(id="r1", status="completed", text="text")
        d = result_to_dict(result)
        assert d["id"] == "r1"
        assert d["text"] == "text"
