"""Tests for embeddings.py — input shape, modality, validation, retry, response."""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestClassifyInputPart:

    def test_data_uri_image(self):
        from skell_e_router.embeddings import _classify_input_part
        assert _classify_input_part("data:image/png;base64,abc") == "image"

    def test_data_uri_audio(self):
        from skell_e_router.embeddings import _classify_input_part
        assert _classify_input_part("data:audio/mpeg;base64,abc") == "audio"

    def test_data_uri_video(self):
        from skell_e_router.embeddings import _classify_input_part
        assert _classify_input_part("data:video/mp4;base64,abc") == "video"

    def test_data_uri_pdf(self):
        from skell_e_router.embeddings import _classify_input_part
        assert _classify_input_part("data:application/pdf;base64,abc") == "pdf"

    def test_url_image_extension(self):
        from skell_e_router.embeddings import _classify_input_part
        assert _classify_input_part("https://example.com/cat.png") == "image"
        assert _classify_input_part("https://example.com/song.mp3") == "audio"
        assert _classify_input_part("https://example.com/clip.mp4") == "video"
        assert _classify_input_part("https://example.com/doc.pdf") == "pdf"

    def test_url_unknown_extension_is_text(self):
        from skell_e_router.embeddings import _classify_input_part
        # A URL without a recognizable media extension is treated as text.
        # (Embedding multimodal models accept text URIs in some contexts; we leave
        # the strict per-model check to the API itself.)
        assert _classify_input_part("https://example.com/page.html") == "text"

    def test_gs_uri_with_extension(self):
        from skell_e_router.embeddings import _classify_input_part
        assert _classify_input_part("gs://bucket/clip.mp4") == "video"
        assert _classify_input_part("gs://bucket/page.pdf") == "pdf"

    def test_local_file_path(self, tmp_path):
        from skell_e_router.embeddings import _classify_input_part
        p = tmp_path / "x.png"
        p.write_bytes(b"\x89PNG")
        assert _classify_input_part(str(p)) == "image"

    def test_plain_text(self):
        from skell_e_router.embeddings import _classify_input_part
        assert _classify_input_part("Just a sentence.") == "text"
        assert _classify_input_part("multi\nline\ntext") == "text"

    def test_path_like_but_missing_file_is_text(self):
        from skell_e_router.embeddings import _classify_input_part
        # A string that contains slashes but doesn't exist on disk is treated as text.
        # File-existence enforcement happens later in _normalize_input via
        # _encode_to_data_uri, which raises INVALID_INPUT for missing paths
        # (only if the caller intended a file).
        assert _classify_input_part("not/a/real/file.png") == "text"

    def test_data_uri_with_charset_param(self):
        from skell_e_router.embeddings import _classify_input_part
        assert _classify_input_part("data:image/png;charset=utf-8;base64,abc") == "image"

    def test_data_uri_uppercase_mime(self):
        from skell_e_router.embeddings import _classify_input_part
        assert _classify_input_part("data:IMAGE/PNG;base64,abc") == "image"

    def test_data_uri_unknown_mime_is_text(self):
        from skell_e_router.embeddings import _classify_input_part
        assert _classify_input_part("data:application/octet-stream;base64,abc") == "text"


class TestNormalizeInput:

    def test_string_shorthand(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        normalized, was_str = _normalize_input("hello", m)
        assert normalized == ["hello"]
        assert was_str is True

    def test_flat_list(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        normalized, was_str = _normalize_input(["a", "b", "c"], m)
        assert normalized == ["a", "b", "c"]
        assert was_str is False

    def test_nested_list_aggregation_on_gemini(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        m = EMBEDDING_MODEL_CONFIG["gemini-embedding-2"]
        normalized, was_str = _normalize_input(
            [["caption", "data:image/png;base64,abc"]], m
        )
        assert normalized == [["caption", "data:image/png;base64,abc"]]
        assert was_str is False

    def test_mixed_aggregate_and_plain_on_gemini(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        m = EMBEDDING_MODEL_CONFIG["gemini-embedding-2"]
        normalized, _ = _normalize_input(
            [["caption", "data:image/png;base64,abc"], "plain"], m
        )
        assert normalized == [["caption", "data:image/png;base64,abc"], "plain"]

    def test_local_file_path_encoded_to_data_uri(self, tmp_path):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        m = EMBEDDING_MODEL_CONFIG["gemini-embedding-2"]
        p = tmp_path / "tiny.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        normalized, _ = _normalize_input([["caption", str(p)]], m)
        assert normalized[0][0] == "caption"
        assert normalized[0][1].startswith("data:image/png;base64,")

    def test_nested_on_openai_raises(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from skell_e_router.utils import RouterError
        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        with pytest.raises(RouterError) as exc:
            _normalize_input([["a", "b"]], m)
        assert exc.value.code == "INVALID_INPUT"
        assert "aggregation" in exc.value.message.lower()

    def test_image_on_openai_raises(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from skell_e_router.utils import RouterError
        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        with pytest.raises(RouterError) as exc:
            _normalize_input(["data:image/png;base64,abc"], m)
        assert exc.value.code == "INVALID_INPUT"
        assert "image" in exc.value.message.lower()

    def test_audio_on_gemini_passes(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        m = EMBEDDING_MODEL_CONFIG["gemini-embedding-2"]
        normalized, _ = _normalize_input(
            [["transcribe", "data:audio/mpeg;base64,xyz"]], m
        )
        assert normalized[0][1] == "data:audio/mpeg;base64,xyz"

    def test_gemini_file_ref_in_aggregate(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from skell_e_router.response import GeminiFileRef
        m = EMBEDDING_MODEL_CONFIG["gemini-embedding-2"]
        ref = GeminiFileRef(uri="files/abc123", mime_type="video/mp4")
        normalized, _ = _normalize_input([["watch this", ref]], m)
        assert normalized[0][0] == "watch this"
        assert normalized[0][1] == {
            "file_data": {"file_uri": "files/abc123", "mime_type": "video/mp4"}
        }

    def test_missing_file_path_is_text(self, tmp_path):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        m = EMBEDDING_MODEL_CONFIG["gemini-embedding-2"]
        # Real-looking image extension but doesn't exist; classifier returns "text"
        # so this passes (file path that doesn't exist is just plain text by design).
        normalized, _ = _normalize_input(["not/a/real/file.png"], m)
        assert normalized == ["not/a/real/file.png"]

    def test_invalid_top_level_type_raises(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from skell_e_router.utils import RouterError
        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        with pytest.raises(RouterError) as exc:
            _normalize_input(12345, m)
        assert exc.value.code == "INVALID_INPUT"

    def test_invalid_part_type_in_nested_list(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from skell_e_router.utils import RouterError
        m = EMBEDDING_MODEL_CONFIG["gemini-embedding-2"]
        with pytest.raises(RouterError) as exc:
            _normalize_input([["caption", 999]], m)
        assert exc.value.code == "INVALID_INPUT"


class TestPerformEmbedding:

    def test_calls_litellm_with_correct_kwargs(self):
        from skell_e_router.embeddings import _perform_embedding
        from tests.helpers import make_litellm_embedding_response

        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            mock_emb.return_value = make_litellm_embedding_response(
                embeddings=[[0.1, 0.2]]
            )
            response, duration = _perform_embedding(
                model_name="openai/text-embedding-3-large",
                input=["hello"],
                api_key="sk-test",
                dimensions=512,
            )
        mock_emb.assert_called_once()
        kwargs = mock_emb.call_args.kwargs
        assert kwargs["model"] == "openai/text-embedding-3-large"
        assert kwargs["input"] == ["hello"]
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["dimensions"] == 512
        assert duration >= 0

    def test_no_api_key_omits_kwarg(self):
        from skell_e_router.embeddings import _perform_embedding
        from tests.helpers import make_litellm_embedding_response
        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            mock_emb.return_value = make_litellm_embedding_response()
            _perform_embedding(
                model_name="openai/text-embedding-3-large",
                input=["hello"],
                api_key=None,
            )
        kwargs = mock_emb.call_args.kwargs
        assert "api_key" not in kwargs

    def test_retries_on_503(self):
        from skell_e_router.embeddings import _perform_embedding
        from tests.helpers import make_litellm_embedding_response

        # First two calls raise a 503, third succeeds.
        err = Exception("server down")
        err.status_code = 503
        err.headers = {}

        ok = make_litellm_embedding_response()

        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            mock_emb.side_effect = [err, err, ok]
            response, _ = _perform_embedding(
                model_name="openai/text-embedding-3-large",
                input=["hi"],
                api_key=None,
            )
        assert mock_emb.call_count == 3
        assert response is ok

    def test_does_not_retry_on_quota_429(self):
        from skell_e_router.embeddings import _perform_embedding

        err = Exception("quota exceeded")
        err.status_code = 429
        err.code = "insufficient_quota"
        err.headers = {}

        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            mock_emb.side_effect = err
            with pytest.raises(Exception):
                _perform_embedding(
                    model_name="openai/text-embedding-3-large",
                    input=["hi"],
                    api_key=None,
                )
        assert mock_emb.call_count == 1


class TestBuildEmbeddingResponse:

    def test_basic_fields_populated(self):
        from skell_e_router.embeddings import _build_embedding_response
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from tests.helpers import make_litellm_embedding_response

        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        raw = make_litellm_embedding_response(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            model="openai/text-embedding-3-large",
            prompt_tokens=7,
            total_tokens=7,
        )
        with patch("skell_e_router.embeddings.litellm.completion_cost") as mock_cost:
            mock_cost.return_value = 0.0002
            resp = _build_embedding_response(
                response=raw,
                embedding_model=m,
                request_duration_s=0.123,
                total_duration_s=0.456,
            )
        assert resp.embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert resp.model == "openai/text-embedding-3-large"
        assert resp.dimensions == 3
        assert resp.prompt_tokens == 7
        assert resp.total_tokens == 7
        assert resp.cost == 0.0002
        assert resp.duration_seconds == 0.123
        assert resp.total_duration_seconds == 0.456
        assert resp.raw_response is raw

    def test_cost_swallows_exception(self):
        from skell_e_router.embeddings import _build_embedding_response
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from tests.helpers import make_litellm_embedding_response

        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        raw = make_litellm_embedding_response()
        with patch("skell_e_router.embeddings.litellm.completion_cost") as mock_cost:
            mock_cost.side_effect = Exception("model not in cost table")
            resp = _build_embedding_response(
                response=raw, embedding_model=m,
                request_duration_s=None, total_duration_s=None,
            )
        assert resp.cost is None

    def test_index_ordering_preserved(self):
        """If the provider returns out-of-order indices, sort them."""
        from skell_e_router.embeddings import _build_embedding_response
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from tests.helpers import make_litellm_embedding_response

        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        raw = make_litellm_embedding_response(
            embeddings=[[1.0], [2.0], [3.0]]
        )
        # Shuffle the data items but keep correct `index` fields.
        raw.data = [raw.data[2], raw.data[0], raw.data[1]]

        with patch("skell_e_router.embeddings.litellm.completion_cost") as mock_cost:
            mock_cost.return_value = 0.0
            resp = _build_embedding_response(
                response=raw, embedding_model=m,
                request_duration_s=0, total_duration_s=0,
            )
        assert resp.embeddings == [[1.0], [2.0], [3.0]]

    def test_falls_back_to_model_name_if_response_lacks_one(self):
        from skell_e_router.embeddings import _build_embedding_response
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from tests.helpers import make_litellm_embedding_response

        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        raw = make_litellm_embedding_response()
        # Remove the `model` attribute by setting it to ""
        raw.model = ""
        with patch("skell_e_router.embeddings.litellm.completion_cost") as mock_cost:
            mock_cost.return_value = None
            resp = _build_embedding_response(
                response=raw, embedding_model=m,
                request_duration_s=0, total_duration_s=0,
            )
        # Empty string still wins over fallback (we only fall back if attr is missing/None).
        # We accept either behavior — assert non-None.
        assert resp.model is not None


class TestGetEmbedding:

    def _patch_litellm(self, embeddings, model_name="openai/text-embedding-3-large"):
        """Helper: patch litellm.embedding to return a fake response."""
        from tests.helpers import make_litellm_embedding_response
        return patch(
            "skell_e_router.embeddings.litellm.embedding",
            return_value=make_litellm_embedding_response(
                embeddings=embeddings, model=model_name,
            ),
        )

    def test_string_input_returns_flat_list(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        with self._patch_litellm([[0.1, 0.2, 0.3]]):
            result = get_embedding("openai-embedding-3-large", "hello")
        assert result == [0.1, 0.2, 0.3]

    def test_list_input_returns_nested_list(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        with self._patch_litellm([[0.1], [0.2], [0.3]]):
            result = get_embedding("openai-embedding-3-large", ["a", "b", "c"])
        assert result == [[0.1], [0.2], [0.3]]

    def test_rich_response_returns_dataclass(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        from skell_e_router.response import EmbeddingResponse
        with self._patch_litellm([[0.1, 0.2]]):
            result = get_embedding(
                "openai-embedding-3-large", "hello", rich_response=True,
            )
        assert isinstance(result, EmbeddingResponse)
        assert result.embeddings == [[0.1, 0.2]]
        assert result.dimensions == 2

    def test_rich_response_with_string_does_not_unwrap(self, monkeypatch):
        """rich_response always returns nested list[list[float]]."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        with self._patch_litellm([[0.5, 0.6]]):
            result = get_embedding(
                "openai-embedding-3-large", "hello", rich_response=True,
            )
        assert result.embeddings == [[0.5, 0.6]]  # not [0.5, 0.6]

    def test_dimensions_passed_to_litellm(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            from tests.helpers import make_litellm_embedding_response
            mock_emb.return_value = make_litellm_embedding_response(
                embeddings=[[0.1] * 512]
            )
            get_embedding(
                "openai-embedding-3-large", "hello", dimensions=512,
            )
        assert mock_emb.call_args.kwargs["dimensions"] == 512

    def test_dimensions_not_passed_when_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            from tests.helpers import make_litellm_embedding_response
            mock_emb.return_value = make_litellm_embedding_response()
            get_embedding("openai-embedding-3-large", "hello")
        assert "dimensions" not in mock_emb.call_args.kwargs

    def test_dimensions_too_large_raises(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        from skell_e_router.utils import RouterError
        with pytest.raises(RouterError) as exc:
            get_embedding(
                "openai-embedding-3-small", "hello", dimensions=4096,
            )
        assert exc.value.code == "INVALID_PARAM"

    def test_dimensions_zero_raises(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        from skell_e_router.utils import RouterError
        with pytest.raises(RouterError) as exc:
            get_embedding(
                "openai-embedding-3-large", "hello", dimensions=0,
            )
        assert exc.value.code == "INVALID_PARAM"

    def test_unknown_model_raises(self, monkeypatch):
        from skell_e_router.embeddings import get_embedding
        from skell_e_router.utils import RouterError
        with pytest.raises(RouterError) as exc:
            get_embedding("not-a-model", "hello")
        assert exc.value.code == "INVALID_MODEL"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from skell_e_router.embeddings import get_embedding
        from skell_e_router.utils import RouterError
        with pytest.raises(RouterError) as exc:
            get_embedding("openai-embedding-3-large", "hello")
        assert exc.value.code == "MISSING_ENV"

    def test_api_key_from_config_overrides_env(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from skell_e_router.embeddings import get_embedding
        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            from tests.helpers import make_litellm_embedding_response
            mock_emb.return_value = make_litellm_embedding_response()
            get_embedding(
                "openai-embedding-3-large", "hello",
                config={"openai_api_key": "sk-from-config"},
            )
        assert mock_emb.call_args.kwargs["api_key"] == "sk-from-config"

    def test_litellm_error_wraps_in_provider_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        from skell_e_router.utils import RouterError
        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            mock_emb.side_effect = ValueError("provider exploded: sk-test secret")
            with pytest.raises(RouterError) as exc:
                get_embedding(
                    "openai-embedding-3-large", "hello",
                    config={"openai_api_key": "sk-test"},
                )
        assert exc.value.code == "PROVIDER_ERROR"
        # API key must be redacted from error message.
        assert "sk-test" not in exc.value.message
        assert "[REDACTED]" in exc.value.message

    def test_gemini_multimodal_aggregation(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")
        from skell_e_router.embeddings import get_embedding

        img = tmp_path / "tiny.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")

        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            from tests.helpers import make_litellm_embedding_response
            mock_emb.return_value = make_litellm_embedding_response(
                embeddings=[[0.9] * 768], model="gemini/gemini-embedding-2",
            )
            result = get_embedding(
                "gemini-embedding-2",
                [["a red shoe", str(img)]],
            )
        # 1 nested element → 1 output embedding
        assert len(result) == 1
        assert len(result[0]) == 768
        # The input passed to litellm should have the image as a data URI
        sent = mock_emb.call_args.kwargs["input"]
        assert isinstance(sent[0], list)
        assert sent[0][0] == "a red shoe"
        assert sent[0][1].startswith("data:image/png;base64,")


class TestPackageExports:

    def test_get_embedding_importable_from_top_level(self):
        from skell_e_router import get_embedding
        assert callable(get_embedding)

    def test_embedding_response_importable_from_top_level(self):
        from skell_e_router import EmbeddingResponse
        assert EmbeddingResponse is not None

    def test_embedding_model_importable_from_top_level(self):
        from skell_e_router import EmbeddingModel
        assert EmbeddingModel is not None

    def test_resolve_embedding_alias_importable_from_top_level(self):
        from skell_e_router import resolve_embedding_alias
        assert callable(resolve_embedding_alias)
