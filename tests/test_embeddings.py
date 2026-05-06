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
