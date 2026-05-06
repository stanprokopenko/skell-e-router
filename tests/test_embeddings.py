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
