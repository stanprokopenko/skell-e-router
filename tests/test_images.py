"""Tests for images.py — registry, validation, routing, response, cost, retry."""

import base64
import os
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from skell_e_router.images import _perform_image_request, generate_image
from skell_e_router.model_config import IMAGE_CONFIG, ImageModel, resolve_image_alias
from skell_e_router.response import ImageResponse
from skell_e_router.utils import RouterError

from .helpers import FAKE_DEEPINFRA_KEY, FAKE_GEMINI_KEY, FAKE_OPENAI_KEY


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
)
PNG_B64 = base64.b64encode(PNG_BYTES).decode()


@contextmanager
def no_retry_sleep():
    """Drop tenacity's backoff to zero. The wait strategy is bound at decoration
    time, so patching the module-level function has no effect — swap it on the
    Retrying object the decorator built."""
    retrying = _perform_image_request.retry
    original = retrying.wait
    retrying.wait = lambda retry_state: 0
    try:
        yield
    finally:
        retrying.wait = original


def make_image_api_response(
    n: int = 1,
    model: str = "gpt-image-2.5-flare",
    input_tokens: int | None = 20,
    output_tokens: int | None = 1056,
    total_tokens: int | None = 1076,
    text_tokens: int | None = 20,
    image_tokens: int | None = 0,
    usage: bool = True,
):
    """Build a mock that looks like an openai images.generate response."""
    data = []
    for _ in range(n):
        item = MagicMock()
        item.b64_json = PNG_B64
        data.append(item)

    response = MagicMock()
    response.data = data
    response.model = model

    if not usage:
        response.usage = None
        return response

    details = MagicMock()
    details.text_tokens = text_tokens
    details.image_tokens = image_tokens

    usage_obj = MagicMock()
    usage_obj.input_tokens = input_tokens
    usage_obj.output_tokens = output_tokens
    usage_obj.total_tokens = total_tokens
    usage_obj.input_tokens_details = details
    response.usage = usage_obj
    return response


@pytest.fixture
def fake_openai_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)


@pytest.fixture
def mock_client():
    """Patch the cached client factory so no network call is ever made."""
    client = MagicMock()
    client.images.generate.return_value = make_image_api_response()
    client.images.edit.return_value = make_image_api_response()
    with patch("skell_e_router.images._get_openai_client", return_value=client) as f:
        f.client = client
        yield f


# ---------------------------------------------------------------------------
# Registry / alias resolution
# ---------------------------------------------------------------------------

class TestAliasResolution:

    def test_default_aliases_point_at_flare(self):
        assert resolve_image_alias("gpt-image").name == "gpt-image-2.5-flare"
        assert resolve_image_alias("gpt-image-2.5").name == "gpt-image-2.5-flare"

    def test_explicit_variants(self):
        assert resolve_image_alias("gpt-image-2.5-flare").name == "gpt-image-2.5-flare"
        assert resolve_image_alias("gpt-image-2.5-sunburst").name == "gpt-image-2.5-sunburst"
        assert resolve_image_alias("gpt-image-2").name == "gpt-image-2"

    def test_unknown_alias_raises_invalid_model(self):
        with pytest.raises(RouterError) as exc:
            resolve_image_alias("dall-e-9000")
        assert exc.value.code == "INVALID_MODEL"
        assert "gpt-image-2.5-flare" in exc.value.message

    def test_gpt_image_entries_are_openai(self):
        for alias in ("gpt-image", "gpt-image-2.5", "gpt-image-2.5-flare",
                      "gpt-image-2.5-sunburst", "gpt-image-2"):
            model = IMAGE_CONFIG[alias]
            assert model.provider == "openai"
            assert model.is_openai

    def test_every_entry_has_a_known_provider(self):
        for model in IMAGE_CONFIG.values():
            assert model.provider in ("openai", "deepinfra", "gemini")

    def test_deepinfra_aliases(self):
        assert resolve_image_alias("seedream-4.5").name == "ByteDance/Seedream-4.5"
        assert resolve_image_alias("seedream-4").name == "ByteDance/Seedream-4"
        assert resolve_image_alias("seedream-5-pro").name == "ByteDance/Seedream-5.0-Pro"
        for alias in ("seedream-4.5", "seedream-4", "seedream-5-pro"):
            model = resolve_image_alias(alias)
            assert model.is_deepinfra
            assert model.api_base == "https://api.deepinfra.com/v1/openai"
            assert model.supported_params == frozenset({"n"})
        # 4.5 refuses anything under 3,686,400 px, so its default opens at 2K square.
        assert resolve_image_alias("seedream-4.5").default_size == "2048x2048"
        assert resolve_image_alias("seedream-4.5").min_pixels == 3_686_400
        assert resolve_image_alias("seedream-4").default_size == "1024x1024"
        assert resolve_image_alias("seedream-5-pro").default_size == "1024x1024"
        assert resolve_image_alias("seedream-4").min_pixels is None

    def test_gemini_aliases_share_one_entry(self):
        entries = [resolve_image_alias(a) for a in
                   ("nano-banana-3", "gemini-3-pro-image", "nano-banana-pro")]
        assert all(e is entries[0] for e in entries)
        assert entries[0].is_gemini
        assert entries[0].chat_alias == "nano-banana-3"
        assert entries[0].supported_params == frozenset()
        assert entries[0].allows_custom_sizes is False

    def test_per_image_pricing(self):
        flat = resolve_image_alias("seedream-4.5")
        assert flat.has_per_image_pricing
        assert flat.price_for_size("4096x4096") == 0.04
        tiered = resolve_image_alias("seedream-5-pro")
        assert tiered.price_for_size("1024x1024") == 0.0495
        assert tiered.price_for_size("1536x1536") == 0.0495     # exactly the tier edge
        assert tiered.price_for_size("1536x1537") == 0.099      # one row over it
        assert tiered.price_for_size("2048x2048") == 0.099
        assert tiered.price_for_size("bogus") is None

    def test_gpt_image_has_no_per_image_price(self):
        assert resolve_image_alias("gpt-image").has_per_image_pricing is False

    def test_25_models_have_extended_quality_tiers(self):
        for alias in ("gpt-image-2.5-flare", "gpt-image-2.5-sunburst"):
            q = IMAGE_CONFIG[alias].supported_qualities
            assert "xhigh" in q and "max" in q

    def test_gpt_image_2_tops_out_at_high(self):
        q = IMAGE_CONFIG["gpt-image-2"].supported_qualities
        assert "xhigh" not in q and "max" not in q
        assert "high" in q

    def test_pricing_registered(self):
        m = IMAGE_CONFIG["gpt-image-2.5-flare"]
        assert (m.text_input_price, m.image_input_price, m.image_output_price) == (5.0, 8.0, 30.0)
        assert m.has_pricing


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_bad_size_raises_before_call(self, fake_openai_env, mock_client):
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image", "a cat", size="1x1")
        assert exc.value.code == "INVALID_PARAM"
        assert "divisible by 16" in exc.value.message
        mock_client.client.images.generate.assert_not_called()

    def test_size_below_pixel_budget(self, fake_openai_env, mock_client):
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image", "a cat", size="512x512")
        assert "total pixels must be >=" in exc.value.message

    def test_size_edge_too_long(self, fake_openai_env, mock_client):
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image", "a cat", size="4096x1024")
        assert "longest edge" in exc.value.message

    def test_size_aspect_ratio_out_of_range(self, fake_openai_env, mock_client):
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image", "a cat", size="3520x1024")
        assert "aspect ratio" in exc.value.message

    def test_size_unparseable(self, fake_openai_env, mock_client):
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image", "a cat", size="huge")
        assert exc.value.code == "INVALID_PARAM"

    def test_named_and_custom_sizes_accepted(self, fake_openai_env, mock_client):
        for size in ("auto", "1024x1024", "1536x1024", "3072x1024"):
            generate_image("gpt-image", "a cat", size=size)
        assert mock_client.client.images.generate.call_count == 4

    def test_bad_quality_lists_allowed_values(self, fake_openai_env, mock_client):
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image", "a cat", quality="bogus")
        assert exc.value.code == "INVALID_PARAM"
        assert "'xhigh'" in exc.value.message
        mock_client.client.images.generate.assert_not_called()

    def test_quality_tier_rejected_on_older_model(self, fake_openai_env, mock_client):
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image-2", "a cat", quality="xhigh")
        assert exc.value.code == "INVALID_PARAM"
        assert "gpt-image-2" in exc.value.message

    def test_bad_background(self, fake_openai_env, mock_client):
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image", "a cat", background="rainbow")
        assert exc.value.code == "INVALID_PARAM"
        assert "transparent" in exc.value.message

    def test_transparent_requires_alpha_format(self, fake_openai_env, mock_client):
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image", "a cat", background="transparent", output_format="jpeg")
        assert "output_format" in exc.value.message

    def test_transparent_allowed_on_png(self, fake_openai_env, mock_client):
        generate_image("gpt-image", "a cat", background="transparent")
        kwargs = mock_client.client.images.generate.call_args.kwargs
        assert kwargs["background"] == "transparent"

    def test_background_omitted_when_none(self, fake_openai_env, mock_client):
        generate_image("gpt-image", "a cat")
        assert "background" not in mock_client.client.images.generate.call_args.kwargs

    def test_bad_output_format(self, fake_openai_env, mock_client):
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image", "a cat", output_format="gif")
        assert exc.value.code == "INVALID_PARAM"

    def test_compression_rejected_for_png(self, fake_openai_env, mock_client):
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image", "a cat", output_compression=50)
        assert "output_compression" in exc.value.message

    def test_compression_range(self, fake_openai_env, mock_client):
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image", "a cat", output_format="jpeg", output_compression=101)
        assert "between 0 and 100" in exc.value.message

    def test_compression_accepted_for_jpeg(self, fake_openai_env, mock_client):
        generate_image("gpt-image", "a cat", output_format="jpeg", output_compression=80)
        assert mock_client.client.images.generate.call_args.kwargs["output_compression"] == 80

    def test_n_out_of_range(self, fake_openai_env, mock_client):
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image", "a cat", n=11)
        assert exc.value.code == "INVALID_PARAM"

    def test_empty_prompt(self, fake_openai_env, mock_client):
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image", "   ")
        assert exc.value.code == "INVALID_INPUT"

    def test_missing_key_raises_missing_env(self, monkeypatch, mock_client):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image", "a cat")
        assert exc.value.code == "MISSING_ENV"


# ---------------------------------------------------------------------------
# Endpoint routing
# ---------------------------------------------------------------------------

class TestRouting:

    def test_no_images_uses_generations(self, fake_openai_env, mock_client):
        generate_image("gpt-image", "a cat", size="1024x1024", quality="low")
        mock_client.client.images.generate.assert_called_once()
        mock_client.client.images.edit.assert_not_called()
        kwargs = mock_client.client.images.generate.call_args.kwargs
        assert kwargs["model"] == "gpt-image-2.5-flare"
        assert kwargs["prompt"] == "a cat"
        assert kwargs["size"] == "1024x1024"
        assert kwargs["quality"] == "low"
        assert kwargs["n"] == 1
        assert kwargs["output_format"] == "png"

    def test_images_route_to_edits(self, fake_openai_env, mock_client, tmp_path):
        ref = tmp_path / "ref.png"
        ref.write_bytes(PNG_BYTES)
        generate_image("gpt-image", "make it blue", images=[str(ref)])
        mock_client.client.images.edit.assert_called_once()
        mock_client.client.images.generate.assert_not_called()
        uploads = mock_client.client.images.edit.call_args.kwargs["image"]
        assert uploads == [("ref.png", PNG_BYTES, "image/png")]

    def test_data_uri_reference_decoded(self, fake_openai_env, mock_client):
        generate_image(
            "gpt-image", "edit", images=[f"data:image/png;base64,{PNG_B64}"]
        )
        uploads = mock_client.client.images.edit.call_args.kwargs["image"]
        assert uploads[0][1] == PNG_BYTES
        assert uploads[0][2] == "image/png"

    def test_raw_bytes_reference(self, fake_openai_env, mock_client):
        generate_image("gpt-image", "edit", images=[PNG_BYTES])
        uploads = mock_client.client.images.edit.call_args.kwargs["image"]
        assert uploads[0][1] == PNG_BYTES

    def test_missing_reference_file(self, fake_openai_env, mock_client):
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image", "edit", images=["/nope/missing.png"])
        assert exc.value.code == "INVALID_INPUT"
        mock_client.client.images.edit.assert_not_called()

    def test_images_wrong_type(self, fake_openai_env, mock_client):
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image", "edit", images="ref.png")
        assert exc.value.code == "INVALID_INPUT"

    def test_extra_kwargs_forwarded(self, fake_openai_env, mock_client):
        generate_image("gpt-image", "a cat", user="stan")
        assert mock_client.client.images.generate.call_args.kwargs["user"] == "stan"


# ---------------------------------------------------------------------------
# Response, decoding, save()
# ---------------------------------------------------------------------------

class TestResponse:

    def test_b64_decoded_to_bytes(self, fake_openai_env, mock_client):
        resp = generate_image("gpt-image", "a cat")
        assert isinstance(resp, ImageResponse)
        assert resp.images == [PNG_BYTES]
        assert resp.images[0].startswith(b"\x89PNG")
        assert resp.format == "png"
        assert resp.model == "gpt-image-2.5-flare"
        assert resp.duration_seconds is not None

    def test_multiple_images(self, fake_openai_env, mock_client):
        mock_client.client.images.generate.return_value = make_image_api_response(n=3)
        resp = generate_image("gpt-image", "a cat", n=3)
        assert len(resp.images) == 3

    def test_empty_data_raises_provider_error(self, fake_openai_env, mock_client):
        empty = MagicMock()
        empty.data = []
        empty.usage = None
        empty.model = "gpt-image-2.5-flare"
        mock_client.client.images.generate.return_value = empty
        with pytest.raises(RouterError) as exc:
            generate_image("gpt-image", "a cat")
        assert exc.value.code == "PROVIDER_ERROR"

    def test_save_to_directory(self, fake_openai_env, mock_client, tmp_path):
        mock_client.client.images.generate.return_value = make_image_api_response(n=2)
        resp = generate_image("gpt-image", "a cat", n=2)
        out_dir = tmp_path / "out"
        paths = resp.save(str(out_dir), stem="cat")
        assert [os.path.basename(p) for p in paths] == ["cat_0.png", "cat_1.png"]
        for p in paths:
            assert open(p, "rb").read() == PNG_BYTES

    def test_save_to_existing_directory(self, fake_openai_env, mock_client, tmp_path):
        resp = generate_image("gpt-image", "a cat")
        paths = resp.save(str(tmp_path))
        assert [os.path.basename(p) for p in paths] == ["image_0.png"]

    def test_save_to_file_path_single_image(self, fake_openai_env, mock_client, tmp_path):
        resp = generate_image("gpt-image", "a cat")
        target = tmp_path / "exact.png"
        paths = resp.save(str(target))
        assert paths == [str(target)]
        assert target.read_bytes() == PNG_BYTES

    def test_save_to_file_path_multiple_images_indexes(self, fake_openai_env, mock_client, tmp_path):
        mock_client.client.images.generate.return_value = make_image_api_response(n=2)
        resp = generate_image("gpt-image", "a cat", n=2)
        paths = resp.save(str(tmp_path / "shot.png"))
        assert [os.path.basename(p) for p in paths] == ["shot_0.png", "shot_1.png"]

    def test_jpeg_saves_with_jpg_extension(self, fake_openai_env, mock_client, tmp_path):
        resp = generate_image("gpt-image", "a cat", output_format="jpeg")
        assert resp.extension == "jpg"
        paths = resp.save(str(tmp_path / "out"))
        assert paths[0].endswith("image_0.jpg")

    def test_repr(self, fake_openai_env, mock_client):
        resp = generate_image("gpt-image", "a cat")
        assert "ImageResponse(" in repr(resp)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

class TestCost:

    def test_cost_from_usage(self, fake_openai_env, mock_client):
        mock_client.client.images.generate.return_value = make_image_api_response(
            input_tokens=120, output_tokens=1000, total_tokens=1120,
            text_tokens=100, image_tokens=20,
        )
        resp = generate_image("gpt-image", "a cat")
        expected = (100 * 5.0 + 20 * 8.0 + 1000 * 30.0) / 1_000_000
        assert resp.cost == pytest.approx(expected)
        assert resp.input_tokens == 120
        assert resp.output_tokens == 1000
        assert resp.total_tokens == 1120

    def test_cost_without_breakdown_treats_input_as_text(self, fake_openai_env, mock_client):
        mock_client.client.images.generate.return_value = make_image_api_response(
            input_tokens=50, output_tokens=100, text_tokens=None, image_tokens=None,
        )
        resp = generate_image("gpt-image", "a cat")
        assert resp.cost == pytest.approx((50 * 5.0 + 100 * 30.0) / 1_000_000)

    def test_cost_none_without_usage(self, fake_openai_env, mock_client):
        mock_client.client.images.generate.return_value = make_image_api_response(usage=False)
        resp = generate_image("gpt-image", "a cat")
        assert resp.cost is None

    def test_cost_none_when_prices_unknown(self, fake_openai_env, mock_client, monkeypatch):
        unpriced = ImageModel(
            name="gpt-image-unpriced",
            provider="openai",
            supported_sizes=("auto", "1024x1024"),
            supported_qualities=("auto", "low"),
            text_input_price=None,
            image_input_price=None,
            image_output_price=None,
        )
        assert unpriced.has_pricing is False
        monkeypatch.setitem(IMAGE_CONFIG, "gpt-image-unpriced", unpriced)
        resp = generate_image("gpt-image-unpriced", "a cat")
        assert resp.cost is None


# ---------------------------------------------------------------------------
# Key resolution and retry
# ---------------------------------------------------------------------------

class TestKeyAndRetry:

    def test_config_api_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
        client = MagicMock()
        client.images.generate.return_value = make_image_api_response()
        with patch("skell_e_router.images._get_openai_client", return_value=client) as factory:
            generate_image(
                "gpt-image", "a cat", config={"openai_api_key": FAKE_OPENAI_KEY}
            )
        assert factory.call_args.args[0] == FAKE_OPENAI_KEY

    def test_config_key_satisfies_missing_env(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = MagicMock()
        client.images.generate.return_value = make_image_api_response()
        with patch("skell_e_router.images._get_openai_client", return_value=client):
            resp = generate_image(
                "gpt-image", "a cat", config={"openai_api_key": FAKE_OPENAI_KEY}
            )
        assert resp.images == [PNG_BYTES]

    def test_retries_retryable_error_then_succeeds(self, fake_openai_env):
        class Transient(Exception):
            status_code = 503

        client = MagicMock()
        client.images.generate.side_effect = [Transient(), make_image_api_response()]
        with patch("skell_e_router.images._get_openai_client", return_value=client), no_retry_sleep():
            resp = generate_image("gpt-image", "a cat")
        assert client.images.generate.call_count == 2
        assert resp.images == [PNG_BYTES]

    def test_non_retryable_error_wrapped_once(self, fake_openai_env):
        class BadRequest(Exception):
            status_code = 400

        client = MagicMock()
        client.images.generate.side_effect = BadRequest("raw provider text")
        with patch("skell_e_router.images._get_openai_client", return_value=client):
            with pytest.raises(RouterError) as exc:
                generate_image("gpt-image", "a cat")
        assert client.images.generate.call_count == 1
        assert exc.value.code == "PROVIDER_ERROR"
        assert "raw provider text" not in exc.value.message
        assert exc.value.details["category"] == "invalid_request"
        assert exc.value.details["endpoint"] == "generations"

    def test_retry_exhausts_and_wraps(self, fake_openai_env):
        class Transient(Exception):
            status_code = 503

        client = MagicMock()
        client.images.generate.side_effect = Transient()
        with patch("skell_e_router.images._get_openai_client", return_value=client), no_retry_sleep():
            with pytest.raises(RouterError) as exc:
                generate_image("gpt-image", "a cat")
        assert client.images.generate.call_count == 3
        assert exc.value.code == "PROVIDER_ERROR"


# ---------------------------------------------------------------------------
# Verbosity
# ---------------------------------------------------------------------------

class TestVerbosity:

    def test_debug_prints_request_and_response(self, fake_openai_env, mock_client, capsys):
        generate_image("gpt-image", "a cat", verbosity="debug")
        out = capsys.readouterr().out
        assert "IMAGE (gpt-image-2.5-flare)" in out
        assert "PROMPT: a cat" in out
        assert "cost=" in out

    def test_invalid_verbosity_warns(self, fake_openai_env, mock_client, capsys):
        generate_image("gpt-image", "a cat", verbosity="loud")
        assert "Invalid verbosity" in capsys.readouterr().out

    def test_none_verbosity_is_silent(self, fake_openai_env, mock_client, capsys):
        generate_image("gpt-image", "a cat")
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# DeepInfra (ByteDance Seedream) via the OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01" + b"\x00" * 16
JPEG_B64 = base64.b64encode(JPEG_BYTES).decode()


def make_deepinfra_response(n: int = 1, payload_b64: str = JPEG_B64,
                            model: str = "ByteDance/Seedream-4.5"):
    """DeepInfra answers in the OpenAI images shape but reports no usage."""
    response = MagicMock()
    response.data = [{"b64_json": payload_b64, "revised_prompt": "a cat"} for _ in range(n)]
    response.model = model
    response.usage = None
    return response


@pytest.fixture
def fake_deepinfra_env(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", FAKE_DEEPINFRA_KEY)


@pytest.fixture
def deepinfra_client():
    client = MagicMock()
    client.images.generate.return_value = make_deepinfra_response()
    with patch("skell_e_router.images._get_openai_client", return_value=client) as f:
        f.client = client
        yield f


class TestDeepInfraRouting:

    def test_uses_deepinfra_base_url_and_key(self, fake_deepinfra_env, deepinfra_client):
        generate_image("seedream-4.5", "a cat")
        args = deepinfra_client.call_args.args
        assert args[0] == FAKE_DEEPINFRA_KEY
        assert args[1] == "https://api.deepinfra.com/v1/openai"

    def test_openai_models_still_get_no_base_url(self, fake_openai_env, mock_client):
        generate_image("gpt-image", "a cat")
        assert mock_client.call_args.args[1] is None

    def test_config_key_overrides_env(self, monkeypatch, deepinfra_client):
        monkeypatch.setenv("DEEPINFRA_API_KEY", "from-env")
        generate_image("seedream-4.5", "a cat", config={"deepinfra_api_key": "from-config"})
        assert deepinfra_client.call_args.args[0] == "from-config"

    def test_missing_key_raises_missing_env(self, monkeypatch, deepinfra_client):
        monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
        with pytest.raises(RouterError) as exc:
            generate_image("seedream-4.5", "a cat")
        assert exc.value.code == "MISSING_ENV"
        assert exc.value.details["required"] == "DEEPINFRA_API_KEY"

    def test_request_body_is_the_documented_four_fields(self, fake_deepinfra_env, deepinfra_client):
        generate_image("seedream-4.5", "a cat", size="2560x1440", n=2)
        kwargs = deepinfra_client.client.images.generate.call_args.kwargs
        assert kwargs == {
            "prompt": "a cat",
            "model": "ByteDance/Seedream-4.5",
            "size": "2560x1440",
            "n": 2,
            "response_format": "b64_json",
        }

    def test_auto_size_resolves_to_the_model_default(self, fake_deepinfra_env, deepinfra_client):
        resp = generate_image("seedream-4.5", "a cat")
        assert deepinfra_client.client.images.generate.call_args.kwargs["size"] == "2048x2048"
        assert resp.size == "2048x2048"

        resp = generate_image("seedream-4", "a cat")
        assert deepinfra_client.client.images.generate.call_args.kwargs["size"] == "1024x1024"
        assert resp.size == "1024x1024"

    def test_custom_pixel_size_passes_through(self, fake_deepinfra_env, deepinfra_client):
        generate_image("seedream-4.5", "a cat", size="3840x2160")
        assert deepinfra_client.client.images.generate.call_args.kwargs["size"] == "3840x2160"

    def test_shorthand_resolutions_are_rejected(self, fake_deepinfra_env, deepinfra_client):
        # DeepInfra's gateway 422s on "2K" — only "{width}x{height}" is accepted.
        for size in ("2K", "4K"):
            with pytest.raises(RouterError) as exc:
                generate_image("seedream-4", "a cat", size=size)
            assert exc.value.code == "INVALID_PARAM"
        deepinfra_client.client.images.generate.assert_not_called()

    def test_size_below_the_model_floor_rejected_locally(self, fake_deepinfra_env, deepinfra_client):
        with pytest.raises(RouterError) as exc:
            generate_image("seedream-4.5", "a cat", size="1024x1024")
        assert exc.value.code == "INVALID_PARAM"
        assert "at least 3,686,400 pixels" in exc.value.message
        deepinfra_client.client.images.generate.assert_not_called()

    def test_no_floor_on_the_other_seedream_models(self, fake_deepinfra_env, deepinfra_client):
        generate_image("seedream-4", "a cat", size="1024x1024")
        generate_image("seedream-5-pro", "a cat", size="1024x1024")
        assert deepinfra_client.client.images.generate.call_count == 2

    def test_unparseable_size_rejected_before_the_call(self, fake_deepinfra_env, deepinfra_client):
        with pytest.raises(RouterError) as exc:
            generate_image("seedream-4.5", "a cat", size="enormous")
        assert exc.value.code == "INVALID_PARAM"
        deepinfra_client.client.images.generate.assert_not_called()

    def test_gpt_image_custom_size_rules_do_not_apply(self, fake_deepinfra_env, deepinfra_client):
        # 300x300 is far below GPT-Image's pixel floor and not a multiple of 16.
        generate_image("seedream-4", "a cat", size="300x300")
        assert deepinfra_client.client.images.generate.call_args.kwargs["size"] == "300x300"

    def test_extra_kwargs_forwarded(self, fake_deepinfra_env, deepinfra_client):
        generate_image("seedream-4.5", "a cat", seed=7)
        assert deepinfra_client.client.images.generate.call_args.kwargs["seed"] == 7

    def test_reference_images_rejected(self, fake_deepinfra_env, deepinfra_client):
        with pytest.raises(RouterError) as exc:
            generate_image("seedream-4.5", "a cat", images=["ref.png"])
        assert exc.value.code == "INVALID_INPUT"
        assert "reference images" in exc.value.message


class TestDeepInfraResponse:

    def test_format_sniffed_from_jpeg_magic_bytes(self, fake_deepinfra_env, deepinfra_client):
        resp = generate_image("seedream-4.5", "a cat")
        assert resp.images == [JPEG_BYTES]
        assert resp.format == "jpeg"
        assert resp.extension == "jpg"

    def test_format_sniffed_from_png_magic_bytes(self, fake_deepinfra_env, deepinfra_client):
        deepinfra_client.client.images.generate.return_value = make_deepinfra_response(
            payload_b64=PNG_B64
        )
        resp = generate_image("seedream-4.5", "a cat")
        assert resp.format == "png"

    def test_format_falls_back_when_magic_bytes_are_unknown(self, fake_deepinfra_env, deepinfra_client):
        blob = base64.b64encode(b"NOTANIMAGE" + b"\x00" * 16).decode()
        deepinfra_client.client.images.generate.return_value = make_deepinfra_response(
            payload_b64=blob
        )
        resp = generate_image("seedream-4.5", "a cat")
        assert resp.format == "png"

    def test_webp_magic_bytes(self, fake_deepinfra_env, deepinfra_client):
        webp = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBPVP8 " + b"\x00" * 8
        deepinfra_client.client.images.generate.return_value = make_deepinfra_response(
            payload_b64=base64.b64encode(webp).decode()
        )
        resp = generate_image("seedream-4.5", "a cat")
        assert resp.format == "webp"

    def test_empty_data_raises_provider_error(self, fake_deepinfra_env, deepinfra_client):
        empty = MagicMock()
        empty.data = []
        empty.usage = None
        empty.model = "ByteDance/Seedream-4.5"
        deepinfra_client.client.images.generate.return_value = empty
        with pytest.raises(RouterError) as exc:
            generate_image("seedream-4.5", "a cat")
        assert exc.value.code == "PROVIDER_ERROR"

    def test_no_usage_reported_leaves_token_fields_none(self, fake_deepinfra_env, deepinfra_client):
        resp = generate_image("seedream-4.5", "a cat")
        assert (resp.input_tokens, resp.output_tokens, resp.total_tokens) == (None, None, None)
        assert resp.duration_seconds is not None


class TestDeepInfraCost:

    def test_flat_price_per_image(self, fake_deepinfra_env, deepinfra_client):
        resp = generate_image("seedream-4.5", "a cat")
        assert resp.cost == pytest.approx(0.04)

    def test_flat_price_scales_with_n(self, fake_deepinfra_env, deepinfra_client):
        deepinfra_client.client.images.generate.return_value = make_deepinfra_response(n=3)
        resp = generate_image("seedream-4", "a cat", n=3)
        assert resp.cost == pytest.approx(0.12)

    def test_tiered_price_low_tier(self, fake_deepinfra_env, deepinfra_client):
        resp = generate_image("seedream-5-pro", "a cat", size="1024x1024")
        assert resp.cost == pytest.approx(0.0495)

    def test_tiered_price_high_tier(self, fake_deepinfra_env, deepinfra_client):
        resp = generate_image("seedream-5-pro", "a cat", size="2048x2048")
        assert resp.cost == pytest.approx(0.099)

    def test_tiered_price_uses_the_size_actually_sent(self, fake_deepinfra_env, deepinfra_client):
        # size="auto" becomes 1024x1024, which prices in the low tier.
        resp = generate_image("seedream-5-pro", "a cat")
        assert resp.size == "1024x1024"
        assert resp.cost == pytest.approx(0.0495)


# ---------------------------------------------------------------------------
# Unsupported parameters per provider
# ---------------------------------------------------------------------------

class TestUnsupportedParams:

    @pytest.mark.parametrize("kwargs, param", [
        ({"quality": "high"}, "quality"),
        ({"background": "transparent"}, "background"),
        ({"background": "opaque"}, "background"),
        ({"output_format": "jpeg"}, "output_format"),
        ({"output_compression": 80}, "output_compression"),
    ])
    def test_deepinfra_rejects_non_default(self, fake_deepinfra_env, deepinfra_client, kwargs, param):
        with pytest.raises(RouterError) as exc:
            generate_image("seedream-4.5", "a cat", **kwargs)
        assert exc.value.code == "INVALID_PARAM"
        assert f"`{param}` is not supported" in exc.value.message
        deepinfra_client.client.images.generate.assert_not_called()

    def test_deepinfra_accepts_every_default(self, fake_deepinfra_env, deepinfra_client):
        resp = generate_image(
            "seedream-4.5", "a cat",
            quality="auto", background=None, output_format="png", output_compression=None, n=1,
        )
        assert resp.images == [JPEG_BYTES]

    def test_deepinfra_accepts_n_because_it_is_supported(self, fake_deepinfra_env, deepinfra_client):
        deepinfra_client.client.images.generate.return_value = make_deepinfra_response(n=2)
        resp = generate_image("seedream-4.5", "a cat", n=2)
        assert len(resp.images) == 2

    @pytest.mark.parametrize("kwargs, param", [
        ({"quality": "high"}, "quality"),
        ({"output_format": "webp"}, "output_format"),
        ({"n": 2}, "n"),
    ])
    def test_gemini_rejects_non_default(self, fake_gemini_env, gemini_ask_ai, kwargs, param):
        with pytest.raises(RouterError) as exc:
            generate_image("nano-banana-3", "a cat", **kwargs)
        assert exc.value.code == "INVALID_PARAM"
        assert f"`{param}` is not supported" in exc.value.message
        gemini_ask_ai.assert_not_called()

    def test_gemini_accepts_every_default(self, fake_gemini_env, gemini_ask_ai):
        resp = generate_image("nano-banana-3", "a cat")
        assert resp.images == [PNG_BYTES]

    def test_openai_still_takes_all_params(self, fake_openai_env, mock_client):
        generate_image("gpt-image", "a cat", quality="high", background="opaque",
                       output_format="webp", output_compression=70, n=1)
        kwargs = mock_client.client.images.generate.call_args.kwargs
        assert kwargs["quality"] == "high"
        assert kwargs["background"] == "opaque"
        assert kwargs["output_compression"] == 70


# ---------------------------------------------------------------------------
# Gemini through the chat path
# ---------------------------------------------------------------------------

def make_ai_response(mime: str = "image/png", payload_b64: str = PNG_B64, count: int = 1):
    """Stand-in for the AIResponse that ask_ai returns for nano-banana-3."""
    response = MagicMock()
    response.images = [
        {"image_url": {"url": f"data:{mime};base64,{payload_b64}", "detail": "auto"},
         "index": i, "type": "image_url"}
        for i in range(count)
    ]
    response.model = "gemini-3-pro-image-preview"
    response.prompt_tokens = 11
    response.completion_tokens = 1290
    response.total_tokens = 1301
    response.cost = 0.134
    response.duration_seconds = 4.2
    return response


@pytest.fixture
def fake_gemini_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)


@pytest.fixture
def gemini_ask_ai():
    with patch("skell_e_router.images.ask_ai", return_value=make_ai_response()) as f:
        yield f


class TestGemini:

    def test_calls_ask_ai_with_the_chat_alias(self, fake_gemini_env, gemini_ask_ai):
        generate_image("nano-banana-3", "a cat")
        args, kwargs = gemini_ask_ai.call_args
        assert args == ("nano-banana-3", "a cat")
        assert kwargs["rich_response"] is True
        assert kwargs["images"] is None

    def test_size_is_never_sent_to_the_api(self, fake_gemini_env, gemini_ask_ai):
        resp = generate_image("nano-banana-3", "a cat", size="1024x1024")
        assert "size" not in gemini_ask_ai.call_args.kwargs
        assert resp.size == "1024x1024"

    def test_only_auto_and_1024_sizes_accepted(self, fake_gemini_env, gemini_ask_ai):
        for size in ("auto", "1024x1024"):
            generate_image("nano-banana-3", "a cat", size=size)
        assert gemini_ask_ai.call_count == 2

    @pytest.mark.parametrize("size", ["1536x1024", "2048x2048", "2K"])
    def test_other_sizes_rejected(self, fake_gemini_env, gemini_ask_ai, size):
        with pytest.raises(RouterError) as exc:
            generate_image("nano-banana-3", "a cat", size=size)
        assert exc.value.code == "INVALID_PARAM"
        assert "picks the output resolution itself" in exc.value.message
        gemini_ask_ai.assert_not_called()

    def test_reference_images_forwarded(self, fake_gemini_env, gemini_ask_ai):
        generate_image("nano-banana-3", "restyle this", images=["ref.png"])
        assert gemini_ask_ai.call_args.kwargs["images"] == ["ref.png"]

    def test_data_urls_decoded_to_bytes(self, fake_gemini_env, gemini_ask_ai):
        resp = generate_image("nano-banana-3", "a cat")
        assert resp.images == [PNG_BYTES]
        assert resp.format == "png"

    def test_format_from_jpeg_payload(self, fake_gemini_env):
        with patch("skell_e_router.images.ask_ai",
                   return_value=make_ai_response(mime="image/jpeg", payload_b64=JPEG_B64)):
            resp = generate_image("nano-banana-3", "a cat")
        assert resp.format == "jpeg"

    def test_format_falls_back_to_the_declared_mime(self, fake_gemini_env):
        blob = base64.b64encode(b"NOTANIMAGE" + b"\x00" * 16).decode()
        with patch("skell_e_router.images.ask_ai",
                   return_value=make_ai_response(mime="image/webp", payload_b64=blob)):
            resp = generate_image("nano-banana-3", "a cat")
        assert resp.format == "webp"

    def test_multiple_images_decoded(self, fake_gemini_env):
        with patch("skell_e_router.images.ask_ai", return_value=make_ai_response(count=3)):
            resp = generate_image("nano-banana-3", "a cat")
        assert len(resp.images) == 3

    def test_usage_and_cost_carried_from_the_chat_response(self, fake_gemini_env, gemini_ask_ai):
        resp = generate_image("nano-banana-3", "a cat")
        assert resp.input_tokens == 11
        assert resp.output_tokens == 1290
        assert resp.total_tokens == 1301
        assert resp.cost == pytest.approx(0.134)
        assert resp.duration_seconds == pytest.approx(4.2)
        assert resp.model == "gemini-3-pro-image-preview"

    def test_text_only_answer_raises_provider_error(self, fake_gemini_env):
        text_only = MagicMock()
        text_only.images = None
        with patch("skell_e_router.images.ask_ai", return_value=text_only):
            with pytest.raises(RouterError) as exc:
                generate_image("nano-banana-3", "a cat")
        assert exc.value.code == "PROVIDER_ERROR"
        assert "no image data" in exc.value.message

    def test_malformed_data_url_raises_provider_error(self, fake_gemini_env):
        broken = MagicMock()
        broken.images = [{"image_url": {"url": "data:image/png;base64"}}]
        with patch("skell_e_router.images.ask_ai", return_value=broken):
            with pytest.raises(RouterError) as exc:
                generate_image("nano-banana-3", "a cat")
        assert exc.value.code == "PROVIDER_ERROR"

    def test_missing_gemini_key_raises_before_the_call(self, monkeypatch, gemini_ask_ai):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(RouterError) as exc:
            generate_image("nano-banana-3", "a cat")
        assert exc.value.code == "MISSING_ENV"
        gemini_ask_ai.assert_not_called()

    def test_save_uses_the_real_format(self, fake_gemini_env, tmp_path):
        with patch("skell_e_router.images.ask_ai",
                   return_value=make_ai_response(mime="image/jpeg", payload_b64=JPEG_B64)):
            resp = generate_image("nano-banana-3", "a cat")
        paths = resp.save(str(tmp_path / "out"))
        assert paths[0].endswith("image_0.jpg")
