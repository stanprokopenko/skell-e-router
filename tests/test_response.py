"""Tests for the AIResponse dataclass."""

from skell_e_router.response import AIResponse


class TestAIResponseCreation:

    def test_minimal_creation(self):
        r = AIResponse(content="hello", model="openai/gpt-5")
        assert r.content == "hello"
        assert r.model == "openai/gpt-5"

    def test_defaults_are_none(self):
        r = AIResponse(content="", model="m")
        assert r.finish_reason is None
        assert r.prompt_tokens is None
        assert r.completion_tokens is None
        assert r.total_tokens is None
        assert r.reasoning_tokens is None
        assert r.cost is None
        assert r.duration_seconds is None
        assert r.grounding_metadata is None
        assert r.safety_ratings is None
        assert r.images is None
        assert r.tool_calls is None
        assert r.function_call is None
        assert r.provider_specific_fields is None
        assert r.raw_response is None

    def test_full_creation(self):
        img_data = [{"image_url": {"url": "data:image/png;base64,abc"}, "index": 0, "type": "image_url"}]
        r = AIResponse(
            content="result",
            model="gemini/gemini-2.5-pro",
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            reasoning_tokens=50,
            cost=0.005,
            duration_seconds=1.23,
            grounding_metadata={"key": "val"},
            safety_ratings=[{"category": "HARM", "probability": "LOW"}],
            images=img_data,
            tool_calls=[{"id": "tc1"}],
            function_call={"name": "fn"},
            provider_specific_fields={"extra": True},
            raw_response="raw",
        )
        assert r.prompt_tokens == 100
        assert r.completion_tokens == 200
        assert r.total_tokens == 300
        assert r.reasoning_tokens == 50
        assert r.cost == 0.005
        assert r.duration_seconds == 1.23
        assert r.grounding_metadata == {"key": "val"}
        assert r.safety_ratings[0]["category"] == "HARM"
        assert r.images == img_data
        assert r.tool_calls == [{"id": "tc1"}]
        assert r.function_call == {"name": "fn"}
        assert r.provider_specific_fields == {"extra": True}
        assert r.raw_response == "raw"

    def test_images_stored(self):
        imgs = [{"image_url": {"url": "data:image/png;base64,abc"}, "index": 0, "type": "image_url"}]
        r = AIResponse(content="here is an image", model="gemini/gemini-3-pro-image-preview", images=imgs)
        assert r.images == imgs
        assert len(r.images) == 1


class TestAIResponseStr:

    def test_str_returns_content(self):
        r = AIResponse(content="my content", model="m")
        assert str(r) == "my content"

    def test_str_empty_content(self):
        r = AIResponse(content="", model="m")
        assert str(r) == ""

    def test_string_concatenation(self):
        r = AIResponse(content="hello", model="m")
        assert "prefix " + str(r) == "prefix hello"


class TestAIResponseRepr:

    def test_repr_short_content(self):
        r = AIResponse(content="short", model="test-model")
        result = repr(r)
        assert "AIResponse" in result
        assert "short" in result
        assert "test-model" in result

    def test_repr_long_content_truncates(self):
        long_text = "a" * 100
        r = AIResponse(content=long_text, model="m")
        result = repr(r)
        # repr truncates to first 50 chars
        assert "..." in result
        assert len(result) < len(long_text) + 50


class TestAIResponseEquality:

    def test_equal_instances(self):
        """Dataclasses support equality by default."""
        r1 = AIResponse(content="x", model="m")
        r2 = AIResponse(content="x", model="m")
        assert r1 == r2

    def test_unequal_instances(self):
        r1 = AIResponse(content="x", model="m")
        r2 = AIResponse(content="y", model="m")
        assert r1 != r2
