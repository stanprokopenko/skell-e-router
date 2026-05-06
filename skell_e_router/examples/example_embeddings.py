"""End-to-end runnable examples for skell_e_router.get_embedding().

Requires: OPENAI_API_KEY and/or GEMINI_API_KEY in your environment.
Run from the repo root:

    python -m skell_e_router.examples.example_embeddings
"""

from skell_e_router import get_embedding, EmbeddingResponse, RouterError


def section(title: str) -> None:
    print(f"\n{'=' * 8}  {title}  {'=' * 8}")


def main() -> None:
    # 1. Single string → list[float]
    section("1. Single string (OpenAI large)")
    v = get_embedding("openai-embedding-3-large", "hello world")
    print(f"  dim={len(v)}  preview={v[:3]}")

    # 2. Batch of strings → list[list[float]]
    section("2. Batch of strings (OpenAI large)")
    vs = get_embedding(
        "openai-embedding-3-large",
        ["hello", "world", "foo", "bar"],
    )
    print(f"  count={len(vs)}  dim_each={len(vs[0])}")

    # 3. Truncated dimensions
    section("3. Truncated dimensions (OpenAI small, 512)")
    v = get_embedding("openai-embedding-3-small", "hello", dimensions=512)
    print(f"  dim={len(v)}")
    assert len(v) == 512

    # 4. Multimodal aggregation: text + image → 1 fused embedding (Gemini)
    section("4. Multimodal aggregation (Gemini, text + image)")
    v = get_embedding(
        "gemini-embedding-2",
        [["a red shoe on a wooden floor", "skell_e_router/examples/vision-test.jpg"]],
    )
    print(f"  count={len(v)}  fused_dim={len(v[0])}")

    # 5. Gemini text batch (the high-throughput path)
    section("5. Gemini text batch")
    vs = get_embedding(
        "gemini-embedding-2",
        ["one", "two", "three"],
    )
    print(f"  count={len(vs)}  dim_each={len(vs[0])}")
    assert len(vs) == 3

    # 6. Rich response
    section("6. Rich response")
    resp: EmbeddingResponse = get_embedding(
        "openai-embedding-3-large", "hello", rich_response=True,
    )
    print(
        f"  model={resp.model}  dim={resp.dimensions}  "
        f"prompt_tokens={resp.prompt_tokens}  cost={resp.cost}"
    )

    # 7. Capability error: OpenAI rejects image input
    section("7. Capability error (expected)")
    try:
        get_embedding(
            "openai-embedding-3-large",
            ["data:image/png;base64,iVBORw0KGgo..."],
        )
    except RouterError as e:
        print(f"  Got expected error: code={e.code}  message={e.message}")


if __name__ == "__main__":
    main()
