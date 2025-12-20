"""
Test script to verify Google Search Grounding is working with litellm.
"""
import os
import json
import litellm

# Enable debug to see what's being sent
litellm._turn_on_debug()

def test_grounding():
    """Test Google Search Grounding directly with litellm."""
    
    print("Testing Google Search Grounding with litellm...")
    print("=" * 60)
    
    # Try using litellm's native web_search_options parameter
    # This should enable Google Search grounding for Gemini models
    response = litellm.completion(
        model="gemini/gemini-3-pro-preview",
        messages=[
            {"role": "user", "content": "What is the latest news on Trump?"}
        ],
        # Use litellm's native web search parameter
        web_search_options={"search_context_size": "high"}
    )
    
    print("\n--- RESPONSE CONTENT ---")
    print(response.choices[0].message.content)
    
    print("\n--- CHECKING FOR GROUNDING METADATA ---")
    
    # Check various places where grounding metadata might be
    print(f"\nResponse type: {type(response)}")
    print(f"Response attributes: {dir(response)}")
    
    # Check model_extra (where litellm often puts provider-specific data)
    if hasattr(response, 'model_extra'):
        print(f"\nmodel_extra: {response.model_extra}")
    
    # Check _hidden_params
    if hasattr(response, '_hidden_params'):
        print(f"\n_hidden_params: {response._hidden_params}")
    
    # Check for grounding metadata specifically
    grounding_metadata = getattr(response, 'groundingMetadata', None)
    if grounding_metadata:
        print(f"\ngroundingMetadata: {grounding_metadata}")
    
    # Check the raw response
    if hasattr(response, '_response'):
        print(f"\n_response: {response._response}")
    
    # Try to dump the full response
    try:
        full_dump = response.model_dump()
        print(f"\nFull response dump:")
        print(json.dumps(full_dump, indent=2, default=str))
    except Exception as e:
        print(f"\nCouldn't dump response: {e}")
    
    # Check choices[0].message for provider_specific_fields
    message = response.choices[0].message
    print(f"\nMessage attributes: {dir(message)}")
    
    if hasattr(message, 'provider_specific_fields'):
        print(f"\nprovider_specific_fields: {message.provider_specific_fields}")
    
    # Check for vertex_ai_grounding_metadata
    if hasattr(response, 'vertex_ai_grounding_metadata'):
        print(f"\nvertex_ai_grounding_metadata: {response.vertex_ai_grounding_metadata}")
    
    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        print("ERROR: GEMINI_API_KEY not set")
        exit(1)
    
    test_grounding()
