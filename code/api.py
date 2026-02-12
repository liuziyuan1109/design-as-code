import os
from openai import OpenAI


def get_api_client(api_key: str = None, model: str = "gpt-4o") -> OpenAI:
    """
    Initialize and return an OpenAI API client.
    
    Args:
        api_key: OpenAI API key. If not provided, will try to read from OPENAI_API_KEY environment variable.
        model: The model to use (e.g., "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo")
    
    Returns:
        OpenAI client instance
    
    Example:
        >>> client = get_api_client()  # Uses OPENAI_API_KEY environment variable
        >>> client = get_api_client(api_key="sk-...")  # Uses provided key
    """
    # Get API key from parameter or environment variable
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please provide it as a parameter "
                "or set the OPENAI_API_KEY environment variable."
            )
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Store model name for reference
    client.default_model = model
    
    return client


if __name__ == "__main__":
    # Example usage - requires OPENAI_API_KEY environment variable to be set
    # or pass api_key directly: get_api_client(api_key="sk-...")
    
    client = get_api_client(model="gpt-4o")
    
    print("Testing OpenAI API client...")
    print(f"Using model: {client.default_model}")
    
    # Test API call
    try:
        response = client.chat.completions.create(
            model=client.default_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
            ],
            max_tokens=100,
        )
        
        print("\nAPI Response:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to set your OpenAI API key:")
        print("export OPENAI_API_KEY='sk-...'")
