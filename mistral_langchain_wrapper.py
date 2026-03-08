from langchain_openai import ChatOpenAI

_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

# Maps short aliases to current Mistral model identifiers
_MODEL_ALIASES = {
    "small": "mistral-small-latest",
    "medium": "mistral-medium-latest",
    "large": "mistral-large-latest",
}


def _resolve_model(model: str) -> str:
    return _MODEL_ALIASES.get(model, model)


class MistralChatWrapper(ChatOpenAI):
    """
    LangChain chat model for the Mistral API, built on langchain_openai.ChatOpenAI.
    Mistral exposes an OpenAI-compatible endpoint, so no extra SDK is needed.

    Model names can be supplied as short aliases ("small", "medium", "large")
    or as full Mistral model IDs (e.g. "mistral-large-latest").
    Tool calling is inherited from ChatOpenAI.

    Usage:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("API_KEY")
        llm = MistralChatWrapper(api_key=api_key, model="small")
        llm_with_tools = llm.bind_tools([my_tool])
    """

    def __init__(self, api_key: str, model: str = "mistral-small-latest", **kwargs):
        super().__init__(
            api_key=api_key,
            model=_resolve_model(model),
            base_url=_MISTRAL_BASE_URL,
            **kwargs,
        )


class MistralLLMWrapper(ChatOpenAI):
    """
    LangChain LLM-style wrapper for the Mistral API (plain text in/out).
    Inherits from ChatOpenAI for compatibility with RAGAS and other pipelines.

    Usage:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("API_KEY")
        llm = MistralLLMWrapper(api_key=api_key, model="small")
        answer = llm.invoke("What is the capital of France?")
    """

    def __init__(self, api_key: str, model: str = "mistral-small-latest", **kwargs):
        super().__init__(
            api_key=api_key,
            model=_resolve_model(model),
            base_url=_MISTRAL_BASE_URL,
            **kwargs,
        )
