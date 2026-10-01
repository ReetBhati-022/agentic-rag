"""
llm_factory.py
--------------
LLMFactory.create_llm(provider) -> a LangChain chat model with
.invoke(prompt) -> response.content, built from config.yaml's `llm`
section. One instance is cached per provider so repeated calls (each
chat turn) don't rebuild the client every time.

Supported providers (config.yaml -> llm.providers): ollama, groq,
huggingface. Each provider's model_name/api_key_env/host_env come from
config.yaml; the actual key/host values come from the real process
environment (loaded from .env by app.py's load_dotenv() call before
this module is ever used).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

SUPPORTED_PROVIDERS = ["ollama", "groq", "huggingface"]


class LLMFactory:
    def __init__(self, llm_config: Dict[str, Any]):
        self.config = llm_config or {}
        self.temperature = self.config.get("temperature", 0.2)
        self.max_tokens = self.config.get("max_tokens", 2048)
        self.default_provider = self.config.get("default_provider", "ollama")
        self.providers_cfg = self.config.get("providers", {})
        self._cache: Dict[str, Any] = {}

    def create_llm(self, provider: Optional[str] = None):
        provider = (provider or self.default_provider).lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider '{provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
            )
        if provider in self._cache:
            return self._cache[provider]

        provider_cfg = self.providers_cfg.get(provider, {})
        builder = getattr(self, f"_build_{provider}")
        llm = builder(provider_cfg)
        self._cache[provider] = llm
        return llm

    # -----------------------------------------------------------------
    def _build_groq(self, provider_cfg: Dict[str, Any]):
        from langchain_groq import ChatGroq

        api_key = os.environ.get(provider_cfg.get("api_key_env", "GROQ_API_KEY"))
        if not api_key:
            raise RuntimeError(
                f"{provider_cfg.get('api_key_env', 'GROQ_API_KEY')} is not set; "
                "add it to .env to use the groq provider."
            )
        return ChatGroq(
            model=provider_cfg.get("model_name"),
            api_key=api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _build_ollama(self, provider_cfg: Dict[str, Any]):
        # PHASE 1.1: langchain_community's ChatOllama (the previous
        # fallback here) does not parse tool_calls out of Ollama
        # responses -- bind_tools()/tool_calls silently do nothing on
        # it, which would make Phase 1's native tool-calling agent loop
        # quietly answer with no tools ever firing. langchain-ollama's
        # ChatOllama has real tool-call support (verified directly
        # against its source and by round-tripping a mocked tool_call
        # response through bind_tools()->response.tool_calls).
        #
        # No fallback to the community version on import failure: per
        # this project's requirement not to let the agent "pretend it
        # worked", a missing/broken langchain-ollama install must be a
        # loud, clear error here rather than silently downgrading to a
        # ChatOllama that will never produce tool_calls.
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "The 'ollama' provider requires the 'langchain-ollama' package "
                "for native tool calling (bind_tools()/tool_calls) -- "
                "the older langchain_community ChatOllama does not support it. "
                "Install it with: pip install langchain-ollama==0.1.3 "
                "(pinned in requirements.txt; compatible with this project's "
                "langchain-core==0.2.38 and ollama==0.3.3 pins)."
            ) from exc

        host = os.environ.get(provider_cfg.get("host_env", "OLLAMA_HOST")) or "http://localhost:11434"
        api_key = os.environ.get(provider_cfg.get("api_key_env", "OLLAMA_API_KEY"))

        kwargs: Dict[str, Any] = {
            "model": provider_cfg.get("model_name"),
            "base_url": host,
            "temperature": self.temperature,
        }
        # Ollama Cloud (host on ollama.com) authenticates via a bearer
        # token header; a local Ollama daemon has no api key at all.
        # langchain_ollama.ChatOllama forwards client_kwargs straight
        # into ollama.Client(**client_kwargs) -> httpx.Client(**kwargs),
        # so this is the same "headers" shape as before the Phase 1.1
        # migration -- verified directly against ollama.Client's source.
        if api_key:
            kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}
        return ChatOllama(**kwargs)

    def _build_huggingface(self, provider_cfg: Dict[str, Any]):
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        api_key = os.environ.get(provider_cfg.get("api_key_env", "HUGGINGFACEHUB_API_TOKEN"))
        if not api_key:
            raise RuntimeError(
                f"{provider_cfg.get('api_key_env', 'HUGGINGFACEHUB_API_TOKEN')} is not set; "
                "add it to .env to use the huggingface provider."
            )
        endpoint = HuggingFaceEndpoint(
            repo_id=provider_cfg.get("model_name"),
            huggingfacehub_api_token=api_key,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,
        )
        return ChatHuggingFace(llm=endpoint)