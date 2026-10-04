#!/usr/bin/env python3
"""
LLM provider abstraction layer - reuses the CTITagger implementation
Supports multiple extensible LLM backends; currently implements OpenAI (gpt-5 and gpt-5-mini)
"""

# Try to import the LLM provider from CTITagger
# BEACON port: CTITagger is an installed sibling package, so the original
# sys.path insertion of the parent directory is no longer needed.
try:
    from CTITagger.core.llm_provider import LLMProvider, OpenAIProvider, create_llm_provider
except ImportError:
    # If the import fails, define the basic interface and implementation (fallback)
    from abc import ABC, abstractmethod
    from typing import Optional
    import os

    try:
        from beacon_cost.recorder import (
            monotonic_time,
            record_openai_error,
            record_openai_response,
        )
    except ImportError:  # Cost tracking must not affect fallback availability.
        from time import monotonic as monotonic_time

        def record_openai_error(**kwargs):
            return None

        def record_openai_response(**kwargs):
            return None
    
    class LLMProvider(ABC):
        """Abstract base class for LLM providers"""
        
        @abstractmethod
        def call(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            cost_module: str = "unknown",
            cost_stage: str = "unspecified",
            cost_attempt: Optional[int] = None,
            **kwargs
        ) -> str:
            """Call the LLM API"""
            pass
    
    class OpenAIProvider(LLMProvider):
        """OpenAI LLM provider implementation (fallback)"""
        
        def __init__(self, model: str = "gpt-5", api_key: Optional[str] = None):
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("Please install the openai library: pip install openai")
            
            self.model = model
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("An OpenAI API key is required (pass it as an argument or set the OPENAI_API_KEY environment variable)")
            
            self.client = OpenAI(api_key=self.api_key)
        
        def call(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: float = 0.3,
            max_tokens: Optional[int] = None,
            cost_module: str = "unknown",
            cost_stage: str = "unspecified",
            cost_attempt: Optional[int] = None,
            **kwargs
        ) -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            if max_tokens is None:
                if "gpt-5" in self.model.lower():
                    max_tokens = 8000
                elif "mini" in self.model.lower():
                    max_tokens = 4000
                else:
                    max_tokens = 4000
            
            try:
                api_params = {
                    "model": self.model,
                    "messages": messages,
                    **kwargs
                }
                
                if "gpt-5" in self.model.lower():
                    api_params["max_completion_tokens"] = max_tokens
                else:
                    api_params["max_tokens"] = max_tokens
                    api_params["temperature"] = temperature
                
                started_at = monotonic_time()
                try:
                    response = self.client.chat.completions.create(**api_params)
                except Exception as api_error:
                    record_openai_error(
                        module=cost_module,
                        stage=cost_stage,
                        endpoint="chat.completions",
                        model_requested=self.model,
                        error=api_error,
                        latency_seconds=monotonic_time() - started_at,
                        application_attempt=cost_attempt,
                    )
                    raise
                record_openai_response(
                    module=cost_module,
                    stage=cost_stage,
                    endpoint="chat.completions",
                    model_requested=self.model,
                    response=response,
                    latency_seconds=monotonic_time() - started_at,
                    application_attempt=cost_attempt,
                )
                choice = response.choices[0]
                content = choice.message.content
                
                if content is None or not content.strip():
                    raise RuntimeError(f"OpenAI API returned empty content")
                
                return content
            except Exception as e:
                raise RuntimeError(f"OpenAI API call failed: {str(e)}")
    
    def create_llm_provider(provider_type: str = "openai", model: str = "gpt-5", **kwargs):
        """Factory function: create an LLM provider instance"""
        if provider_type.lower() == "openai":
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            return OpenAIProvider(model=model, api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")

__all__ = ["LLMProvider", "create_llm_provider"]
