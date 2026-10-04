#!/usr/bin/env python3
"""
LLM provider abstraction layer - supports multiple extensible LLM interfaces
Currently implements OpenAI (gpt-5 and gpt-5-mini), but the interface is designed so other providers can be added later
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import os

try:
    from beacon_cost.recorder import (
        monotonic_time,
        record_openai_error,
        record_openai_response,
    )
except ImportError:  # Cost tracking must never make the provider unavailable.
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
        """
        Call the LLM API
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            **kwargs: Other parameters (e.g. temperature, max_tokens)
        
        Returns:
            Text content returned by the LLM
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation"""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize the OpenAI provider
        
        Args:
            model: Model name (gpt-5, gpt-5-mini, etc. are supported)
            api_key: OpenAI API key (if None, read from the environment variable)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install the openai library: pip install openai")
        
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("An OpenAI API key is required (via the api_key argument or the OPENAI_API_KEY environment variable)")
        
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
        """
        Call the OpenAI API
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Temperature parameter
            max_tokens: Maximum number of output tokens
            **kwargs: Other OpenAI API parameters
        
        Returns:
            Text content returned by the LLM
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Set the default max_tokens based on the model
        if max_tokens is None:
            if "gpt-5" in self.model.lower():
                max_tokens = 8000
            elif "mini" in self.model.lower():
                max_tokens = 4000
            else:
                max_tokens = 4000
        
        try:
            # gpt-5 models require max_completion_tokens instead of max_tokens
            # gpt-5 models do not support a custom temperature; only the default is allowed
            api_params = {
                "model": self.model,
                "messages": messages,
                **kwargs
            }
            
            # Choose the correct parameters based on the model type
            if "gpt-5" in self.model.lower():
                # gpt-5 uses max_completion_tokens; temperature is not passed (default of 1 is used)
                api_params["max_completion_tokens"] = max_tokens
            else:
                # Other models use max_tokens and a custom temperature
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
            finish_reason = getattr(choice, "finish_reason", None)
            
            if content is None:
                error_msg = f"OpenAI API returned empty content (content is None)"
                if finish_reason:
                    error_msg += f", finish_reason: {finish_reason}"
                raise RuntimeError(error_msg)
            
            if not content.strip():
                error_msg = f"OpenAI API returned empty string"
                if finish_reason:
                    error_msg += f", finish_reason: {finish_reason}"
                raise RuntimeError(error_msg)
            
            return content
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")


def create_llm_provider(provider_type: str = "openai", model: str = "gpt-4o", **kwargs) -> LLMProvider:
    """
    Factory function: create an LLM provider instance
    
    Args:
        provider_type: Provider type ("openai", etc.)
        model: Model name
        **kwargs: Provider-specific parameters
    
    Returns:
        LLMProvider instance
    """
    if provider_type.lower() == "openai":
        return OpenAIProvider(model=model, **kwargs)
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")
