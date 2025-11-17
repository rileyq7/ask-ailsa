"""
LLM client with GPT-5.1 support.

This project now uses GPT-5.1 (released November 12, 2025) for enhanced
conversational quality with smart routing to optimize costs.

GPT-5.1 Features:
- More natural, conversational responses
- Better instruction following
- Adaptive reasoning with reasoning_effort parameter
- 50% fewer tokens used than GPT-4
- Smart query routing to optimize costs

Note: Prompt caching is NOT currently enabled.

Environment:
    OPENAI_API_KEY must be set

Usage:
    from src.llm.client import LLMClient

    client = LLMClient()  # Defaults to GPT-5.1 Instant
    response = client.chat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ])
"""

import os
import logging
from typing import List, Dict, Optional
from enum import Enum

from openai import OpenAI


logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available models (November 2025)."""
    GPT_5_1_INSTANT = "gpt-5.1-chat-latest"  # Conversational, warmer
    GPT_5_1_THINKING = "gpt-5.1"             # Advanced reasoning
    GPT_4O_MINI = "gpt-4o-mini"              # Cheap fallback for simple queries


class QueryComplexity(Enum):
    """Query complexity levels for routing."""
    SIMPLE = "simple"        # Deadlines, URLs, basic facts
    MODERATE = "moderate"    # Grant recommendations
    COMPLEX = "complex"      # Strategy, analysis, advice


class LLMClient:
    """
    Enhanced LLM client with GPT-5.1 and intelligent routing.

    GPT-5.1 was released November 12, 2025 and provides:
    - More natural conversation
    - Better instruction following
    - Adaptive reasoning
    - Cost efficiency with smart routing
    """

    def __init__(self, model: str = None):
        """
        Initialize LLM client with GPT-5.1 support.

        Args:
            model: OpenAI model name (default: "gpt-5.1-chat-latest")
                  Can also specify "gpt-4o-mini" for simple queries

        Raises:
            ValueError: If OPENAI_API_KEY not set
        """
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Get your key from: https://platform.openai.com/api-keys"
            )

        self.client = OpenAI(api_key=api_key)

        # Default to GPT-5.1 Instant for best conversational quality
        self.model = model or ModelType.GPT_5_1_INSTANT.value

        logger.info(f"LLM client initialized: {self.model}")
        if "5.1" in self.model:
            logger.info("✓ GPT-5.1 support enabled (Released Nov 12, 2025)")

    def analyze_query_complexity(self, query: str) -> QueryComplexity:
        """
        Determine query complexity for model routing.

        Simple queries can use GPT-4o-mini to save costs.
        Complex queries need GPT-5.1 for quality.
        """
        query_lower = query.lower()

        # Simple: Direct factual questions
        simple_patterns = [
            'deadline', 'when does', 'how much funding',
            'what time', 'link', 'url', 'website',
            'closes', 'opens', 'amount', 'contact'
        ]

        if any(pattern in query_lower for pattern in simple_patterns):
            return QueryComplexity.SIMPLE

        # Complex: Strategy, analysis, advice
        complex_patterns = [
            'strategy', 'how should', 'what do you think',
            'advice', 'recommend', 'which is better',
            'help me understand', 'explain why', 'compare'
        ]

        if any(pattern in query_lower for pattern in complex_patterns):
            return QueryComplexity.COMPLEX

        return QueryComplexity.MODERATE

    def select_model(self, query: str, force_model: Optional[str] = None) -> str:
        """
        Intelligently select the best model for the query.

        Cost optimization:
        - Simple queries: GPT-4o-mini ($0.15/1M input)
        - Moderate/Complex: GPT-5.1 (better quality)
        """
        if force_model:
            return force_model

        # If already using a specific model, stick with it
        if self.model != ModelType.GPT_5_1_INSTANT.value:
            return self.model

        complexity = self.analyze_query_complexity(query)

        # Use GPT-4o-mini only for very simple factual queries
        if complexity == QueryComplexity.SIMPLE:
            logger.info(f"Simple query detected, using GPT-4o-mini")
            return ModelType.GPT_4O_MINI.value

        # Everything else uses GPT-5.1 for quality
        logger.info(f"Query complexity: {complexity.value}, using GPT-5.1")
        return ModelType.GPT_5_1_INSTANT.value

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
        reasoning_effort: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> str:
        """
        Send chat completion request with GPT-5.1 optimizations.

        Args:
            messages: List of message dicts with "role" and "content"
                     Role must be "system", "user", or "assistant"
            temperature: Sampling temperature (0.0-2.0)
                        Note: GPT-5.1 only supports temperature=1 (default)
                        This param is used for GPT-4o-mini fallback
            max_tokens: Maximum tokens in response
            reasoning_effort: For GPT-5.1 - "none", "low", "medium", "high"
                            Defaults to "none" for simple queries, omitted for complex
            model_override: Override automatic model selection

        Returns:
            Response text from GPT

        Raises:
            Exception: If API call fails
        """
        # Extract user query for model selection
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        # Select appropriate model
        selected_model = model_override or self.select_model(user_query)

        # Build request parameters
        params = {
            "model": selected_model,
            "messages": messages,
        }

        # Add model-specific parameters
        if selected_model.startswith("gpt-5.1"):
            # GPT-5.1: Uses max_completion_tokens and reasoning_effort
            # Note: GPT-5.1 only supports temperature=1 (default), don't set it
            params["max_completion_tokens"] = max_tokens

            # Add reasoning effort control
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort
            elif user_query:
                # Auto-set reasoning_effort based on complexity
                complexity = self.analyze_query_complexity(user_query)
                if complexity == QueryComplexity.SIMPLE:
                    params["reasoning_effort"] = "none"  # Fast mode
                # For moderate/complex, omit reasoning_effort to use adaptive default

        elif selected_model.startswith("gpt-5"):
            # GPT-5 (non-5.1) only supports temperature=1 and max_completion_tokens
            params["max_completion_tokens"] = max_tokens
        else:
            # GPT-4o-mini and older models support temperature
            params["temperature"] = temperature
            params["max_tokens"] = max_tokens

        try:
            response = self.client.chat.completions.create(**params)

            # Log usage for cost tracking
            if response.usage:
                logger.info(
                    f"Token usage - Model: {selected_model}, "
                    f"Input: {response.usage.prompt_tokens}, "
                    f"Output: {response.usage.completion_tokens}"
                )

            # Debug: check response structure
            choice = response.choices[0]
            message = choice.message
            logger.debug(f"Finish reason: {choice.finish_reason}")

            # Handle refusal (GPT-5/5.1 feature)
            if hasattr(message, 'refusal') and message.refusal:
                logger.warning(f"Model refused to respond: {message.refusal}")
                return ""

            content = message.content
            if content is None:
                logger.warning("Model returned None content")
                return ""

            return content.strip()

        except Exception as e:
            logger.error(f"API call failed with {selected_model}: {e}")

            # Fallback to GPT-4o-mini if GPT-5.1 fails
            if "5.1" in selected_model:
                logger.info("Falling back to GPT-4o-mini")
                params["model"] = ModelType.GPT_4O_MINI.value
                params.pop("reasoning_effort", None)
                params["max_tokens"] = params.pop("max_completion_tokens", max_tokens)
                params["temperature"] = temperature

                try:
                    response = self.client.chat.completions.create(**params)
                    return response.choices[0].message.content.strip()
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise fallback_error

            raise

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
        reasoning_effort: Optional[str] = None,
        model_override: Optional[str] = None,
    ):
        """
        Stream chat completion with GPT-5.1 optimizations.

        Yields tokens as they arrive for real-time display.

        Args:
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            reasoning_effort: For GPT-5.1 - "none", "low", "medium", "high"
            model_override: Override automatic model selection

        Yields:
            String chunks as they arrive

        Raises:
            Exception: If API call fails
        """
        # Extract user query for model selection
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        # Select appropriate model
        selected_model = model_override or self.select_model(user_query)

        # Build request parameters
        params = {
            "model": selected_model,
            "messages": messages,
            "stream": True,
        }

        # Add model-specific parameters
        if selected_model.startswith("gpt-5.1"):
            # GPT-5.1 only supports temperature=1 (default), don't set it
            params["max_completion_tokens"] = max_tokens

            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort
            elif user_query:
                complexity = self.analyze_query_complexity(user_query)
                if complexity == QueryComplexity.SIMPLE:
                    params["reasoning_effort"] = "none"

        elif selected_model.startswith("gpt-5"):
            params["max_completion_tokens"] = max_tokens
        else:
            params["temperature"] = temperature
            params["max_tokens"] = max_tokens

        try:
            response = self.client.chat.completions.create(**params)

            for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

        except Exception as e:
            logger.error(f"Streaming failed with {selected_model}: {e}")

            # Fallback to GPT-4o-mini
            if "5.1" in selected_model:
                logger.info("Stream fallback to GPT-4o-mini")
                params["model"] = ModelType.GPT_4O_MINI.value
                params.pop("reasoning_effort", None)
                params["max_tokens"] = params.pop("max_completion_tokens", max_tokens)
                params["temperature"] = temperature

                try:
                    response = self.client.chat.completions.create(**params)
                    for chunk in response:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            yield delta.content
                except Exception as fallback_error:
                    logger.error(f"Stream fallback failed: {fallback_error}")
                    raise fallback_error
            else:
                raise
