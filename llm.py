"""
llm.py - Adapter Groq untuk port LLM NutriChef AI
"""

from typing import Dict, Iterator, List

from groq import Groq


class GroqLLM:
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int = 2048):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )
        for chunk in completion:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token
