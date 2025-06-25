import os
from dataclasses import dataclass

import backoff
from google.genai import Client, types

g_client = Client(api_key=os.environ["GOOGLE_API_KEY"])

@dataclass
class GenAIResponse:
    text: str
    total_tokens: str
    model: str

@backoff.on_exception(backoff.expo, RuntimeError)
def score_dialog_googleai(user_prompt, system_prompt, api_client=g_client, model='gemini-1.5-flash'):
    """model: gemini-1.5-flash, gemini-2.0-flash-001"""
    response = api_client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
        ),
    )
    response = GenAIResponse(text=response.text, total_tokens=response.usage_metadata.total_token_count, model=model)
    return response