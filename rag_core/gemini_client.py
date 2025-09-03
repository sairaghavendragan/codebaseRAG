import logging
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeminiClient:
    MODEL_NAME = "gemini-2.5-flash"
    
    SAFETY_SETTINGS = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]
    
    GENERATION_CONFIG = types.GenerateContentConfig(
        safety_settings=SAFETY_SETTINGS,
        # add other params here if needed
    )
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
    
    def generate_response(self, prompt: str) -> str:
        try:
            logging.debug(f"Sending prompt to Gemini (first 200 chars): {prompt[:200]}")

            response = self.client.models.generate_content(
                model=self.MODEL_NAME,
                contents=prompt,
                config=self.GENERATION_CONFIG
            )

            text = response.text or ""
            if not text.strip():
                logging.warning("Gemini returned an empty or whitespace-only response.")
                return "I'm sorry, I couldn't generate a response."

            logging.debug(f"Gemini response (first 200 chars): {text[:200]}")
            return text

        except genai.types.BlockedPromptException as e:
            logging.error(f"Prompt was blocked due to safety filters: {e}")
            return "I'm sorry, your request was blocked due to safety concerns."

        except Exception as e:
            logging.error(f"Gemini API call failed: {e}", exc_info=True)
            return "I'm sorry, something went wrong while generating a response."
