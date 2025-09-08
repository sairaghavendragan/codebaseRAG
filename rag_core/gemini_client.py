import logging
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Type, Optional

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
    
    def __init__(self, google_api_key: str):
        self.client = genai.Client(api_key=google_api_key)
    
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

         

        except Exception as e:
            logging.error(f"Gemini API call failed: {e}", exc_info=True)
            return "I'm sorry, something went wrong while generating a response."
    def generate_structured_response(
        self,
        prompt: str,
        response_schema: Type[BaseModel],
        response_mime_type: str = "application/json"
    ) -> Optional[BaseModel]:
        """
        Generates a response from Gemini, enforcing a structured output format
        based on a Pydantic schema.

        Args:
            prompt (str): The prompt to send to Gemini.
            response_schema (Type[BaseModel]): The Pydantic model representing the
                                               desired JSON structure for the response.
            response_mime_type (str): The MIME type for the response, typically "application/json".

        Returns:
            Optional[BaseModel]: An instance of the response_schema Pydantic model if successful,
                                 otherwise None.
        """
        try:
            logging.debug(f"Sending structured prompt to Gemini (first 200 chars): {prompt[:200]}")

           
            

            response = self.client.models.generate_content(
                model=self.MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig( safety_settings=self.SAFETY_SETTINGS,
                                                    response_mime_type=  response_mime_type,
                                                    response_schema= response_schema,
                                                   )  
            )

            if response.parsed:
                logging.debug(f"Gemini structured response parsed successfully: {response.parsed}")
                return response.parsed
            else:
                logging.warning(f"Gemini returned an empty or unparseable structured response for prompt: {prompt[:100]}... Raw text: {response.text}")
                return None

         
        except Exception as e:
            logging.error(f"Gemini structured API call failed: {e}", exc_info=True)
            return None