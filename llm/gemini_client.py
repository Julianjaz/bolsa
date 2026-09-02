import os
import json
import logging
from typing import Dict, Any
from google import genai
from pydantic import BaseModel, Field
from llm.prompts.v1 import PROMPT_V1

logger = logging.getLogger(__name__)

class GeminiResponseSchema(BaseModel):
    decision: str = Field(description="LONG | SHORT | HOLD")
    confidence: float = Field(description="Confidence between 0.0 and 1.0")
    setup_type: str = Field(description="BREAKOUT | PULLBACK | CONSOLIDATION | NONE")
    technical_confluence: str = Field(description="STRONG | MODERATE | WEAK")
    news_confluence: str = Field(description="POSITIVE | NEUTRAL | NEGATIVE | UNKNOWN")
    divergence_detected: bool = Field(description="True if news or regime diverge from technical setup")
    risk_assessment: str = Field(description="LOW | MODERATE | HIGH")
    reasoning: str = Field(description="Detailed explanation of your evaluation")

class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables. LLM logic will be bypassed.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)
            
    def evaluate_signal(self, analysis_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Envía el estado al LLM y obtiene una decisión estructurada.
        """
        if not self.client:
            return {
                "decision": "HOLD",
                "confidence": 0.0,
                "reasoning": "LLM_UNAVAILABLE (Missing API Key)",
                "error": True
            }
            
        try:
            state_json = json.dumps(analysis_state, indent=2)
            prompt = f"{PROMPT_V1}\n\nState to evaluate:\n{state_json}"
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": GeminiResponseSchema,
                    "temperature": 0.0 # Búsqueda de respuestas lo más determinísticas posibles
                }
            )
            
            result = json.loads(response.text)
            result['model_used'] = self.model_name
            result['prompt_version'] = "v1"
            return result
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return {
                "decision": "HOLD",
                "confidence": 0.0,
                "reasoning": f"LLM_ERROR: {str(e)}",
                "error": True
            }
