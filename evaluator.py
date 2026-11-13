from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, field_validator
from models import JSONResume, LLMProvider
from llm_utils import initialize_llm_provider, extract_json_from_response
import logging
import json
import re

from prompt import (
    DEFAULT_MODEL,
    MODEL_PARAMETERS,
)
from prompts.template_manager import TemplateManager

logger = logging.getLogger(__name__)


class ResumeEvaluator:
    def __init__(
        self,
        role,
        evaluation_model,
        model_name: str = DEFAULT_MODEL,
        model_params: Optional[dict] = None,
        provider: Optional[LLMProvider] = None,
    ):
        if not model_name:
            raise ValueError("Model name cannot be empty")

        self.role = role
        self.evaluation_model = evaluation_model
        self.model_name = model_name
        self.model_params = (
            model_params
            if model_params is not None
            else MODEL_PARAMETERS.get(model_name, {"temperature": 0.5, "top_p": 0.9})
        )
        self.template_manager = TemplateManager()
        if provider is not None:
            self.provider = provider
        else:
            self._initialize_llm_provider()

    def _initialize_llm_provider(self):
        """Initialize the appropriate LLM provider based on the model."""
        self.provider = initialize_llm_provider(self.model_name)

    def _load_evaluation_prompt(self, resume_text: str) -> str:
        return self.template_manager.render_string(
            self.role.criteria_source, text_content=resume_text
        )

    def evaluate_resume(self, resume_text: str) -> BaseModel:
        self._last_resume_text = resume_text
        full_prompt = self._load_evaluation_prompt(resume_text)
        try:
            system_message = self.template_manager.render_string(
                self.role.system_message_source,
                bonus_rules=self.role.bonus_rules,
                bonus_max=self.role.bonus_max,
            )

            # Prepare chat parameters
            chat_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": full_prompt},
                ],
                "options": {
                    "stream": False,
                    **self.model_params,
                },
            }

            # Add format parameter for structured output
            kwargs = {"format": self.evaluation_model.model_json_schema()}
            # Use the appropriate provider to make the API call
            response = self.provider.chat(**chat_params, **kwargs)

            if not isinstance(response, dict) or not isinstance(
                response.get("message"), dict
            ):
                raise ValueError(f"Malformed provider response shape: {response!r}")

            raw_content = response["message"].get("content")
            if raw_content is None:
                raise ValueError("Provider response message content is None")

            response_text = extract_json_from_response(raw_content)
            logger.debug(f"🔤 Prompt response: {response_text}")

            evaluation_dict = json.loads(response_text)
            evaluation_data = self.evaluation_model(**evaluation_dict)

            return evaluation_data

        except Exception as e:
            logger.error(f"Error evaluating resume: {str(e)}")
            raise
