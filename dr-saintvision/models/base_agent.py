"""
Base Agent class for all AI agents in DR-Saintvision
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all AI agents"""

    def __init__(
        self,
        model_name: str,
        use_quantization: bool = True,
        device_map: str = "auto",
        max_memory: Optional[Dict] = None
    ):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.device_map = device_map
        self.max_memory = max_memory
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    def load_model(self):
        """Load the model and tokenizer"""
        if self._is_loaded:
            logger.info(f"Model {self.model_name} already loaded")
            return

        logger.info(f"Loading model: {self.model_name}")

        # Configure quantization for memory efficiency
        if self.use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            bnb_config = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_kwargs = {
                "device_map": self.device_map,
                "trust_remote_code": True,
                "torch_dtype": torch.float16
            }

            if bnb_config:
                model_kwargs["quantization_config"] = bnb_config

            if self.max_memory:
                model_kwargs["max_memory"] = self.max_memory

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            self._is_loaded = True
            logger.info(f"Model {self.model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True
    ) -> str:
        """Generate a response from the model"""
        if not self._is_loaded:
            self.load_model()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        # Move inputs to the same device as the model
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the input prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response

    @abstractmethod
    async def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """Process a query and return results"""
        pass

    def unload_model(self):
        """Unload the model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._is_loaded = False

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Model {self.model_name} unloaded")

    def __del__(self):
        self.unload_model()
