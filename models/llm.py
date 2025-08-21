"""
LLM wrapper using a local GGUF model with llama-cpp-python
Optimized for local inference on 8GB systems
"""

import llama_cpp
import asyncio
import httpx
from typing import Dict, Any, Optional, List
import logging
import json
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class LLMModel:
    """Wrapper for local GGUF LLM client using llama-cpp-python"""

    def __init__(self, 
                 model_path: str = "/Users/govindkumar/AI Market Analysis/ai_market_analyst/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"):
        """
        Initialize LLM client

        Args:
            model_path: Path to the GGUF model file
        """
        self.model_path = model_path
        
        # Verify that the model file exists before proceeding
        if not Path(self.model_path).exists():
            logger.error(f"FATAL: Model file not found at '{self.model_path}'")
            raise FileNotFoundError(f"The specified model file was not found. Please ensure the path is correct.")

        # Initialize the Llama client from the local model file
        # n_gpu_layers=-1 attempts to offload all layers to a GPU if available, improving speed.
        # Set to 0 if you want to run on CPU only.
        self.client = llama_cpp.Llama(
            model_path=self.model_path,
            n_ctx=2048,        # Context length
            n_gpu_layers=0,   # Offload all possible layers to GPU
            verbose=False      # Suppress talkative output
        )

        # Default generation parameters optimized for 8GB systems
        self.default_options = {
            "temperature": 0.1,      # Low temperature for consistency
            "top_p": 0.9,           # Nucleus sampling
            "top_k": 40,            # Top-k sampling
            "repeat_penalty": 1.1,  # Prevent repetition
            "max_tokens": 512,      # Max tokens to generate (equivalent to num_predict)
        }

        logger.info(f"Initialized LLM client from local model: {self.model_path}")

    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate text using the local LLM

        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            options: Generation options override

        Returns:
            Dictionary with response and metadata
        """
        try:
            # Merge and adapt options for llama-cpp-python
            gen_options = {**self.default_options, **(options or {})}
            if "num_predict" in gen_options:
                gen_options["max_tokens"] = gen_options.pop("num_predict")

            # Prepare messages in the format expected by create_chat_completion
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Define the synchronous blocking call for use with asyncio
            def _blocking_generate():
                start_time = time.time()
                response = self.client.create_chat_completion(
                    messages=messages,
                    **gen_options
                )
                end_time = time.time()
                return response, end_time - start_time

            # Run the synchronous call in a separate thread to avoid blocking the event loop
            response, duration_sec = await asyncio.to_thread(_blocking_generate)
            
            # Convert response to the format expected by the other tools
            return {
                "response": response['choices'][0]['message']['content'],
                "model": Path(self.model_path).name,
                "total_duration": int(duration_sec * 1_000_000_000), # Convert to nanoseconds for compatibility
                "eval_count": response['usage']['completion_tokens'],
                "prompt_eval_count": response['usage']['prompt_tokens']
            }

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    async def generate_json(self, 
                           prompt: str,
                           system_prompt: Optional[str] = None,
                           schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate JSON response (useful for data extraction)

        Args:
            prompt: User prompt requesting JSON
            system_prompt: System prompt
            schema: Expected JSON schema (used in prompt)

        Returns:
            Parsed JSON response
        """
        try:
            # Enhanced system prompt for JSON generation
            json_system = (system_prompt or "") + """

Please respond with valid JSON only. Do not include any explanations or additional text.
Ensure the JSON is properly formatted and valid."""

            if schema:
                json_system += f"\nExpected JSON schema: {json.dumps(schema, indent=2)}"

            # Use the json_object response format supported by Llama 3
            response = await self.generate(
                prompt=prompt,
                system_prompt=json_system,
                options={
                    "temperature": 0.0, 
                    "response_format": {"type": "json_object"}
                }
            )

            # Try to parse JSON from the response
            try:
                json_response = json.loads(response["response"])
                response["parsed_json"] = json_response
                response["is_valid_json"] = True
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")
                response["parsed_json"] = None
                response["is_valid_json"] = False

            return response

        except Exception as e:
            logger.error(f"JSON generation failed: {e}")
            raise