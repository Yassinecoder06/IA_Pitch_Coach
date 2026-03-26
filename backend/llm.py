"""
LLM Module
==========
Lightweight multi-provider LLM integration.
Providers are auto-enabled from .env keys/endpoints.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx


DEFAULT_LLM = os.getenv("DEFAULT_LLM", "ollama").lower()
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "")


class MessageRole(str, Enum):
	SYSTEM = "system"
	USER = "user"
	ASSISTANT = "assistant"


@dataclass
class Message:
	role: MessageRole
	content: str
	name: Optional[str] = None

	def to_dict(self) -> Dict[str, str]:
		result = {"role": self.role.value, "content": self.content}
		if self.name:
			result["name"] = self.name
		return result

	@classmethod
	def system(cls, content: str) -> "Message":
		return cls(role=MessageRole.SYSTEM, content=content)

	@classmethod
	def user(cls, content: str) -> "Message":
		return cls(role=MessageRole.USER, content=content)

	@classmethod
	def assistant(cls, content: str) -> "Message":
		return cls(role=MessageRole.ASSISTANT, content=content)


class BaseProvider:
	name: str

	async def stream(
		self,
		messages: List[Message],
		model: Optional[str] = None,
		temperature: float = 0.7,
		max_tokens: int = 512,
	) -> AsyncGenerator[str, None]:
		raise NotImplementedError

	async def generate(
		self,
		messages: List[Message],
		model: Optional[str] = None,
		temperature: float = 0.7,
		max_tokens: int = 512,
	) -> str:
		chunks: List[str] = []
		async for chunk in self.stream(messages, model=model, temperature=temperature, max_tokens=max_tokens):
			chunks.append(chunk)
		return "".join(chunks)

	async def check_availability(self) -> Dict[str, Any]:
		raise NotImplementedError

	async def list_models(self) -> List[str]:
		raise NotImplementedError


async def _read_error_body(response: httpx.Response) -> str:
	try:
		return (await response.aread()).decode("utf-8", errors="replace").strip()
	except Exception:
		return ""


class OllamaProvider(BaseProvider):
	name = "ollama"

	def __init__(self):
		self.endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
		self.default_model = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
		self._cached_models: List[str] = []

	async def stream(
		self,
		messages: List[Message],
		model: Optional[str] = None,
		temperature: float = 0.7,
		max_tokens: int = 512,
	) -> AsyncGenerator[str, None]:
		payload = {
			"model": model or self.default_model,
			"messages": [m.to_dict() if isinstance(m, Message) else m for m in messages],
			"stream": True,
			"options": {
				"temperature": temperature,
				"num_predict": max_tokens,
				"top_p": 1.0,
				"num_ctx": 2048,
			},
		}

		try:
			async with httpx.AsyncClient() as client:
				async with client.stream("POST", f"{self.endpoint}/api/chat", json=payload, timeout=120.0) as response:
					if response.status_code == 404:
						chosen_model = payload["model"]
						yield f"Model '{chosen_model}' not found. Run: ollama pull {chosen_model}"
						return

					response.raise_for_status()

					async for line in response.aiter_lines():
						if not line:
							continue
						try:
							data = json.loads(line)
						except json.JSONDecodeError:
							continue

						content = data.get("message", {}).get("content", "")
						if content:
							yield content

						if data.get("done", False):
							break

		except httpx.ConnectError:
			yield "Error: Cannot connect to Ollama. Ensure 'ollama serve' is running."
		except httpx.TimeoutException:
			yield "Error: Request timed out. The model may be overloaded."
		except Exception as exc:
			yield f"Error: {exc}"

	async def check_availability(self) -> Dict[str, Any]:
		try:
			async with httpx.AsyncClient() as client:
				response = await client.get(f"{self.endpoint}/api/tags", timeout=5.0)

			if response.status_code != 200:
				return {"available": False, "message": f"Unexpected response: {response.status_code}", "models": []}

			data = response.json()
			models = [m.get("name", "") for m in data.get("models", [])]
			self._cached_models = models
			return {"available": True, "message": "Ollama is running", "models": models}
		except httpx.ConnectError:
			return {"available": False, "message": "Cannot connect to Ollama. Run 'ollama serve' first.", "models": []}
		except Exception as exc:
			return {"available": False, "message": str(exc), "models": []}

	async def list_models(self) -> List[str]:
		status = await self.check_availability()
		if status.get("available"):
			return status.get("models", [])
		return self._cached_models


class GoogleProvider(BaseProvider):
	name = "google"
	DEFAULT_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

	def __init__(self):
		self.api_key = os.getenv("GOOGLE_API_KEY", "")
		self.endpoint = os.getenv("GOOGLE_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta")
		self.default_model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
		self._cached_models: List[str] = []

	def _prepare_messages(self, messages: List[Message]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
		system_instruction = None
		contents: List[Dict[str, Any]] = []

		for msg in messages:
			role = msg.role.value if isinstance(msg, Message) else msg.get("role")
			content = msg.content if isinstance(msg, Message) else msg.get("content", "")

			if role == MessageRole.SYSTEM.value:
				system_instruction = content
			else:
				contents.append({
					"role": "user" if role == MessageRole.USER.value else "model",
					"parts": [{"text": content}],
				})

		return system_instruction, contents

	async def stream(
		self,
		messages: List[Message],
		model: Optional[str] = None,
		temperature: float = 0.7,
		max_tokens: int = 512,
	) -> AsyncGenerator[str, None]:
		if not self.api_key:
			yield "Error: Google API key not configured. Set GOOGLE_API_KEY in .env"
			return

		chosen_model = model or self.default_model
		system_instruction, contents = self._prepare_messages(messages)

		payload: Dict[str, Any] = {
			"contents": contents,
			"generationConfig": {
				"temperature": temperature,
				"maxOutputTokens": max_tokens,
				"topP": 1.0,
			},
		}
		if system_instruction:
			payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

		url = f"{self.endpoint}/models/{chosen_model}:streamGenerateContent?key={self.api_key}&alt=sse"

		max_retries = 2
		attempt = 0
		while True:
			try:
				async with httpx.AsyncClient() as client:
					async with client.stream("POST", url, json=payload, timeout=120.0) as response:
						response.raise_for_status()

						async for line in response.aiter_lines():
							if not line or not line.startswith("data: "):
								continue
							data_str = line[6:]
							try:
								data = json.loads(data_str)
							except json.JSONDecodeError:
								continue

							candidates = data.get("candidates", [])
							if not candidates:
								continue

							parts = candidates[0].get("content", {}).get("parts", [])
							for part in parts:
								text = part.get("text", "")
								if text:
									yield text
				return

			except httpx.ConnectError:
				yield "Error: Cannot connect to Google Gemini API"
				return
			except httpx.HTTPStatusError as err:
				status = err.response.status_code
				if status == 503 and attempt < max_retries:
					await asyncio.sleep(attempt + 1)
					attempt += 1
					continue

				body = await _read_error_body(err.response)
				if not body:
					if status == 503:
						body = "Google Gemini service temporarily unavailable. Try again in a few seconds or switch model/provider."
					else:
						body = getattr(err.response, "reason_phrase", "") or "No response body returned by provider"
				yield f"Error: {status} - {body}"
				return
			except Exception as exc:
				yield f"Error: {exc}"
				return

	async def check_availability(self) -> Dict[str, Any]:
		if not self.api_key:
			return {"available": False, "message": "Google API key not configured", "models": []}

		try:
			url = f"{self.endpoint}/models?key={self.api_key}"
			async with httpx.AsyncClient() as client:
				response = await client.get(url, timeout=10.0)

			if response.status_code != 200:
				return {"available": False, "message": f"API returned status {response.status_code}", "models": []}

			data = response.json()
			models = [
				m.get("name", "").replace("models/", "")
				for m in data.get("models", [])
				if "generateContent" in m.get("supportedGenerationMethods", [])
			]
			self._cached_models = models
			return {"available": True, "message": "Google Gemini API is available", "models": models}
		except Exception as exc:
			return {"available": False, "message": str(exc), "models": []}

	async def list_models(self) -> List[str]:
		status = await self.check_availability()
		if status.get("available"):
			models = status.get("models", [])
			return models if models else self.DEFAULT_MODELS
		return self._cached_models if self._cached_models else self.DEFAULT_MODELS


class OpenAICompatibleProvider(BaseProvider):
	def __init__(
		self,
		name: str,
		endpoint: str,
		api_key: str,
		default_models: List[str],
		default_model: Optional[str] = None,
	):
		self.name = name
		self.endpoint = endpoint.rstrip("/")
		self.api_key = api_key
		self.default_models = default_models
		self.default_model = default_model or (default_models[0] if default_models else "")
		self._cached_models: List[str] = []

	def _headers(self) -> Dict[str, str]:
		return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

	async def stream(
		self,
		messages: List[Message],
		model: Optional[str] = None,
		temperature: float = 0.7,
		max_tokens: int = 512,
	) -> AsyncGenerator[str, None]:
		chosen_model = model or self.default_model
		if not chosen_model:
			yield "Error: No model configured for provider"
			return

		payload = {
			"model": chosen_model,
			"messages": [m.to_dict() if isinstance(m, Message) else m for m in messages],
			"temperature": temperature,
			"max_tokens": max_tokens,
			"stream": True,
		}

		try:
			async with httpx.AsyncClient() as client:
				async with client.stream(
					"POST",
					f"{self.endpoint}/chat/completions",
					headers=self._headers(),
					json=payload,
					timeout=120.0,
				) as response:
					response.raise_for_status()

					async for line in response.aiter_lines():
						if not line or not line.startswith("data: "):
							continue
						data_str = line[6:]
						if data_str.strip() == "[DONE]":
							break
						try:
							data = json.loads(data_str)
						except json.JSONDecodeError:
							continue

						delta = data.get("choices", [{}])[0].get("delta", {})
						content = delta.get("content", "")
						if content:
							yield content

		except httpx.ConnectError:
			yield f"Error: Cannot connect to {self.name} API"
		except httpx.HTTPStatusError as err:
			body = await _read_error_body(err.response)
			if not body:
				body = getattr(err.response, "reason_phrase", "") or "No response body returned by provider"
			yield f"Error: {err.response.status_code} - {body}"
		except Exception as exc:
			yield f"Error: {exc}"

	async def check_availability(self) -> Dict[str, Any]:
		if not self.api_key:
			return {"available": False, "message": f"{self.name} API key not configured", "models": []}

		try:
			async with httpx.AsyncClient() as client:
				response = await client.get(f"{self.endpoint}/models", headers=self._headers(), timeout=10.0)

			if response.status_code != 200:
				return {"available": False, "message": f"API returned status {response.status_code}", "models": []}

			data = response.json()
			models = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
			self._cached_models = models
			return {"available": True, "message": f"{self.name} API is available", "models": models}
		except Exception as exc:
			return {"available": False, "message": str(exc), "models": []}

	async def list_models(self) -> List[str]:
		status = await self.check_availability()
		if status.get("available"):
			models = status.get("models", [])
			return models if models else self.default_models
		return self._cached_models if self._cached_models else self.default_models


class AnthropicProvider(BaseProvider):
	name = "anthropic"
	DEFAULT_MODELS = [
		"claude-sonnet-4-20250514",
		"claude-3-5-sonnet-20241022",
		"claude-3-haiku-20240307",
	]

	def __init__(self):
		self.endpoint = os.getenv("ANTHROPIC_ENDPOINT", "https://api.anthropic.com")
		self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
		self.default_model = os.getenv("ANTHROPIC_MODEL", self.DEFAULT_MODELS[0])

	def _headers(self) -> Dict[str, str]:
		return {
			"x-api-key": self.api_key,
			"anthropic-version": "2023-06-01",
			"Content-Type": "application/json",
		}

	def _prepare_messages(self, messages: List[Message]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
		system_prompt = None
		chat_messages: List[Dict[str, Any]] = []

		for msg in messages:
			role = msg.role.value if isinstance(msg, Message) else msg.get("role")
			content = msg.content if isinstance(msg, Message) else msg.get("content", "")
			if role == MessageRole.SYSTEM.value:
				system_prompt = content
			else:
				chat_messages.append({"role": role, "content": content})

		return system_prompt, chat_messages

	async def stream(
		self,
		messages: List[Message],
		model: Optional[str] = None,
		temperature: float = 0.7,
		max_tokens: int = 512,
	) -> AsyncGenerator[str, None]:
		if not self.api_key:
			yield "Error: Anthropic API key not configured"
			return

		chosen_model = model or self.default_model
		system_prompt, chat_messages = self._prepare_messages(messages)

		payload: Dict[str, Any] = {
			"model": chosen_model,
			"messages": chat_messages,
			"max_tokens": max_tokens,
			"temperature": temperature,
			"stream": True,
		}
		if system_prompt:
			payload["system"] = system_prompt

		try:
			async with httpx.AsyncClient() as client:
				async with client.stream(
					"POST",
					f"{self.endpoint}/v1/messages",
					headers=self._headers(),
					json=payload,
					timeout=120.0,
				) as response:
					response.raise_for_status()

					async for line in response.aiter_lines():
						if not line or not line.startswith("data: "):
							continue
						data_str = line[6:]
						try:
							data = json.loads(data_str)
						except json.JSONDecodeError:
							continue

						event_type = data.get("type")
						if event_type == "content_block_delta":
							delta = data.get("delta", {})
							if delta.get("type") == "text_delta":
								text = delta.get("text", "")
								if text:
									yield text
						elif event_type == "message_stop":
							break

		except httpx.ConnectError:
			yield "Error: Cannot connect to Anthropic API"
		except httpx.HTTPStatusError as err:
			body = await _read_error_body(err.response)
			if not body:
				body = getattr(err.response, "reason_phrase", "") or "No response body returned by provider"
			yield f"Error: {err.response.status_code} - {body}"
		except Exception as exc:
			yield f"Error: {exc}"

	async def check_availability(self) -> Dict[str, Any]:
		if not self.api_key:
			return {"available": False, "message": "Anthropic API key not configured", "models": []}

		try:
			async with httpx.AsyncClient() as client:
				response = await client.post(
					f"{self.endpoint}/v1/messages",
					headers=self._headers(),
					json={
						"model": self.default_model,
						"max_tokens": 1,
						"messages": [{"role": "user", "content": "ping"}],
					},
					timeout=10.0,
				)

			if response.status_code in (200, 400):
				return {"available": True, "message": "Anthropic API is available", "models": self.DEFAULT_MODELS}
			if response.status_code == 401:
				return {"available": False, "message": "Invalid API key", "models": []}
			return {"available": False, "message": f"API returned status {response.status_code}", "models": []}
		except Exception as exc:
			return {"available": False, "message": str(exc), "models": []}

	async def list_models(self) -> List[str]:
		return self.DEFAULT_MODELS


class AzureOpenAIProvider(BaseProvider):
	name = "azure"

	def __init__(self):
		self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
		self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
		self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
		self.default_model = os.getenv("AZURE_OPENAI_DEPLOYMENT", DEFAULT_MODEL)

	def _headers(self) -> Dict[str, str]:
		return {"api-key": self.api_key, "Content-Type": "application/json"}

	def _url(self, deployment: str) -> str:
		return f"{self.endpoint}/openai/deployments/{deployment}/chat/completions?api-version={self.api_version}"

	async def stream(
		self,
		messages: List[Message],
		model: Optional[str] = None,
		temperature: float = 0.7,
		max_tokens: int = 512,
	) -> AsyncGenerator[str, None]:
		if not self.api_key or not self.endpoint:
			yield "Error: Azure OpenAI endpoint or API key not configured"
			return

		deployment = model or self.default_model
		if not deployment:
			yield "Error: Azure deployment name is not configured (set AZURE_OPENAI_DEPLOYMENT or pick model in UI)"
			return

		payload = {
			"messages": [m.to_dict() if isinstance(m, Message) else m for m in messages],
			"temperature": temperature,
			"max_tokens": max_tokens,
			"stream": True,
		}

		try:
			async with httpx.AsyncClient() as client:
				async with client.stream(
					"POST",
					self._url(deployment),
					headers=self._headers(),
					json=payload,
					timeout=120.0,
				) as response:
					response.raise_for_status()
					async for line in response.aiter_lines():
						if not line or not line.startswith("data: "):
							continue
						data_str = line[6:]
						if data_str.strip() == "[DONE]":
							break
						try:
							data = json.loads(data_str)
						except json.JSONDecodeError:
							continue
						delta = data.get("choices", [{}])[0].get("delta", {})
						content = delta.get("content", "")
						if content:
							yield content
		except httpx.ConnectError:
			yield "Error: Cannot connect to Azure OpenAI"
		except httpx.HTTPStatusError as err:
			body = await _read_error_body(err.response)
			if not body:
				body = getattr(err.response, "reason_phrase", "") or "No response body returned by provider"
			yield f"Error: {err.response.status_code} - {body}"
		except Exception as exc:
			yield f"Error: {exc}"

	async def check_availability(self) -> Dict[str, Any]:
		if not self.api_key or not self.endpoint:
			return {"available": False, "message": "Azure OpenAI endpoint or API key not configured", "models": []}

		try:
			async with httpx.AsyncClient() as client:
				response = await client.get(
					f"{self.endpoint}/openai/deployments?api-version={self.api_version}",
					headers=self._headers(),
					timeout=10.0,
				)

			if response.status_code != 200:
				return {"available": False, "message": f"API returned status {response.status_code}", "models": []}

			data = response.json()
			deployments = [d.get("id", "") for d in data.get("data", []) if d.get("id")]
			return {"available": True, "message": "Azure OpenAI is available", "models": deployments}
		except Exception as exc:
			return {"available": False, "message": str(exc), "models": []}

	async def list_models(self) -> List[str]:
		status = await self.check_availability()
		if status.get("available"):
			return status.get("models", [])
		return []


def _openai_compat_from_env(
	provider_name: str,
	endpoint_env: str,
	key_env: str,
	default_endpoint: str,
	default_models: List[str],
	model_env: Optional[str] = None,
) -> Optional[OpenAICompatibleProvider]:
	api_key = os.getenv(key_env, "")
	if not api_key:
		return None

	endpoint = os.getenv(endpoint_env, default_endpoint)
	chosen_model = os.getenv(model_env, "") if model_env else ""

	return OpenAICompatibleProvider(
		name=provider_name,
		endpoint=endpoint,
		api_key=api_key,
		default_models=default_models,
		default_model=chosen_model,
	)


def _build_providers() -> Dict[str, BaseProvider]:
	providers: Dict[str, BaseProvider] = {
		"ollama": OllamaProvider(),
	}

	google_key = os.getenv("GOOGLE_API_KEY", "")
	if google_key:
		providers["google"] = GoogleProvider()

	anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
	if anthropic_key:
		providers["anthropic"] = AnthropicProvider()

	azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
	azure_key = os.getenv("AZURE_OPENAI_API_KEY", "")
	if azure_endpoint and azure_key:
		providers["azure"] = AzureOpenAIProvider()

	openai = _openai_compat_from_env(
		provider_name="openai",
		endpoint_env="OPENAI_ENDPOINT",
		key_env="OPENAI_API_KEY",
		default_endpoint="https://api.openai.com/v1",
		default_models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
		model_env="OPENAI_MODEL",
	)
	if openai:
		providers["openai"] = openai

	deepseek = _openai_compat_from_env(
		provider_name="deepseek",
		endpoint_env="DEEPSEEK_ENDPOINT",
		key_env="DEEPSEEK_API_KEY",
		default_endpoint="https://api.deepseek.com",
		default_models=["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
		model_env="DEEPSEEK_MODEL",
	)
	if deepseek:
		providers["deepseek"] = deepseek

	mistral = _openai_compat_from_env(
		provider_name="mistral",
		endpoint_env="MISTRAL_ENDPOINT",
		key_env="MISTRAL_API_KEY",
		default_endpoint="https://api.mistral.ai/v1",
		default_models=["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
		model_env="MISTRAL_MODEL",
	)
	if mistral:
		providers["mistral"] = mistral

	grok = _openai_compat_from_env(
		provider_name="grok",
		endpoint_env="GROK_ENDPOINT",
		key_env="GROK_API_KEY",
		default_endpoint="https://api.x.ai/v1",
		default_models=["grok-2", "grok-2-mini", "grok-beta"],
		model_env="GROK_MODEL",
	)
	if grok:
		providers["grok"] = grok

	alibaba = _openai_compat_from_env(
		provider_name="alibaba",
		endpoint_env="ALIBABA_ENDPOINT",
		key_env="ALIBABA_API_KEY",
		default_endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
		default_models=["qwen-turbo", "qwen-plus", "qwen-max"],
		model_env="ALIBABA_MODEL",
	)
	if alibaba:
		providers["alibaba"] = alibaba

	modelscope = _openai_compat_from_env(
		provider_name="modelscope",
		endpoint_env="MODELSCOPE_ENDPOINT",
		key_env="MODELSCOPE_API_KEY",
		default_endpoint="https://api-inference.modelscope.cn/v1",
		default_models=["qwen2.5-coder-32b-instruct"],
		model_env="MODELSCOPE_MODEL",
	)
	if modelscope:
		providers["modelscope"] = modelscope

	moonshot = _openai_compat_from_env(
		provider_name="moonshot",
		endpoint_env="MOONSHOT_ENDPOINT",
		key_env="MOONSHOT_API_KEY",
		default_endpoint="https://api.moonshot.cn/v1",
		default_models=["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
		model_env="MOONSHOT_MODEL",
	)
	if moonshot:
		providers["moonshot"] = moonshot

	siliconflow_endpoint = os.getenv("SiliconFLOW_ENDPOINT", os.getenv("SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/"))
	siliconflow_key = os.getenv("SiliconFLOW_API_KEY", os.getenv("SILICONFLOW_API_KEY", ""))
	if siliconflow_key:
		providers["siliconflow"] = OpenAICompatibleProvider(
			name="siliconflow",
			endpoint=siliconflow_endpoint,
			api_key=siliconflow_key,
			default_models=["deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-72B-Instruct"],
			default_model=os.getenv("SILICONFLOW_MODEL", ""),
		)

	return providers


_providers: Dict[str, BaseProvider] = _build_providers()


def get_provider(name: Optional[str] = None) -> Optional[BaseProvider]:
	if name:
		return _providers.get(name.lower())

	if DEFAULT_LLM in _providers:
		return _providers[DEFAULT_LLM]

	return _providers.get("ollama")


def list_providers() -> List[str]:
	return list(_providers.keys())


async def get_available_providers() -> Dict[str, Dict[str, Any]]:
	result: Dict[str, Dict[str, Any]] = {}
	for name, provider in _providers.items():
		try:
			result[name] = await provider.check_availability()
		except Exception as exc:
			result[name] = {"available": False, "message": str(exc), "models": []}
	return result
