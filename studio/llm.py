"""Prompt-writing LLMs: local (Ollama / LM Studio) or cloud APIs."""
import json
import re
from typing import Iterator, Optional

import requests

from .sysinfo import port_listening

OLLAMA_URL = "http://127.0.0.1:11434"

# provider -> (kind, default base url, suggested models)
PROVIDERS = {
    "Ollama (local)": ("ollama", OLLAMA_URL, ["qwen3:4b-instruct", "qwen3:4b", "llama3.2:3b", "gemma3:4b"]),
    "LM Studio (local)": ("openai", "http://localhost:1234/v1", ["local-model"]),
    "OpenAI": ("openai", "https://api.openai.com/v1", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"]),
    "Google Gemini": ("openai", "https://generativelanguage.googleapis.com/v1beta/openai",
                      ["gemini-2.5-flash", "gemini-2.5-pro"]),
    "Groq": ("openai", "https://api.groq.com/openai/v1", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]),
    "OpenRouter": ("openai", "https://openrouter.ai/api/v1", ["openai/gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct"]),
    "DeepSeek": ("openai", "https://api.deepseek.com/v1", ["deepseek-chat"]),
    "Anthropic Claude": ("anthropic", "https://api.anthropic.com/v1", ["claude-sonnet-4-5", "claude-haiku-4-5"]),
    "Custom (OpenAI-compatible)": ("openai", "", [""]),
}
LOCAL_PROVIDERS = {"Ollama (local)", "LM Studio (local)"}

SYSTEM_PROMPTS = {
    "image": (
        "You are an expert prompt engineer for text-to-image diffusion models. Rewrite the user's idea into a "
        "single vivid, concrete prompt: subject, action, setting, composition, camera/lens, lighting, color "
        "palette, style and quality descriptors. Keep it under 90 words. Output ONLY the prompt text."
    ),
    "edit": (
        "You are an expert prompt engineer for reference-based image editing models (FLUX.2 Klein, FLUX Kontext, "
        "Qwen Image Edit, Nano Banana). The user supplies reference image(s) the model can see, plus an idea. "
        "Rewrite the idea into one clear instruction describing the NEW image: subject, pose, setting, framing, "
        "lighting and style. Refer to the supplied item as 'the <item> from the reference image' and tell the model "
        "to keep its exact design, colors, pattern and details. Do not invent a different description of the item. "
        "Keep it under 80 words. Output ONLY the prompt text."
    ),
    "video": (
        "You are an expert prompt engineer for text-to-video models (Wan, Kling, Veo, Sora). Rewrite the user's "
        "idea into one cinematic prompt describing the subject, the motion over time, camera movement, setting, "
        "lighting and mood, as one continuous shot of about 5 seconds. Keep it under 100 words. Output ONLY the prompt."
    ),
    "audio": (
        "You are an expert prompt writer for AI music and sound models (Suno, MiniMax, ElevenLabs). Rewrite the "
        "user's idea into a concise description of genre, mood, tempo/BPM, instruments, vocals and structure. "
        "Output ONLY the prompt."
    ),
    "negative": (
        "Write a short comma-separated negative prompt (things to avoid) for a diffusion model given the user's "
        "prompt. Output ONLY the comma-separated list."
    ),
    "chat": "You are a helpful creative assistant that helps users write prompts for image, video and audio generation.",
}

THINK_RE = re.compile(r"<think>.*?</think>", re.S)
WORD_COUNT_RE = re.compile(r"\s*\(\s*\d+\s+words?\s*\)\s*$", re.I)


def clean(text: str) -> str:
    text = THINK_RE.sub("", text or "")
    if "</think>" in text:  # thinking models sometimes omit the opening tag
        text = text.rsplit("</think>", 1)[1]
    text = WORD_COUNT_RE.sub("", text.strip())
    return text.strip().strip('"').strip()


def ollama_unload_all(base: str = OLLAMA_URL) -> list[str]:
    """Evict loaded Ollama models from VRAM so local image/video jobs get the whole GPU."""
    if base == OLLAMA_URL and not port_listening(11434):
        return []
    try:
        loaded = [m["name"] for m in requests.get(f"{base}/api/ps", timeout=3).json().get("models", [])]
        for name in loaded:
            requests.post(f"{base}/api/generate", json={"model": name, "keep_alive": 0}, timeout=30)
        return loaded
    except (requests.RequestException, ValueError):
        return []


def ollama_running(base: str = OLLAMA_URL) -> bool:
    if base == OLLAMA_URL and not port_listening(11434):
        return False
    try:
        return requests.get(f"{base}/api/tags", timeout=2).ok
    except requests.RequestException:
        return False


def ollama_models(base: str = OLLAMA_URL) -> list[str]:
    if base == OLLAMA_URL and not port_listening(11434):
        return []
    try:
        r = requests.get(f"{base}/api/tags", timeout=3)
        return [m["name"] for m in r.json().get("models", [])]
    except (requests.RequestException, ValueError):
        return []


def ollama_pull(model: str, base: str = OLLAMA_URL) -> Iterator[dict]:
    with requests.post(f"{base}/api/pull", json={"name": model, "stream": True}, stream=True, timeout=None) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                yield json.loads(line)


def chat(provider: str, model: str, messages: list[dict], *, api_key: str = "", base_url: str = "",
         temperature: float = 0.7, timeout: int = 180) -> str:
    kind, default_base, _ = PROVIDERS[provider]
    base = (base_url or default_base).rstrip("/")
    if not base:
        raise ValueError("Base URL is required for a custom provider.")

    if kind == "ollama":
        r = requests.post(f"{base}/api/chat", timeout=timeout, json={
            "model": model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": temperature},
        })
        if not r.ok:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:300]}")
        return clean(r.json()["message"]["content"])

    if kind == "anthropic":
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        r = requests.post(f"{base}/messages", timeout=timeout, headers={
            "x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json",
        }, json={"model": model, "max_tokens": 1024, "system": system, "messages": rest, "temperature": temperature})
        if not r.ok:
            raise RuntimeError(f"Anthropic error {r.status_code}: {r.text[:300]}")
        return clean("".join(b.get("text", "") for b in r.json().get("content", [])))

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.post(f"{base}/chat/completions", timeout=timeout, headers=headers,
                      json={"model": model, "messages": messages, "temperature": temperature})
    if not r.ok:
        raise RuntimeError(f"{provider} error {r.status_code}: {r.text[:300]}")
    return clean(r.json()["choices"][0]["message"]["content"])


def enhance(provider: str, model: str, prompt: str, target: str = "image", *, api_key: str = "",
            base_url: str = "", target_model: Optional[str] = None) -> str:
    user = prompt if not target_model else f"Target model: {target_model}\nIdea: {prompt}"
    return chat(provider, model, [
        {"role": "system", "content": SYSTEM_PROMPTS[target]},
        {"role": "user", "content": user},
    ], api_key=api_key, base_url=base_url)
