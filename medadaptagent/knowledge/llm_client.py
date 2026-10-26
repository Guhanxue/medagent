"""Explicit cloud or local OpenAI-compatible LLM, including image messages.

Credentials are read from the environment and never included in errors or caches.
The release CLI defaults to a loopback server; it never falls back to a paid API.
"""
import hashlib
import ipaddress
import json
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit


def _endpoint(url):
    try:
        parts = urlsplit(url)
        host = parts.hostname
        _ = parts.port
        local = host == 'localhost'
        try:
            local = local or ipaddress.ip_address(host).is_loopback
        except ValueError:
            pass
        if (not host or parts.scheme not in ('http', 'https') or parts.username
                or parts.password or parts.query or parts.fragment
                or any(c.isspace() for c in url)):
            raise ValueError
        if parts.scheme == 'http' and not local:
            raise ValueError
        return local
    except (ValueError, TypeError):
        raise ValueError('LLM endpoint must use HTTPS (or loopback HTTP), without URL credentials or query parameters') from None


class UnifiedLLMClient:
    def __init__(self, backend, model, cache_dir=None, **kwargs):
        import httpx
        from openai import OpenAI, AzureOpenAI
        self.backend = 'openai' if backend == 'local' else backend
        self.model = self._model_name = model
        self.local = backend == 'local'
        self._cache_hits = self._cache_misses = 0
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not model:
            raise ValueError('An explicit model name is required')
        if backend in ('local', 'openai'):
            base = kwargs.get('base_url') or os.environ.get('MEDADAPT_LLM_BASE_URL' if self.local else 'OPENAI_BASE_URL')
            base = base or ('http://127.0.0.1:11434/v1' if self.local else 'https://api.openai.com/v1')
            loopback = _endpoint(base)
            key = kwargs.get('api_key') or os.environ.get('MEDADAPT_LLM_API_KEY' if self.local else 'OPENAI_API_KEY', '')
            if not key and not loopback:
                raise ValueError('Configure the API key for the selected remote endpoint')
            self._endpoint_id = base
            self._client = OpenAI(base_url=base, api_key=key or 'local-no-key', max_retries=0,
                http_client=httpx.Client(trust_env=not loopback, follow_redirects=False, timeout=120))
        elif backend == 'azure':
            base = kwargs.get('endpoint') or os.environ.get('AZURE_OPENAI_ENDPOINT', '')
            _endpoint(base)
            key = kwargs.get('api_key') or os.environ.get('AZURE_OPENAI_API_KEY')
            if not key:
                raise ValueError('Set AZURE_OPENAI_API_KEY for the selected Azure endpoint')
            self._endpoint_id = base
            proxy = (kwargs.get('proxy') or os.environ.get('HTTPS_PROXY')
                     or os.environ.get('https_proxy') or os.environ.get('HTTP_PROXY')
                     or os.environ.get('http_proxy'))
            self._client = AzureOpenAI(azure_endpoint=base, api_key=key,
                api_version=kwargs.get('api_version') or os.environ.get('AZURE_API_VERSION', '2024-10-21'),
                max_retries=0, http_client=httpx.Client(proxy=proxy, follow_redirects=False, timeout=120))
        elif backend == 'anthropic':
            import anthropic
            key = kwargs.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
            if not key:
                raise ValueError('Set ANTHROPIC_API_KEY for the selected backend')
            self._endpoint_id = 'anthropic'
            self._client = anthropic.Anthropic(api_key=key, max_retries=0,
                http_client=httpx.Client(follow_redirects=False, timeout=120))
        else:
            raise ValueError('Unsupported LLM backend')

    def cached_complete(self, messages, max_tokens=2000, temperature=0, seed=42, system=None):
        if self.backend not in ('azure', 'openai'):
            raise NotImplementedError('Image completion requires a local/OpenAI-compatible or Azure backend')
        full = ([{'role':'system', 'content':system}] if system else []) + list(messages or [])
        args = dict(model=self.model, messages=full, temperature=temperature)
        args['max_tokens' if self.local else 'max_completion_tokens'] = max_tokens
        if seed is not None:
            args['seed'] = seed
        cache_file = None
        if self.cache_dir:
            key = hashlib.sha256(json.dumps([self._endpoint_id, args], sort_keys=True).encode()).hexdigest()
            cache_file = self.cache_dir / (key + '.json')
            if cache_file.exists():
                from openai.types.chat import ChatCompletion
                response = ChatCompletion.model_validate_json(cache_file.read_text())
                self._cache_hits += 1
                return response
        self._cache_misses += 1
        try:
            response = self._client.chat.completions.create(**args)
        except Exception:
            raise RuntimeError('LLM request failed; check endpoint, model, authentication and supported parameters. Server details omitted.') from None
        if (not response.choices or response.choices[0].finish_reason != 'stop'
                or not response.choices[0].message.content):
            raise RuntimeError('LLM returned an empty or incomplete response; no output accepted')
        if cache_file:
            cache_file.write_text(response.model_dump_json())
        return response

    def chat(self, prompt, max_tokens=2000, system=None):
        if self.backend != 'anthropic':
            return self.cached_complete([{'role':'user','content':prompt}], max_tokens=max_tokens, system=system).choices[0].message.content
        args = dict(model=self.model, max_tokens=max_tokens, messages=[{'role':'user','content':prompt}])
        if system:
            args['system'] = system
        try:
            response = self._client.messages.create(**args)
        except Exception:
            raise RuntimeError('LLM request failed; server details omitted') from None
        if response.stop_reason != 'end_turn':
            raise RuntimeError('LLM response did not complete')
        return response.content[0].text

    @property
    def messages(self):
        return self

    def create(self, model=None, max_tokens=2000, messages=None, **kwargs):
        if self.backend in ('azure', 'openai'):
            text = self.cached_complete(messages, max_tokens=max_tokens, system=kwargs.get('system')).choices[0].message.content
        else:
            try:
                return self._client.messages.create(model=self.model, max_tokens=max_tokens, messages=messages, **kwargs)
            except Exception:
                raise RuntimeError('LLM request failed; server details omitted') from None
        return SimpleNamespace(content=[SimpleNamespace(text=text)])

    def cache_stats(self):
        total = self._cache_hits + self._cache_misses
        return dict(hits=self._cache_hits, misses=self._cache_misses, total=total, hit_rate=self._cache_hits/max(total, 1))


def create_llm_client(backend=None, model=None, **kwargs):
    # Existing benchmark runners can still select a backend explicitly.
    backend = backend or os.environ.get('MEDADAPT_LLM_BACKEND', 'local')
    model = model or os.environ.get('MEDADAPT_LLM_MODEL')
    if not model and backend == 'local':
        model = 'qwen3-vl:4b-instruct'
    if not model:
        raise ValueError('Set MEDADAPT_LLM_MODEL or pass model explicitly')
    return UnifiedLLMClient(backend, model, **kwargs)
