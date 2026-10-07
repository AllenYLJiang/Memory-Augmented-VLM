"""Offline Transformers providers. No remote API fallback or semantic retries."""
from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def model_identity(directory):
    directory = Path(directory).resolve()
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    index = directory / 'model.safetensors.index.json'
    weights = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
    paths = [directory / name for name in weights]
    paths += sorted(p for p in directory.iterdir() if p.suffix in ('.json', '.jinja', '.txt', '.model') and p.is_file())
    if any(not p.is_file() for p in paths):
        raise ValueError('Incomplete local weights: ' + str(directory))
    text = config.get('text_config', config)
    return {'directory': str(directory), 'model_type': config['model_type'],
            'text_shape': {k: text.get(k) for k in ('hidden_size', 'num_hidden_layers', 'intermediate_size')},
            'files_sha256': {p.name: sha(p) for p in paths}}


@contextmanager
def network_disabled():
    """Fail closed, including dormant vendor API functions accidentally reached."""
    old_connect, old_ex, old_create = socket.socket.connect, socket.socket.connect_ex, socket.create_connection
    def deny(*args, **kwargs):
        raise RuntimeError('LOCAL_ONLY: network inference is disabled')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = old_connect, old_ex, old_create


def vlm_messages(paths, prompt, max_pixels):
    content = []
    for i, path in enumerate(paths):
        content.extend([{'type': 'text', 'text': f'T{i}'},
                        {'type': 'image', 'image': str(Path(path).resolve()), 'max_pixels': max_pixels}])
    content.append({'type': 'text', 'text': prompt})
    return [{'role': 'user', 'content': content}]


class LocalVLM:
    def __init__(self, directory, device='cuda:0', dtype='bfloat16', attention='sdpa'):
        self.directory, self.device, self.dtype, self.attention = str(directory), device, dtype, attention
        self.model = self.processor = None
        self.lock = threading.RLock()

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available() or not self.device.startswith('cuda:'):
            raise RuntimeError('CUDA required; refusing silent CPU/offload fallback')
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.directory, local_files_only=True, trust_remote_code=False,
            dtype=getattr(torch, self.dtype), device_map={'': self.device},
            attn_implementation=self.attention).eval()
        self.processor = AutoProcessor.from_pretrained(self.directory, local_files_only=True, trust_remote_code=False)
        print(f'[local-vlm] loaded {self.directory}; {self.device}; {self.dtype}; {self.attention}', flush=True)

    def __call__(self, config, media, prompt):
        import torch
        with self.lock, torch.inference_mode():
            self.load()
            if len(media['image_paths']) != 8:
                raise ValueError('Exactly eight chronological frames required')
            messages = vlm_messages(media['image_paths'], prompt, config['image_max_pixels'])
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            started = time.perf_counter()
            inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors='pt', images_kwargs={'max_pixels': config['image_max_pixels']})
            inputs = inputs.to(self.model.device)
            n = inputs.input_ids.shape[-1]
            # Same nominal zero-temperature protocol as the API arm; no extra system prompt.
            generated = self.model.generate(**inputs, max_new_tokens=config['max_output_tokens'], do_sample=False,
                temperature=None, top_p=None, top_k=None, use_cache=True)
            ids = generated[0, n:]
            raw = self.processor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            torch.cuda.synchronize(self.device)
            usage = {'input_tokens': n, 'output_tokens': len(ids), 'total_tokens': n + len(ids),
                'local_generation_seconds': time.perf_counter() - started,
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device),
                'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(self.device),
                'output_cap_reached': len(ids) == config['max_output_tokens'], 'remote_API_calls': 0}
            del inputs, generated, ids
            return raw, usage


class LocalLLM:
    """Optional text-only teacher; not invoked by V919 Stage 2/3."""
    def __init__(self, directory, device='cuda:0', thinking=False):
        self.directory, self.device, self.thinking = str(directory), device, thinking
        self.model = self.tokenizer = None

    def generate(self, messages, max_new_tokens=8192):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.directory, local_files_only=True,
                trust_remote_code=False, dtype=torch.bfloat16, device_map={'': self.device}, attn_implementation='sdpa').eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.directory, local_files_only=True, trust_remote_code=False)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=self.thinking)
        inputs = self.tokenizer([text], return_tensors='pt').to(self.model.device)
        params = dict(do_sample=True, temperature=.6, top_p=.95, top_k=20) if self.thinking else dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **params)
        raw = self.tokenizer.decode(output[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        reasoning, final = raw.rsplit('</think>', 1) if '</think>' in raw else ('', raw)
        return {'raw': raw, 'content': final.strip(), 'thinking_content': reasoning,
                'enable_thinking': self.thinking, 'remote_API_calls': 0}
