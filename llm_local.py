"""Optional local text teacher, isolated from the effectiveness experiment."""
import argparse
import json
import os
from pathlib import Path
from local_backend import LocalLLM, network_disabled, model_identity

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--messages', type=Path, required=True, help='JSON array of role/content messages')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=8192)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('Output exists; refusing to overwrite an earlier teacher response')
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    identity = model_identity(args.model_dir)
    if (identity['model_type'] != 'qwen3' or identity['text_shape'] !=
            {'hidden_size': 4096, 'num_hidden_layers': 36, 'intermediate_size': 12288}):
        raise SystemExit('Expected local Qwen3-8B directory')
    with network_disabled():
        answer = LocalLLM(args.model_dir, thinking=args.thinking).generate(
            json.loads(args.messages.read_text(encoding='utf-8')), args.max_new_tokens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'model_identity': identity, 'answer': answer}, ensure_ascii=False, indent=2), encoding='utf-8')
