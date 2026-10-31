"""Explicit download for C only, fixed revision and verified weight hash."""
import argparse,hashlib,json,os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);load_dotenv(os.environ.get('ENV_FILE',str(ROOT/'.env')))
p=argparse.ArgumentParser();p.add_argument('--model-root',default=os.environ.get('MODEL_ROOT','models'));a=p.parse_args()
manifest=json.loads((ROOT/'model-manifest.json').read_text());source=manifest['active_model']
target=Path(a.model_root)/'qwen-reranker'
snapshot_download(source['repo'],revision=source['revision'],local_dir=target,allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
weights=list(target.glob('*.safetensors'))
assert len(weights)==1,'Expected the pinned single-file model'
with weights[0].open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
assert actual==manifest['weight_sha256'],'Weight checksum mismatch'
print('Verified pinned Qwen3 reranker:',target)
