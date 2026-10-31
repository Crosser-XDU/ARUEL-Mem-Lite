"""Portable branch entry point; keeps credentials in ENV_FILE/.env."""
import argparse,os,sys
from pathlib import Path
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
load_dotenv(os.environ.get('ENV_FILE',str(ROOT/'.env')))
PROFILE='B'
p=argparse.ArgumentParser()
p.add_argument('component',nargs='?',choices=['api','worker'],default='api')
p.add_argument('--host',default='127.0.0.1')
p.add_argument('--port',type=int)
p.add_argument('--check-config',action='store_true')
a=p.parse_args()
os.environ.setdefault('MEMORY_PROFILE',PROFILE)
if os.environ['MEMORY_PROFILE'].upper()!=PROFILE:p.error('This branch is configured for profile '+PROFILE)
def require(name):
 value=os.environ.get(name,'')
 if not value or value.startswith('replace-with-'):p.error('Set '+name+' in your private .env')
if a.component=='worker':
 if PROFILE!='C':p.error('Profile B does not use a model worker')
 if a.host not in ('127.0.0.1','::1'):p.error('The model worker must listen on loopback')
 require('WORKER_TOKEN');require('MODEL_ROOT')
 if not (Path(os.environ['MODEL_ROOT'])/'qwen-reranker/config.json').is_file():p.error('Download the pinned reranker first')
 os.environ['ENABLE_SIGLIP']='0'
 os.environ.setdefault('HF_HUB_OFFLINE','1')
 app='aurel_memory.model_worker:create_worker';port=a.port or 18220
else:
 require('MEMORY_API_TOKEN');require('OPENAI_API_KEY')
 if PROFILE=='C':require('WORKER_TOKEN')
 os.environ.setdefault('SEARCH_CONCURRENCY','1' if PROFILE=='C' else '2')
 app='aurel_memory.app:create_app';port=a.port or 18110
if a.check_config:
 print('Configuration valid:',PROFILE,a.component,a.host,port)
else:
 import uvicorn
 uvicorn.run(app,factory=True,host=a.host,port=port,access_log=False,log_level='warning')
