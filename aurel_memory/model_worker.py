"""Local-only Apple Silicon worker. SigLIP2 encodes; Qwen uses causal yes/no logits."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hmac
import json
import os
from pathlib import Path
import threading
import time
import numpy as np
from fastapi import FastAPI,HTTPException,Request
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from .media import decode
from io import BytesIO
from PIL import Image


class SigRequest(BaseModel):
    texts:list[str]=Field(default_factory=list,max_length=256)
    images:list[str]=Field(default_factory=list,max_length=128)


class RankRequest(BaseModel):
    query:str
    documents:list[str]=Field(max_length=100)
    instruction:str="Assess whether the passage provides evidence for the query."


class TextRequest(BaseModel):
    text:str
    max_chunks:int|None=Field(default=None,ge=1,le=128)


class Models:
    def __init__(self,root):
        self.root=Path(root);self.siglip=None;self.qwen=None;self.processor=None;self.tokenizer=None
        self.lock=threading.Lock();self.metrics={"image_vectors":0,"text_vectors":0,"rerank_pairs":0}

    def load(self):
        # Current deployment is B/C only. Never load/download the cancelled SigLIP2 variant.
        if os.environ.get('ENABLE_SIGLIP','0')=='1':
            import torch
            from transformers import AutoModel,AutoProcessor
            torch.set_num_threads(2)
            self.processor=AutoProcessor.from_pretrained(self.root/"siglip2",local_files_only=True)
            self.siglip=AutoModel.from_pretrained(self.root/"siglip2",local_files_only=True).eval().to("cpu")
        from mlx_lm import load
        self.qwen,self.tokenizer=load(str(self.root/"qwen-reranker"))
        self.yes=self.tokenizer.convert_tokens_to_ids("yes");self.no=self.tokenizer.convert_tokens_to_ids("no")
        if self.yes==self.no or self.yes is None or self.no is None:raise RuntimeError("invalid reranker token IDs")

    def sig(self,texts,images):
        import torch
        with self.lock,torch.inference_mode():
            tv=[];iv=[]
            for start in range(0,len(texts),16):
                batch=texts[start:start+16]
                inp=self.processor(text=batch,padding="max_length",truncation=True,max_length=64,return_tensors="pt")
                vec=self.siglip.get_text_features(**inp)
                if not isinstance(vec,torch.Tensor):vec=vec.pooler_output
                vec=vec/vec.norm(dim=-1,keepdim=True)
                tv+=vec.cpu().float().tolist()
            for start in range(0,len(images),4):
                ims=[Image.open(BytesIO(decode(x)[1])).convert("RGB") for x in images[start:start+4]]
                inp=self.processor(images=ims,return_tensors="pt")
                vec=self.siglip.get_image_features(**inp)
                if not isinstance(vec,torch.Tensor):vec=vec.pooler_output
                vec=vec/vec.norm(dim=-1,keepdim=True)
                iv+=vec.cpu().float().tolist()
            self.metrics["text_vectors"]+=len(tv);self.metrics["image_vectors"]+=len(iv)
            return {"text_vectors":tv,"image_vectors":iv}

    def text_chunks(self,text,max_chunks=None):
        # Use actual SigLIP token IDs, reserving special tokens; preserve exact Unicode substrings.
        tok=self.processor.tokenizer
        spans=[];start=0
        while start<len(text):
            lo,hi=start+1,len(text)
            while lo<hi:
                mid=(lo+hi+1)//2
                if len(tok.encode(text[start:mid],add_special_tokens=True))<=64:lo=mid
                else:hi=mid-1
            spans.append((text[start:lo],start,lo));start=lo
        if max_chunks and len(spans)>max_chunks:
            spans=[spans[i] for i in np.linspace(0,len(spans)-1,max_chunks,dtype=int)]
        vec=self.sig([x[0] for x in spans],[])["text_vectors"]
        return {"chunks":[{"text":x[0],"start":x[1],"end":x[2],"vector":v} for x,v in zip(spans,vec)]}

    def rank_ids(self,query,doc,instruction):
        prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        pre=self.tokenizer.encode(prefix,add_special_tokens=False)
        post=self.tokenizer.encode(suffix,add_special_tokens=False)
        qids=self.tokenizer.encode(query,add_special_tokens=False)[:256]
        qtext=self.tokenizer.decode(qids)
        head=f"<Instruct>: {instruction}\n<Query>: {qtext}\n<Document>: "
        headids=self.tokenizer.encode(head,add_special_tokens=False)
        d=self.tokenizer.encode(doc,add_special_tokens=False)
        ids=pre+headids+d[:max(1,1024-len(pre)-len(post)-len(headids))]+post
        return ids

    def rank(self,query,docs,instruction):
        import mlx.core as mx
        scores=[];started=time.monotonic()
        with self.lock:
            for doc in docs:
                ids=self.rank_ids(query,doc,instruction)
                inp=mx.array([ids])
                # Project only the last hidden state, numerically the same causal next-token position.
                hidden=self.qwen.model(inp)[:,-1:,:]
                logits=self.qwen.model.embed_tokens.as_linear(hidden) if self.qwen.args.tie_word_embeddings else self.qwen.lm_head(hidden)
                margin=(logits[0,0,self.yes]-logits[0,0,self.no]).astype(mx.float32)
                mx.eval(margin)
                value=float(margin.item())
                if not np.isfinite(value):raise RuntimeError("nonfinite reranker score")
                scores.append(value)
                del hidden,logits,margin,inp
            self.metrics["rerank_pairs"]+=len(scores)
            self.metrics["last_rerank_seconds"]=round(time.monotonic()-started,3)
            self.metrics["last_rerank_pairs"]=len(scores)
            self.metrics["mlx_peak_bytes"]=mx.get_peak_memory()
            mx.clear_cache()
        return {"scores":scores,"score_type":"yes_minus_no_logit","model":"Qwen3-Reranker-4B-mxfp8"}


    def self_test(self):
        import mlx.core as mx
        query='What is the capital of France?'
        docs=['The capital of France is Paris.','Berlin is the capital of Germany.','Apples are a kind of fruit.']
        instruction='Given a web search query, retrieve relevant passages that answer the query'
        scores=self.rank(query,docs,instruction)['scores']
        with self.lock:
            inp=mx.array([self.rank_ids(query,docs[0],instruction)])
            logits=self.qwen(inp)
            reference=(logits[0,-1,self.yes]-logits[0,-1,self.no]).astype(mx.float32)
            mx.eval(reference);ref=float(reference.item())
            error=abs(ref-scores[0]);del inp,logits,reference;mx.clear_cache()
        return {'scores':scores,'expected_order':scores[0]>max(scores[1:]),'reference_margin':ref,'projection_abs_error':error,'projection_equivalent':error<=0.125,'quantization':json.loads((self.root/'qwen-reranker/config.json').read_text())['quantization']}


def create_worker():
    load_dotenv(os.environ.get("ENV_FILE",".env"))
    token=os.environ.get("WORKER_TOKEN","")
    models=Models(os.environ["MODEL_ROOT"])
    executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="models")
    queue=asyncio.Semaphore(4)
    async def run(fn,*args):
        async with queue:
            return await asyncio.get_running_loop().run_in_executor(executor,fn,*args)
    @asynccontextmanager
    async def lifespan(app):
        await run(models.load)
        yield
        executor.shutdown(wait=True)
    app=FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None)
    def auth(request):
        if not token or not hmac.compare_digest(request.headers.get("authorization",""),"Bearer "+token):raise HTTPException(401)
    @app.get("/healthz")
    async def health():return {"siglip":models.siglip is not None,"reranker":models.qwen is not None}
    @app.post("/siglip")
    async def sig(body:SigRequest,request:Request):
        auth(request);return await run(models.sig,body.texts,body.images)
    @app.post("/siglip-text-chunks")
    async def chunk(body:TextRequest,request:Request):
        auth(request);return await run(models.text_chunks,body.text,body.max_chunks)
    @app.post("/rerank")
    async def rank(body:RankRequest,request:Request):
        auth(request);return await run(models.rank,body.query,body.documents,body.instruction)
    @app.post("/self-test")
    async def test(request:Request):auth(request);return await run(models.self_test)
    @app.get("/stats")
    async def stats(request:Request):auth(request);return models.metrics
    return app
