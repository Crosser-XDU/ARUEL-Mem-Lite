import asyncio
from collections import Counter
import json
import sqlite3
import time
import math
from pathlib import Path
import httpx
from openai import AsyncOpenAI, APIConnectionError, APIStatusError
from . import prompts
from .normalize import parse_output,caption_text
from .util import dumps, ident, normvec, tokens


class UpstreamError(Exception):
    pass


class Provider:
    def __init__(self, settings):
        self.s=settings
        self.calls=Counter({'format_retry':0})
        self.sem=asyncio.Semaphore(settings.llm_concurrency)
        self.embedding_sem=asyncio.Semaphore(4)
        self.client=AsyncOpenAI(api_key=settings.api_key or "unconfigured",base_url=settings.api_base,
                               timeout=settings.upstream_timeout,max_retries=0)
        self.ec=AsyncOpenAI(api_key=settings.embedding_key or "unconfigured",base_url=settings.embedding_base or settings.api_base,
                           timeout=settings.upstream_timeout,max_retries=0)
        self.worker=httpx.AsyncClient(base_url=settings.worker_url,timeout=600,
                                      headers={"Authorization":"Bearer "+settings.worker_token},trust_env=False)
        self.cache=Path(settings.data_dir)/"provider-cache.sqlite3"
        with sqlite3.connect(self.cache) as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY,value TEXT NOT NULL)")

    def cached(self,key):
        with sqlite3.connect(self.cache,timeout=30) as c:
            row=c.execute("SELECT value FROM cache WHERE key=?",(key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self,key,value):
        with sqlite3.connect(self.cache,timeout=30) as c:
            c.execute("INSERT OR REPLACE INTO cache VALUES (?,?)",(key,dumps(value)))

    async def chat(self, system, content, purpose, max_tokens=6000, cache_scope=None):
        key=ident("gpt-4o-mini","prompt-v3",system,content,cache_scope) if cache_scope else None
        if key:
            old=self.cached(key)
            if old is not None:
                self.calls[purpose+"_cache_hit"]+=1
                return old
        async with self.sem:
            # Only transport/rate/service errors retry. Shape/JSON repair NEVER calls the model again.
            for attempt in range(2 if purpose!="judge" else 1):
                try:
                    self.calls[purpose]+=1
                    response=await self.client.chat.completions.create(model="gpt-4o-mini",temperature=0,
                        messages=[{"role":"system","content":system},{"role":"user","content":content}],
                        response_format={"type":"json_object"},max_tokens=max_tokens)
                    choice=response.choices[0]
                    value,mode=parse_output(choice.message.content)
                    if purpose in {'caption','query_image'} and mode=='unstructured':
                        value={'caption':str(choice.message.content or '').strip()}
                    self.calls["parse_"+mode]+=1
                    if choice.finish_reason=="length":
                        self.calls["output_length_limited"]+=1
                    if key:
                        self.save(key,value)
                    return value
                except (APIConnectionError,APIStatusError) as exc:
                    status=getattr(exc,"status_code",None)
                    self.calls['upstream_status_'+str(status or type(exc).__name__)]+=1
                    retry=status in {408,429,500,502,503,504} or isinstance(exc,APIConnectionError)
                    if not retry or attempt==1 or purpose=="judge":
                        self.calls["upstream_error"]+=1
                        raise UpstreamError(f"{purpose} upstream unavailable") from None
                    self.calls["transport_retry"]+=1
                    await asyncio.sleep(2)
                except Exception:
                    self.calls["upstream_error"]+=1
                    raise UpstreamError(f"{purpose} upstream response unavailable") from None

    async def embed(self,texts):
        out=[None]*len(texts)
        pending=[]
        for i,t in enumerate(texts):
            if not t.strip():t="empty source"
            key=ident("text-embedding-v4",self.s.dimensions,t)
            old=self.cached(key)
            if old is not None:
                out[i]=normvec(old)
            else:pending.append((i,t,key))
        async def batch(items):
            async with self.embedding_sem:
                for attempt in range(2):
                    try:
                        self.calls["embedding"]+=1
                        r=await self.ec.embeddings.create(model="text-embedding-v4",dimensions=self.s.dimensions,
                            input=[x[1] for x in items],encoding_format="float")
                        rows=sorted(r.data,key=lambda x:x.index)
                        if [x.index for x in rows]!=list(range(len(items))):raise ValueError("embedding index mismatch")
                        for item,row in zip(items,rows):
                            if len(row.embedding)!=self.s.dimensions:raise ValueError("embedding dimensions mismatch")
                            out[item[0]]=normvec(row.embedding)
                            self.save(item[2],row.embedding)
                        return
                    except (APIConnectionError,APIStatusError) as exc:
                        if attempt or (isinstance(exc,APIStatusError) and exc.status_code not in {408,429,500,502,503,504}):
                            raise UpstreamError("embedding upstream unavailable") from None
                        self.calls["embedding_transport_retry"]+=1
                        await asyncio.sleep(2)
                    except Exception:
                        raise UpstreamError("invalid embedding response") from None
        await asyncio.gather(*(batch(pending[i:i+10]) for i in range(0,len(pending),10)))
        return out

    async def caption(self,uri,scope):
        result=await self.chat(prompts.CAPTION,[{"type":"image_url","image_url":{"url":uri,"detail":"high"}}],
                               "caption",3500,cache_scope=scope)
        value=caption_text(result)
        if value:return value
        # The image and its original context remain searchable; no endless format retries.
        self.calls["caption_unavailable"]+=1
        return "[Caption unavailable; original image preserved for visual inspection.]"

    async def vision(self,texts=None,images=None):
        try:
            r=await self.worker.post("/siglip",json={"texts":texts or [],"images":images or []})
            r.raise_for_status()
            v=r.json()
            return [normvec(x) for x in v["text_vectors"]],[normvec(x) for x in v["image_vectors"]]
        except Exception:
            raise UpstreamError("SigLIP2 worker unavailable") from None

    async def rerank(self,query,docs):
        key=ident("qwen-rerank-v1",query,docs)
        old=self.cached(key)
        if old is not None:return old
        try:
            r=await self.worker.post("/rerank",json={"query":query,"documents":docs,
                                                    "instruction":prompts.RERANK_INSTRUCTION})
            r.raise_for_status()
            scores=r.json()["scores"]
            if len(scores)!=len(docs):raise ValueError("count")
            if not all(math.isfinite(float(v)) for v in scores):raise ValueError('nonfinite score')
            self.calls["rerank_pairs"]+=len(docs)
            self.save(key,scores)
            return scores
        except Exception:
            self.calls["rerank_degraded"]+=1
            return None

    async def vision_query(self,text):
        try:
            r=await self.worker.post('/siglip-text-chunks',json={'text':text,'max_chunks':8})
            r.raise_for_status()
            return [normvec(x['vector']) for x in r.json()['chunks']]
        except Exception:raise UpstreamError('SigLIP2 query encoding unavailable') from None

    async def close(self):
        await self.client.close()
        await self.ec.close()
        await self.worker.aclose()
