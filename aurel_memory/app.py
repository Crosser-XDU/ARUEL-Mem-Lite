import asyncio
from collections import Counter
from contextlib import asynccontextmanager
import hmac
import logging
import time
import weakref
from fastapi import FastAPI,Request,HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from . import __version__
from .config import Settings
from .schemas import AddRequest,SearchRequest
from .storage import Store,Conflict
from .provider import Provider,UpstreamError
from .ingestion import Ingestor
from .retrieval import Retriever

log=logging.getLogger("aurel")


class BodyLimit:
    def __init__(self,app,limit):self.app=app;self.limit=limit
    async def __call__(self,scope,receive,send):
        if scope["type"]!="http":return await self.app(scope,receive,send)
        size=0;messages=[]
        while True:
            message=await receive()
            if message["type"]=="http.disconnect":return
            size+=len(message.get("body",b""))
            if size>self.limit:
                return await JSONResponse({"detail":"request exceeds configured body limit"},status_code=413)(scope,receive,send)
            messages.append(message)
            if not message.get("more_body"):break
        async def replay():
            if messages:return messages.pop(0)
            return await receive()
        await self.app(scope,replay,send)


def create_app(settings=None,provider=None):
    s=settings or Settings.from_env()
    store=Store(s.data_dir);p=provider or Provider(s)
    ingestor=Ingestor(s,store,p);retriever=Retriever(s,store,p)
    locks=weakref.WeakValueDictionary();search_sem=asyncio.Semaphore(s.search_concurrency)
    add_sem=asyncio.Semaphore(2)
    metrics=Counter();started=time.time()
    @asynccontextmanager
    async def lifespan(app):
        from .util import encoding
        encoding()  # Preloaded local tokenizer cache; fail at startup rather than first Add.
        yield
        if hasattr(p,"close"):await p.close()
    app=FastAPI(title="AUREL-Mem-"+s.profile,version=__version__,lifespan=lifespan,docs_url=None,redoc_url=None)
    app.add_middleware(BodyLimit,limit=s.max_body_bytes)
    app.state.settings=s;app.state.store=store;app.state.provider=p;app.state.retriever=retriever

    def auth(request):
        header=request.headers.get("authorization","")
        parts=header.split(None,1)
        token=parts[1] if len(parts)==2 and parts[0].lower() in {"bearer","token"} else request.headers.get("x-api-key","")
        if not s.service_token or not hmac.compare_digest(token,s.service_token):
            raise HTTPException(401,"invalid or missing service token")

    @app.exception_handler(RequestValidationError)
    async def validation(request,exc):
        return JSONResponse({"detail":[{"loc":e["loc"],"type":e["type"],"msg":e["msg"]} for e in exc.errors()]},422)

    @app.exception_handler(Conflict)
    async def conflict(request,exc):return JSONResponse({"detail":"request_id already used with a different payload"},409)

    @app.exception_handler(UpstreamError)
    async def upstream(request,exc):
        metrics["upstream_errors"]+=1
        return JSONResponse({"detail":str(exc)},503,headers={"Retry-After":"5"})

    @app.exception_handler(Exception)
    async def unexpected(request,exc):
        metrics["internal_errors"]+=1
        # No payloads, query strings or upstream secrets in error logs.
        log.error("request failed error_type=%s",type(exc).__name__,exc_info=False)
        return JSONResponse({"detail":"temporary service failure"},503,headers={"Retry-After":"5"})

    @app.get("/healthz")
    async def health():
        status={"status":"ok","profile":s.profile,"version":__version__,"uptime_seconds":int(time.time()-started)}
        if s.profile in {"A","C"} and hasattr(p,"worker"):
            try:
                r=await p.worker.get("/healthz",timeout=3)
                data=r.json();name="siglip" if s.profile=="A" else "reranker"
                if not data.get(name):return JSONResponse({**status,"status":"model_not_ready"},503)
            except Exception:return JSONResponse({**status,"status":"model_not_ready"},503)
        return status

    @app.get("/stats")
    async def stats(request:Request):
        auth(request)
        return {"profile":s.profile,"storage":store.stats(),"provider":dict(p.calls),"retrieval":dict(retriever.metrics),"http":dict(metrics)}

    @app.post("/add")
    async def add(body:AddRequest,request:Request):
        auth(request)
        lock=locks.get(body.user_id)
        if lock is None:lock=asyncio.Lock();locks[body.user_id]=lock
        try:
            async with asyncio.timeout(s.add_timeout):
                async with lock,add_sem:
                    answer=await ingestor.add(body)
            metrics["add_ok"]+=1
            return answer
        except TimeoutError:
            metrics["add_timeout"]+=1
            raise HTTPException(503,"Add deadline exceeded; same request_id is safe to retry",headers={"Retry-After":"5"}) from None

    @app.post("/search")
    async def search(body:SearchRequest,request:Request):
        auth(request)
        try:
            async with asyncio.timeout(s.search_timeout):
                async with search_sem:
                    result=await retriever.search(body)
            metrics["search_ok"]+=1
            return result
        except TimeoutError:
            metrics["search_timeout"]+=1
            raise HTTPException(503,"Search deadline exceeded",headers={"Retry-After":"5"}) from None
    return app
