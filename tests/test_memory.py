import base64
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import numpy as np
import pytest
import httpx
from PIL import Image
from aurel_memory.app import create_app
from aurel_memory.config import Settings
from aurel_memory.normalize import parse_output,normalize,decision
from aurel_memory.media import split_captions,decode
from aurel_memory.util import terms,normvec,ident


def picture(color="red"):
    im=Image.new("RGB",(32,32),color)
    f=BytesIO();im.save(f,format="PNG")
    return "data:image/png;base64,"+base64.b64encode(f.getvalue()).decode()


class Provider:
    def __init__(self):self.calls=Counter();self.judgments=[]
    async def embed(self,texts):
        self.calls["embedding"]+=1
        out=[]
        for t in texts:
            v=np.zeros(1024,np.float32)
            for word in terms(t) or ["empty"]:v[int(hashlib.sha256(word.encode()).hexdigest()[:8],16)%1024]+=1
            out.append(normvec(v))
        return out
    async def caption(self,uri,scope):
        self.calls["caption"]+=1
        return "A red backpack with visible code ZX-482."
    async def chat(self,system,content,purpose,max_tokens=6000,cache_scope=None):
        self.calls[purpose]+=1
        if purpose=="judge":
            payload=json.loads(content[0]['text'] if isinstance(content,list) else content)
            if self.judgments:return self.judgments.pop(0)
            return {"decision":"sufficient","support_ids":[x["id"] for x in payload["evidence"][:1]]}
        if purpose=="query_image":return {"caption":"red backpack ZX-482"}
        payload=json.loads(content)
        cells=[]
        for m in payload["new_messages"]:
            cells.append({"sources":[m["id"]],"title":"backpack events","summary":m["text"],
                "entities":["小雨"],"scope":"topic","facts":[{"text":m["text"],"sources":[m["id"]],
                "subject":"小雨","predicate":"backpack","object":"blue" if "蓝" in m["text"] else "red",
                "kind":"state","assertion":"actual","action":"replace"}]})
        images=[]
        for m in payload["earlier_context"]:
            for im in m["images"]:
                images.append({"id":im["id"],"context":"This backpack belongs to 小雨.",
                               "sources":[m["id"],payload["new_messages"][0]["id"]],"entities":["小雨"]})
        return {"cells":cells,"images":images}
    async def rerank(self,query,docs):
        self.calls["rerank_pairs"]+=len(docs)
        return [float("ZX-482" in d)+float("小雨" in d) for d in docs]


@pytest.fixture
def stack(tmp_path):
    p=Provider();s=Settings(data_dir=str(tmp_path/"b"),service_token="test-token",profile="B")
    app=create_app(s,p)
    return app,p


def add_body(rid="r1",text="小雨的红色背包编号 ZX-482。",user="u",session="s"):
    return {"request_id":rid,"user_id":user,"session_id":session,"messages":[{"role":"user","timestamp":1790812800000,"content":text}]}


def auth():return {"Authorization":"Bearer test-token"}


@pytest.mark.parametrize("header",[{"Authorization":"Bearer test-token"},{"Authorization":"Token test-token"},{"X-Api-Key":"test-token"}])
async def test_contract_idempotency_isolation(stack,header):
    app,p=stack
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as c:
        assert (await c.get("/healthz")).status_code==200
        body=add_body()
        assert (await c.post("/add",json=body)).status_code==401
        r=await c.post("/add",json=body,headers=header)
        assert r.status_code==200,r.text
        assert r.json()=={"success":True,"request_id":"r1","user_id":"u","session_id":"s"}
        before=p.calls.copy()
        assert (await c.post("/add",json=body,headers=header)).json()==r.json()
        assert p.calls==before
        body["messages"][0]["content"]="different"
        assert (await c.post("/add",json=body,headers=header)).status_code==409
        sr=await c.post("/search",json={"user_id":"u","query":"小雨的背包编号是什么？","top_k":1},headers=header)
        assert sr.status_code==200,sr.text
        assert len(sr.json()["data"])==1
        assert "ZX-482" in json.dumps(sr.json(),ensure_ascii=False)
        assert (await c.post("/search",json={"user_id":"other","query":"小雨","top_k":100},headers=header)).json()=={"data":[]}


async def test_images_caption_once_cross_add_context(stack):
    app,p=stack
    uri=picture()
    body=add_body(text=[{"type":"text","text":"Image Caption: A red backpack with code ZX-482."},
                       {"type":"image_url","image_url":{"url":uri}}])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test",headers=auth()) as c:
        assert (await c.post("/add",json=body)).status_code==200
        assert p.calls["caption"]==0
        assert (await c.post("/add",json=add_body("r2","这是妹妹小雨的背包。"))).status_code==200
        r=await c.post("/search",json={"user_id":"u","query":"小雨的红色背包 ZX-482","top_k":100})
        assert r.status_code==200,r.text
        data=r.json()["data"]
        images=[part for x in data if isinstance(x["content"],list) for part in x["content"] if part["type"]=="image_url"]
        assert len(images)==1
        text="\n".join(x["content"] if isinstance(x["content"],str) else "\n".join(p["text"] for p in x["content"] if p["type"]=="text") for x in data)
        assert "妹妹小雨" in text
        # Caption copied into model summary is removed by output source dedup in production renderer.
        assert text.count("Image Caption:") == 1
        assert p.calls["judge"]<=2


async def test_missing_caption_and_qwen_branch(tmp_path):
    p=Provider();app=create_app(Settings(data_dir=str(tmp_path/"c"),profile="C",service_token="test-token"),p)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test",headers=auth()) as c:
        r=await c.post("/add",json=add_body(text=[{"type":"image_url","image_url":{"url":picture()}}]))
        assert r.status_code==200,r.text
        assert p.calls["caption"]==1
        r=await c.post("/search",json={"user_id":"u","query":"red backpack ZX-482","top_k":100})
        assert r.status_code==200,r.text
        assert p.calls["rerank_pairs"]>0


async def test_two_judges_max(stack):
    app,p=stack
    p.judgments=[{"decision":"insufficient","query":"backpack blue"},{"decision":"unknown","query":"travel red backpack"}]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test",headers=auth()) as c:
        await c.post("/add",json=add_body())
        r=await c.post("/search",json={"user_id":"u","query":"小雨背包","top_k":100})
        assert r.status_code==200,r.text
        assert p.calls["judge"]==2
        assert app.state.retriever.metrics["rounds"]==3


async def test_concurrent_replay_single_write(stack):
    import asyncio
    app,p=stack
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test",headers=auth()) as c:
        a,b=await asyncio.gather(c.post("/add",json=add_body()),c.post("/add",json=add_body()))
        assert a.status_code==b.status_code==200
        assert app.state.store.stats()["messages"]==1
        assert p.calls["extract"]==1


async def test_persisted_after_reopen(stack):
    app,p=stack
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test",headers=auth()) as c:
        await c.post("/add",json=add_body())
    other=create_app(app.state.settings,Provider())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=other),base_url="http://test",headers=auth()) as c:
        r=await c.post("/search",json={"user_id":"u","query":"ZX-482","top_k":10})
        assert "ZX-482" in json.dumps(r.json())


@pytest.mark.parametrize("body",[{"user_id":"u","query":"x","top_k":101},{"user_id":"u","query":"","top_k":1},{"user_id":"u","query":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}],"top_k":1}])
async def test_bad_search_contract(stack,body):
    app,p=stack
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test",headers=auth()) as c:
        r=await c.post("/search",json=body)
        assert r.status_code==422
        assert "input" not in r.text


@pytest.mark.parametrize("raw",['```json\n{"cells": []}\n```',"{'cells': [],}",'{"cells": []','{"cells":null}'])
def test_lenient_local_repair(raw):
    data,mode=parse_output(raw)
    assert isinstance(data,dict)


def test_source_attribution_not_guessed():
    messages=[{"id":"a","role":"user"},{"id":"b","role":"user"}]
    value={"cells":[{"sources":["invented"],"facts":["unsupported"],"summary":"bad"}]}
    n=normalize(value,messages,[])
    assert len(n["cells"])==2
    assert all(not c["facts"] for c in n["cells"])


@pytest.mark.parametrize("value",[{}, {"is_sufficient":"false"}, {"decision":"sufficient","support_ids":["missing"]}])
def test_judge_never_fail_open(value):
    assert decision(value,{"real"})["decision"]!="sufficient"


def test_multi_image_caption_mapping():
    content=[{"type":"text","text":"Image Caption: red"},{"type":"image_url","image_url":{"url":picture()}},
             {"type":"text","text":"Image Caption: blue"},{"type":"image_url","image_url":{"url":picture('blue')}}]
    mapping,raw=split_captions(content)
    assert mapping=={1:"red",3:"blue"}
    assert raw==""


async def test_state_history_retained(stack):
    app,p=stack
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test",headers=auth()) as c:
        await c.post("/add",json=add_body())
        await c.post("/add",json=add_body('r2','小雨后来把红色背包换成蓝色。'))
        assert app.state.store.stats()["states"]==2
        r=await c.post("/search",json={"user_id":"u","query":"小雨背包颜色的变化","top_k":100})
        assert "蓝色" in json.dumps(r.json(),ensure_ascii=False)

async def test_cell_revision_replaces_derived_indexes(tmp_path):
    class Continuing(Provider):
        async def chat(self,*args,**kwargs):
            r=await super().chat(*args,**kwargs)
            if args[2]=='extract':
                payload=json.loads(args[1]);old=payload.get('open_cell')
                if old:r['cells'][0]['continue_cell']=old['id']
            return r
    p=Continuing();app=create_app(Settings(data_dir=str(tmp_path/'b'),service_token='test-token'),p)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test',headers=auth()) as c:
        await c.post('/add',json=add_body())
        before=app.state.store.snapshot('u');oldid=next(iter(before['cells']))
        await c.post('/add',json=add_body('r2','小雨现在把背包换成蓝色。'))
        after=app.state.store.snapshot('u');cell=json.loads(after['cells'][oldid]['body'])
        assert len(after['cells'])==1 and cell['revision']==2
        assert len(cell['facts'])==2 and len(cell['sources'])==2
        assert len([d for d in after['docs'] if d['kind']=='cell'])==1
        assert app.state.store.scene_candidates('u')[0]['count']==1
        assert app.state.store.stats()['states']==2


async def test_atomic_publish_and_retry(stack):
    from aurel_memory.provider import UpstreamError
    app,p=stack;original=p.embed
    async def unavailable(texts):raise UpstreamError('embedding unavailable')
    p.embed=unavailable
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test',headers=auth()) as c:
        r=await c.post('/add',json=add_body());assert r.status_code==503
        assert r.headers['retry-after']=='5'
        assert app.state.store.stats()['requests']==0
        assert app.state.store.stats()['messages']==0
        p.embed=original
        assert (await c.post('/add',json=add_body())).status_code==200
        assert app.state.store.stats()['messages']==1


async def test_provider_format_error_never_retries(tmp_path):
    from aurel_memory.provider import Provider as RealProvider
    from types import SimpleNamespace
    p=RealProvider(Settings(data_dir=str(tmp_path),api_key='fake'))
    count=0
    async def create(**kwargs):
        nonlocal count
        count+=1
        assert kwargs['response_format']=={'type':'json_object'}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Malformed and missing JSON'),finish_reason='stop')])
    p.client.chat.completions.create=create
    try:
        r=await p.chat('JSON evidence','source','extract',cache_scope='x')
        assert r=={} and count==1
        assert await p.chat('JSON evidence','source','extract',cache_scope='x')=={}
        assert count==1
    finally:await p.close()


def test_budget_image_count_and_caption_dedup(tmp_path):
    from aurel_memory.packing import pack
    from aurel_memory.media import store_image
    from aurel_memory.schemas import SearchRequest
    from aurel_memory.util import tokens
    snap={'images':{},'messages':{},'cells':{}};candidates=[]
    for i in range(50):
        asset=store_image(tmp_path,picture((i*5,50,100)));oid=str(i)
        snap['images'][oid]={'id':oid,'asset':asset['id'],'metadata':asset,'caption':'Unique caption number '+str(i),'caption_source':'provided','context':'','sources':[]}
        candidates.append({'id':'image:'+oid,'kind':'image','parent':oid,'sources':[]})
    req=SearchRequest(user_id='u',query='images',top_k=100)
    s=Settings(image_base_tokens=85,image_tile_tokens=170)
    out,meta=pack(candidates,snap,req,s,{x['id'] for x in candidates},set())
    images=[p for d in out['data'] for p in d['content'] if p['type']=='image_url']
    assert len(images)==12
    assert len(out['data'])==42
    assert sum(p['text'].count('Image Caption:') for d in out['data'] for p in d['content'] if p['type']=='text')==42
    small=SearchRequest(user_id='u',query='images',top_k=3)
    assert len(pack(candidates,snap,small,s,{x['id'] for x in candidates},set())[0]['data'])==3


def test_unstated_year_is_not_fabricated():
    msg={'id':'a','role':'user','timestamp':1790812800000,'text':'她10月1日去了苏州。'}
    value={'cells':[{'sources':['a'],'facts':[{'text':'去苏州','time_basis':'10月1日','valid_from':'2023-10-01'}]}]}
    fact=normalize(value,[msg],[])['cells'][0]['facts'][0]
    assert fact['valid_from']=='' and fact['time_basis']=='10月1日'


async def test_shared_asset_concurrent_writes(tmp_path):
    import asyncio
    from aurel_memory.media import store_image,asset_uri
    uri=picture()
    results=await asyncio.gather(*(asyncio.to_thread(store_image,tmp_path,uri) for _ in range(16)))
    assert len({r['id'] for r in results})==1
    assert all(asset_uri(r)==uri for r in results)


def test_parent_sources_repaired_from_valid_new_fact():
    messages=[{'id':'old','role':'user','history':True},{'id':'new','role':'user','text':'changed to blue'}]
    obj={'cells':[{'sources':['old'],'facts':[{'text':'changed to blue','sources':['new']}]}]}
    r=normalize(obj,messages,[])
    assert len(r['cells'])==1 and set(r['cells'][0]['sources'])=={'old','new'}
    assert r['cells'][0]['facts'][0]['text']=='changed to blue'


def test_short_aliases_restore_only_identifier_fields():
    from aurel_memory.normalize import alias_input,restore_ids,caption_text
    payload={'earlier_context':[],'new_messages':[{'id':'long-message-id','images':[{'id':'long-image-id'}]}]}
    encoded,aliases=alias_input(payload)
    assert encoded['new_messages'][0]['id']=='M1'
    restored=restore_ids({'cells':[{'sources':'m1','summary':'M1 is literal text'}],'images':[{'id':'I1'}]},aliases)
    assert restored['cells'][0]['sources']=='long-message-id'
    assert restored['cells'][0]['summary']=='M1 is literal text'
    assert restored['images'][0]['id']=='long-image-id'
    assert 'ZX-482' in caption_text({'caption':'red bag','ocr':['ZX-482']})


def test_unsupported_year_removed_from_derived_text_only():
    from aurel_memory.normalize import grounded_dates
    assert grounded_dates('小雨于2023年10月1日出行。','10月1日出行。')=='小雨于10月1日出行。'
    assert grounded_dates('2023年10月1日出行。','2023年10月1日出行。')=='2023年10月1日出行。'
    assert grounded_dates('设备代码 ZX-2023','设备代码 ZX-2023')=='设备代码 ZX-2023'


def test_budget_keeps_retrieved_tail_span_instead_of_unrelated_prefix():
    from aurel_memory.packing import pack
    from aurel_memory.schemas import SearchRequest
    from aurel_memory.util import tokens
    raw='unrelated context '*20000+'The serial code is ZX-482.'
    m={'id':'m','role':'user','session':'s','seq':0,'timestamp':None,'raw':raw}
    snap={'messages':{'m':m},'images':{},'cells':{}}
    cand={'id':'message:m','kind':'message','parent':'m','sources':['m'],'source_spans':{'m':[(len(raw)-25,len(raw))]}}
    req=SearchRequest(user_id='u',query='What is the serial code?',top_k=1)
    out,meta=pack([cand],snap,req,Settings(input_budget=3000,template_reserve=256),set(),set())
    text=out['data'][0]['content']
    assert 'ZX-482' in text and 'source chars' in text
    assert 'message:m' not in meta['complete_ids']
    assert tokens(text)<2744


async def test_failed_reranker_falls_back_once_per_search(tmp_path):
    p=Provider();p.judgments=[{'decision':'insufficient','query':'blue backpack'},{'decision':'unknown','query':'travel backpack'}]
    async def unavailable(query,docs):p.calls['failed_rerank']+=1;return None
    p.rerank=unavailable
    app=create_app(Settings(data_dir=str(tmp_path/'c'),profile='C',service_token='test-token'),p)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test',headers=auth()) as c:
        await c.post('/add',json=add_body())
        r=await c.post('/search',json={'user_id':'u','query':'小雨背包','top_k':100})
        assert r.status_code==200 and r.json()['data']
        assert p.calls['failed_rerank']==1
        assert app.state.retriever.metrics['rerank_degraded_searches']==1


@pytest.mark.parametrize('caption',['Image Caption: ""','Image Caption: ]','Image Caption: 2025-09-09 08:10:00'])
def test_placeholder_caption_does_not_skip_visual_description(caption):
    mapping,raw=split_captions([{'type':'text','text':caption},{'type':'image_url','image_url':{'url':picture()}}])
    assert not mapping and caption in raw


def test_ambiguous_after_image_captions_are_not_assigned_to_wrong_image():
    content=[{'type':'image_url','image_url':{'url':picture()}},{'type':'text','text':'Image Caption: red'},
             {'type':'image_url','image_url':{'url':picture('blue')}},{'type':'text','text':'Image Caption: blue'}]
    mapping,raw=split_captions(content)
    assert mapping=={} and 'red' in raw and 'blue' in raw


def test_generic_fact_label_with_explicit_state_semantics_is_accepted():
    msg={'id':'m','role':'user','text':'小雨换成了蓝色背包。'}
    obj={'cells':[{'sources':['m'],'facts':[{'text':msg['text'],'subject':'小雨','predicate':'更换','object':'蓝色背包','scope':'背包','kind':'fact','assertion':'actual','time_semantics':'current state','action':'assert'}]}]}
    f=normalize(obj,[msg],[])['cells'][0]['facts'][0]
    assert f['kind']=='state' and f['state_slot']=='current_possession' and f['action']=='replace'
    msg['role']='assistant'
    assert normalize(obj,[msg],[])['cells'][0]['facts'][0]['kind']=='fact'


async def test_retrieval_crosses_session_boundary(stack):
    app,p=stack
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test',headers=auth()) as c:
        await c.post('/add',json=add_body(session='first'))
        await c.post('/add',json=add_body('r2','小雨后来把红色背包换成了蓝色。',session='later'))
        r=await c.post('/search',json={'user_id':'u','query':'小雨的背包从什么颜色变成什么颜色？','top_k':100})
        assert r.status_code==200 and '红色' in r.text and '蓝色' in r.text


def test_preference_facet_recovers_counterexample():
    from aurel_memory.retrieval import Index,Retriever
    docs=[];messages={}
    for i,(text,facet,v) in enumerate([('vegetarian meal','diet',[1,0]),('ate beef on a trip','diet',[0,1]),('travel by bus','transport',[0,1])]):
        mid='m'+str(i);cid='c'+str(i)
        messages[mid]={'id':mid,'session':'s'+str(i),'seq':0,'raw':text,'role':'user','timestamp':None}
        docs.append({'id':'d'+str(i),'kind':'fact','text':text,'parent':cid,'sources':[mid],'entities':[],'vec':np.array(v,dtype=np.float32),'sigvec':None,'meta':{'kind':'behavior','subject':'user','facet':facet,'observed':i}})
    snapshot={'docs':docs,'messages':messages,'images':{},'cells':{},'version':1}
    candidates=Retriever(Settings(),None,None).recall(Index(snapshot),'vegetarian',np.array([1,0],dtype=np.float32))
    found={c['parent'] for c in candidates}
    assert 'c0' in found and 'c1' in found and 'c2' not in found
