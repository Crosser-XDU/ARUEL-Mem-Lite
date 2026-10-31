import asyncio
from collections import Counter,defaultdict,OrderedDict
import json
import math
import re
import numpy as np
from . import prompts
from .media import text_of,images_of
from .normalize import decision,caption_text
from .util import terms,canonical,clip,tokens,dumps,ident,normvec
from .packing import pack, candidate_text, image_tokens


class Index:
    def __init__(self,snapshot):
        self.s=snapshot
        self.docs=snapshot["docs"]
        self.matrix=np.stack([d["vec"] for d in self.docs]) if self.docs else np.empty((0,1024),np.float32)
        self.postings=defaultdict(list);self.lengths=[]
        self.entity_docs=defaultdict(set);self.parent_docs=defaultdict(list);self.scene_docs=defaultdict(list)
        self.state_docs=defaultdict(list);self.preference_docs=defaultdict(list)
        for i,d in enumerate(self.docs):
            words=terms(d["text"]);count=Counter(words);self.lengths.append(len(words))
            for w,n in count.items():self.postings[w].append((i,n))
            self.parent_docs[d["parent"]].append(i)
            if d["meta"].get("scene"):self.scene_docs[d["meta"]["scene"]].append(i)
            for ent in d["entities"]:
                if canonical(ent):self.entity_docs[canonical(ent)].add(i)
            m=d['meta']
            if d['kind']=='fact' and m.get('kind')=='state':
                key=(canonical(m.get('subject','')),canonical(m.get('state_slot') or m.get('predicate','')),canonical(m.get('scope','')))
                if key[0] and key[1]:self.state_docs[key].append(i)
            if d['kind']=='fact' and m.get('kind') in {'preference','behavior'}:
                key=(canonical(m.get('subject','')),canonical(m.get('facet') or m.get('predicate','')))
                if key[1]:self.preference_docs[key].append(i)
        self.lengths=np.asarray(self.lengths,dtype=np.float32)
        self.avg=float(self.lengths.mean()) if len(self.lengths) else 1
        self.size=self.matrix.nbytes*2+sum(len(d["text"])*4+2000 for d in self.docs)+sum(len(d["raw"])*4+1000 for d in snapshot["messages"].values())

    def bm25(self,query):
        n=len(self.docs);scores=np.zeros(n,dtype=np.float32)
        for word in set(terms(query)):
            rows=self.postings.get(word,())
            if not rows:continue
            idf=math.log(1+(n-len(rows)+0.5)/(len(rows)+0.5))
            for i,tf in rows:
                scores[i]+=idf*tf*2.2/(tf+1.2*(.25+.75*self.lengths[i]/max(1,self.avg)))
        return scores


class Retriever:
    def __init__(self,s,store,provider):
        self.s=s;self.store=store;self.p=provider
        self.cache=OrderedDict();self.cache_size=0;self.metrics=Counter()

    async def index(self,user):
        version=self.store.version(user)
        old=self.cache.get(user)
        if old and old.s["version"]==version:
            self.cache.move_to_end(user);return old
        snap=await asyncio.to_thread(self.store.snapshot,user)
        idx=await asyncio.to_thread(Index,snap)
        if old:self.cache_size-=old.size;self.cache.pop(user,None)
        if idx.size<=self.s.cache_bytes:
            while self.cache and self.cache_size+idx.size>self.s.cache_bytes:
                _,v=self.cache.popitem(last=False);self.cache_size-=v.size
            self.cache[user]=idx;self.cache_size+=idx.size
        return idx

    @staticmethod
    def key(d,idx):
        if d["kind"] in {"caption","image_context"}:return "image:"+d["parent"]
        if d["kind"]=="scene":return "scene:"+d["parent"]
        if d["kind"] in {"fact","cell"}:return "cell:"+d["parent"]
        return "message:"+d["parent"]

    def recall(self,idx,query,vec,sigtext=None,sigimages=()):
        dense=idx.matrix@vec;sparse=idx.bm25(query)
        scores=defaultdict(float);hits=defaultdict(list);rawscore=defaultdict(float)
        families=[("caption",.35),("image_context",.15),("raw",.5),("fact",.5),("cell",.25),("scene",.2)]
        for kind,w in families:
            ids=[i for i,d in enumerate(idx.docs) if d["kind"]==kind]
            for values,weight in ((dense,w),(sparse,w)):
                ordered=sorted(ids,key=lambda i:float(values[i]),reverse=True)[:100]
                voted=set()
                rank=0
                for i in ordered:
                    if values[i]<=0 or (values is dense and values[i]<.16):continue
                    d=idx.docs[i];key=self.key(d,idx)
                    # One parent vote per channel; repeated chunks do not multiply relevance.
                    hits[key].append(i);rawscore[key]=max(rawscore[key],float(dense[i]))
                    if key in voted:continue
                    rank+=1;voted.add(key);scores[key]+=weight/(60+rank)
        # Merge Caption/context dense+BM25 first, then fuse that rank with the pixel rank.
        image_text=sorted((k for k in scores if k.startswith('image:')),key=scores.get,reverse=True)
        image_text_ranks={key:rank for rank,key in enumerate(image_text,1)}
        for rank,key in enumerate(image_text,1):scores[key]=1/(60+rank)
        if sigtext is not None or sigimages:
            imscores=[]
            for im in idx.s["images"].values():
                if im["vision"] is None:continue
                sims=[]
                if sigtext is not None:sims.append(max(float(im["vision"]@v) for v in sigtext))
                if sigimages:sims.append(max(float(im["vision"]@q) for q in sigimages))
                imscores.append((max(sims),im["id"]))
            for rank,(_,oid) in enumerate(sorted(imscores,reverse=True)[:100],1):
                scores["image:"+oid]+=1/(60+rank)
            if sigimages:
                sims=[]
                for i,d in enumerate(idx.docs):
                    if d["sigvec"] is not None:
                        sims.append((max(float(d["sigvec"]@q) for q in sigimages),i))
                seen=set()
                for _,i in sorted(sims,reverse=True)[:100]:
                    key=self.key(idx.docs[i],idx)
                    if key not in seen:scores[key]+=.4/(60+len(seen)+1);seen.add(key)
                    hits[key].append(i)
        # Scene-guided expansion is additional to (never a replacement for) direct recall.
        scenes=defaultdict(float)
        for key in sorted(scores,key=scores.get,reverse=True)[:50]:
            for i in hits[key]:
                sid=idx.docs[i]["meta"].get("scene")
                if sid:scenes[sid]=max(scenes[sid],scores[key])
        bridges=set()
        # Slot/facet expansion recovers state changes and behavioral counterexamples.
        seeds={i for key in sorted(scores,key=scores.get,reverse=True)[:30] for i in hits[key]}
        for grouping in (idx.state_docs,idx.preference_docs):
            for members in grouping.values():
                if not seeds.intersection(members):continue
                ordered=sorted(members,key=lambda i:idx.docs[i]['meta'].get('observed',0))
                chosen=list(dict.fromkeys(ordered[-6:]+ordered[:2]+sorted(members,key=lambda i:float(dense[i]),reverse=True)[:4]))
                for rank,i in enumerate(chosen,1):
                    d=idx.docs[i];key=self.key(d,idx)
                    scores[key]=max(scores[key],.85/(60+rank));hits[key].append(i);bridges.add(key)
        for sid in sorted(scenes,key=scenes.get,reverse=True)[:4]:
            candidates=sorted(idx.scene_docs[sid],key=lambda i:float(dense[i]),reverse=True)
            parents=set()
            for i in candidates:
                d=idx.docs[i];key=self.key(d,idx)
                if key in parents:continue
                if len(parents)>=6:break
                parents.add(key);scores[key]=max(scores[key],scenes[sid]*.7);hits[key].append(i);bridges.add(key)
        # Entity bridges (up to two hops, bounded nodes), plus their linked image occurrences.
        frontier=set()
        for key in sorted(scores,key=scores.get,reverse=True)[:20]:
            for i in hits[key]:frontier.update(canonical(e) for e in idx.docs[i]["entities"] if canonical(e))
        graph_added=set()
        for _ in range(2):
            next_frontier=set()
            candidates=set().union(*(idx.entity_docs[e] for e in frontier)) if frontier else set()
            for i in sorted(candidates,key=lambda i:float(dense[i]),reverse=True):
                if len(graph_added)>=24:break
                d=idx.docs[i];key=self.key(d,idx)
                if key in graph_added:continue
                graph_added.add(key);scores[key]=max(scores[key],.6/(60+len(graph_added)))
                hits[key].append(i);bridges.add(key)
                next_frontier.update(canonical(e) for e in d["entities"] if canonical(e))
            frontier=next_frontier-frontier
        message_images=defaultdict(list)
        for oid,im in idx.s["images"].items():message_images[im["message"]].append(oid)
        for key in list(sorted(scores,key=scores.get,reverse=True)[:60]):
            refs=set()
            for i in hits[key]:refs.update(idx.docs[i]["sources"])
            for ref in refs:
                for oid in message_images[ref]:
                    ik="image:"+oid;scores[ik]=max(scores[ik],scores[key]*.85)
        out=[]
        for key in scores:
            if key.startswith("scene:"):continue
            unique=list(dict.fromkeys(hits[key]))
            sources=[];entities=[];factdocs=[];view=[]
            source_spans=defaultdict(list);caption_spans=[]
            for i in sorted(unique,key=lambda i:float(dense[i]),reverse=True)[:6]:
                d=idx.docs[i];sources+=d["sources"];entities+=d["entities"];view.append(d["text"])
                if d["kind"]=="fact":factdocs.append(d)
                if d['kind']=='raw':source_spans[d['parent']].append((d['meta'].get('start',0),d['meta'].get('end',len(d['text']))))
                if d['kind']=='caption':caption_spans.append((d['meta'].get('start',0),d['meta'].get('end',len(d['text']))))
            typ,parent=key.split(":",1)
            if typ=="image":
                im=idx.s["images"][parent];sources=im["sources"]+sources
                if not view:view=[im["caption"],im["context"]]
            if typ=="cell":
                c=idx.s["cells"].get(parent)
                if c:sources+=json.loads(c["sources"])
            if typ=="message":sources.append(parent)
            sources=list(dict.fromkeys(s for s in sources if s in idx.s["messages"]))
            out.append({"id":key,"kind":typ,"parent":parent,"score":scores[key],"dense":rawscore[key],
                        "sources":sources,"entities":list(dict.fromkeys(entities)),"facts":factdocs,
                        "view":"\n".join(dict.fromkeys(view)),"bridge":key in bridges})
            out[-1]['caption_priority']=1/(60+image_text_ranks[key]) if key in image_text_ranks else 0
            out[-1]['source_spans']=dict(source_spans);out[-1]['caption_spans']=caption_spans
        return sorted(out,key=lambda c:c["score"],reverse=True)

    def pool(self,candidates):
        picked=[];seen=set()
        for predicate,limit in ((lambda c:c["kind"]=="image",40),(lambda c:c["kind"]!="image" and not c["bridge"],24),(lambda c:c["bridge"],16)):
            n=0
            for c in candidates:
                if c["id"] not in seen and predicate(c):
                    picked.append(c);seen.add(c["id"]);n+=1
                    if n>=limit:break
        for c in candidates:
            if len(picked)>=self.s.candidate_limit:break
            if c["id"] not in seen:picked.append(c);seen.add(c["id"])
        return sorted(picked,key=lambda c:c["score"],reverse=True)[:self.s.candidate_limit]

    async def search(self,req):
        idx=await self.index(req.user_id)
        if not idx.docs:return {"data":[]}
        query=req.model_dump(exclude_none=True)["query"]
        q0=text_of(query);qimages=images_of(query)
        if qimages:
            content=[{"type":"text","text":"Describe retrieval features of each query image; do not answer. Question: "+q0}]
            content += [{"type":"image_url","image_url":{"url":uri,"detail":"high"}} for _,uri in qimages]
            r=await self.p.chat(prompts.CAPTION,content,"query_image",3000,cache_scope=(req.user_id,ident(query)))
            if caption_text(r):q0+="\nQuery image description: "+caption_text(r)
        q0=q0 or "Find memories related to the query image."
        sigimages=[]
        if self.s.profile=="A" and qimages:
            _,sigimages=await self.p.vision(images=[uri for _,uri in qimages])
        pool={};image_eligible=set();pin=set();missing=[];q=q0;seen_queries=set();rank_cache={}
        rerank_available=self.s.profile=='C'
        for roundno in range(1,4):
            self.metrics["rounds"]+=1
            vec=(await self.p.embed([clip(q,1800)]))[0]
            sigtext=None
            if self.s.profile=="A":
                sigtext=await self.p.vision_query(clip(q,1800))
            current=await asyncio.to_thread(self.recall,idx,q,vec,sigtext,sigimages)
            admitted=self.pool(current)
            if rerank_available:
                new=[c for c in admitted if ident(c["id"],c["view"]) not in rank_cache]
                if new:
                    scores=await self.p.rerank(q0,[clip(candidate_text(c,idx.s),768) for c in new])
                    if scores is not None:
                        rank_cache.update({ident(c["id"],c["view"]):v for c,v in zip(new,scores)})
                    else:rerank_available=False;self.metrics['rerank_degraded_searches']+=1
                for c in admitted:
                    if ident(c["id"],c["view"]) in rank_cache:
                        c["rerank"]=float(rank_cache[ident(c["id"],c["view"])])
                if all("rerank" in c for c in admitted):admitted.sort(key=lambda c:c["rerank"],reverse=True)
                if rerank_available and roundno>1 and missing:
                    gap=admitted[:24]
                    gs=await self.p.rerank(q,[clip(candidate_text(c,idx.s),768) for c in gap])
                    if gs is not None:
                        for rank,(c,_) in enumerate(sorted(zip(gap,gs),key=lambda x:x[1],reverse=True),1):c["gap_rank"]=rank
                    else:rerank_available=False;self.metrics['rerank_degraded_searches']+=1
            # Retain all text-ranked image candidates for the extra-caption path.
            merged_order=admitted+[c for c in current if c["id"] not in {x["id"] for x in admitted}]
            for rank,c in enumerate(merged_order,1):
                c["round"]=roundno;c["rank"]=rank
                c["priority"]=1/(60+rank)
                if "gap_rank" in c:c["priority"]+=.3/(60+c["gap_rank"])
                old=pool.get(c["id"])
                if old:c['caption_priority']=max(c['caption_priority'],old['caption_priority'])
                if old is None or c["priority"]>old["priority"]:pool[c["id"]]=c
                elif old:old['caption_priority']=c['caption_priority']
            chosen_assets=set();n=0
            for c in merged_order:
                if c["kind"]!="image":continue
                asset=idx.s["images"][c["parent"]]["asset"]
                if asset in chosen_assets:continue
                image_eligible.add(c["id"]);chosen_assets.add(asset);n+=1
                if n==8:break
            ranked=sorted(pool.values(),key=lambda c:(c["id"] in pin,c["priority"]),reverse=True)
            result,delivered=pack(ranked,idx.s,req,self.s,image_eligible,pin)
            if roundno==3:break
            seen_queries.add(canonical(q))
            view=[];remaining=18000
            byid={c["id"]:c for c in ranked}
            for evidence in result["data"]:
                c=byid.get(evidence["id"])
                if not c:continue
                actual=evidence["content"]
                actual=actual if isinstance(actual,str) else "\n".join(p["text"] for p in actual if p["type"]=="text")
                text=clip(actual,1800)
                cost=tokens(text)
                if cost>remaining:continue
                remaining-=cost
                view.append({"id":c["id"],"text":text,"pixels_returned":c["id"] in delivered["pixels"],"excerpt_only":tokens(actual)>1800})
            try:
                payload=dumps({"question":q0,"options":req.options or [],"evidence":view,
                        "omitted_candidate_count":len(pool)-len(view)})
                judge_content=[{'type':'text','text':payload}];pixel_budget=50000;pixel_count=0
                for item in result['data']:
                    c=byid.get(item['id'])
                    if not c or c['kind']!='image' or item['id'] not in {v['id'] for v in view}:continue
                    cost=image_tokens(idx.s['images'][c['parent']]['metadata'],self.s)
                    if pixel_count>=2 or cost>pixel_budget:continue
                    for part in item['content']:
                        if part['type']=='image_url':
                            judge_content.extend([{'type':'text','text':'Pixels for evidence ID '+item['id']},part])
                            pixel_budget-=cost;pixel_count+=1
                raw=await self.p.chat(prompts.JUDGE,judge_content if pixel_count else payload,"judge",1800)
                dec=decision(raw,{v["id"] for v in view})
            except Exception:
                self.metrics["judge_unknown"]+=1
                dec={"decision":"unknown","support_ids":[],"missing":[],"query":"","need_pixels":[]}
            pin.update(dec["support_ids"])
            for iid in dec["need_pixels"]:
                if iid in pool and pool[iid]['kind']=='image':image_eligible.add(iid);pin.add(iid)
            if dec["decision"]=="sufficient" and set(dec["support_ids"]).issubset(delivered["complete_ids"]) and set(dec['need_pixels']).issubset(delivered['pixels']):
                self.metrics["early_stop"]+=1;break
            missing=dec["missing"]
            nextq=dec["query"].strip() or (q0+" "+" ".join(missing))
            if canonical(nextq) in seen_queries:
                ents=list(dict.fromkeys(e for c in ranked[:8] for e in c["entities"]))[:8]
                nextq=q0+" "+" ".join(ents+missing)
            if canonical(nextq) in seen_queries:break
            q=nextq
        # Repack after final pinned requirements, using the SAME snapshot and raw candidate pool.
        ranked=sorted(pool.values(),key=lambda c:(c["id"] in pin,c["priority"]),reverse=True)
        final,_=pack(ranked,idx.s,req,self.s,image_eligible,pin)
        self.metrics["searches"]+=1
        return final
