import asyncio
from datetime import datetime,timezone
import time
from pathlib import Path
import numpy as np
from . import prompts
from .media import split_captions,images_of,store_image,text_of,asset_uri
from .normalize import normalize,alias_input,restore_ids
from .util import ident,dumps,clip,tokens,chunks,canonical,normvec,iso


class Ingestor:
    def __init__(self,s,store,provider):
        self.s=s; self.store=store; self.p=provider

    async def add(self,req):
        payload=req.model_dump(exclude_none=True)
        user,rid,session=req.user_id,req.request_id,req.session_id
        ph=ident(payload)
        replay=self.store.replay(user,rid,ph)
        if replay:return replay
        history=self.store.tail(user,session,6)
        nextseq=(history[-1]["seq"]+1) if history else 0
        previous=self.store.open_cell(user,session)
        if previous and len(previous['sources'])<12:
            extra=self.store.get_messages(user,previous['sources'])
            history=sorted({m['id']:m for m in [*extra.values(),*history]}.values(),key=lambda m:m['seq'])
        else:previous=None
        messages=[]; images=[]; assets=[]; missing=[]
        old_images=self.store.images_for_messages(user,[m["id"] for m in history])
        created=int(time.time()*1000)
        for i,m in enumerate(payload["messages"]):
            mid=ident(user,rid,i)
            mapping,raw=split_captions(m["content"])
            content=m["content"] if isinstance(m["content"],str) else [dict(p) for p in m["content"]]
            for part,uri in images_of(m["content"]):
                # Shared immutable pixels; every lookup still requires a scoped occurrence.
                asset=await asyncio.to_thread(store_image,Path(self.s.data_dir).parent/"shared-media",uri)
                assets.append(asset)
                im={"id":ident(mid,part),"message":mid,"asset":asset["id"],"caption":mapping.get(part,""),
                    "context":"","sources":[mid],"entities":[],"caption_source":"provided" if part in mapping else "generated",
                    "vision":None,"metadata":asset}
                images.append(im)
                content[part]={"type":"image_ref","asset":asset["id"]}
                if part not in mapping:missing.append((im,uri))
            messages.append({"id":mid,"user":user,"session":session,"seq":nextseq+i,
                             "role":m["role"],"timestamp":m.get("timestamp"),"content":content,"raw":raw,"created":created})
        async def cap(im,uri):
            im["caption"]=await self.p.caption(uri,(user,im["asset"]))
        await asyncio.gather(*(cap(im,uri) for im,uri in missing))
        # Model never receives provided-caption pixels during text extraction.
        model_messages=[]
        for m in history+messages:
            caps=[{"id":im["id"],"caption":im["caption"],"caption_source":im["caption_source"]} for im in old_images+images if im["message"]==m["id"]]
            model_messages.append({"id":m["id"],"role":m["role"],"timestamp":m["timestamp"],"timestamp_iso_utc":iso(m['timestamp']),
                "text":m["raw"],"images":caps,"history":m in history})
        scene_candidates=self.store.scene_candidates(user)
        # Recall scene candidates by current text embedding; no history-vector reload on Add.
        scene_prompt=[]
        if scene_candidates:
            seed="\n".join(m["raw"] for m in messages)+"\n"+"\n".join(im["caption"] for im in images)
            sv=(await self.p.embed([clip(seed,1500) or "image memory"]))[0]
            ordered=sorted(scene_candidates,key=lambda x:float(normvec(x["centroid"])@sv),reverse=True)[:8]
            scene_prompt=[{"id":x["id"],"title":x["title"],"scope":x["scope"],"entities":x["entities"],"summary":clip(x["summary"],250)} for x in ordered]
        batches=[]; batch=[]; size=0
        for m in model_messages:
            if m["history"]:continue
            # Segment long inputs, retaining IDs and all content; no fixed character truncation.
            combined=m["text"]+"\n"+"\n".join("Image "+x["id"]+" Caption: "+x["caption"] for x in m["images"])
            for segment,start,end in chunks(combined or "[image]",2500,80):
                item={**m,"text":segment,"images":[{"id":x["id"],"caption_source":x["caption_source"]} for x in m["images"]],"segment_start":start}
                cost=tokens(segment)
                if batch and (size+cost>5000 or len(batch)>=8):
                    batches.append(batch);batch=[];size=0
                batch.append(item);size+=cost
        if batch:batches.append(batch)
        extracted=[]; image_updates={}
        hist=[{**m,"text":clip(m["text"],500),"images":[{**x,"caption":clip(x["caption"],400)} for x in m["images"]]} for m in model_messages if m["history"]]
        async def extract(batch):
            open_cell={k:v for k,v in previous.items() if not k.startswith('_')} if previous else None
            prompt_input,aliases=alias_input({"earlier_context":hist,"new_messages":batch,"scene_candidates":scene_prompt,"open_cell":open_cell})
            obj=await self.p.chat(prompts.EXTRACT,dumps(prompt_input),
                                  "extract",7000,cache_scope=(user,rid))
            return normalize(restore_ids(obj,aliases),hist+batch,old_images+images)
        results=await asyncio.gather(*(extract(b) for b in batches))
        for result in results:
            self.p.calls['extraction_fallback_cells']+=sum(bool(c.get('fallback')) for c in result['cells'])
            extracted.extend(result["cells"])
            for im in result["images"]:
                prev=image_updates.get(im["id"])
                if prev:
                    im["context"]="\n".join(dict.fromkeys([prev["context"],im["context"]]))
                    im["sources"]=list(dict.fromkeys(prev["sources"]+im["sources"]))
                image_updates[im["id"]]=im
        touched=[];new_image_ids={im['id'] for im in images}
        for im in old_images+images:
            up=image_updates.get(im["id"])
            if up:
                im["context"]="\n".join(dict.fromkeys(x for x in (im["context"],up["context"]) if x))
                im["sources"]=list(dict.fromkeys(im["sources"]+up["sources"]))
                im["entities"]=list(dict.fromkeys(im["entities"]+up["entities"]))
            if im['id'] in new_image_ids or up:touched.append(im)
        all_msg={m["id"]:m for m in history+messages}
        docs=[];cells=[]
        def doc(kind,text,parent,sources,entities=(),meta=None,stable=None):
            if not text.strip():return
            did=stable or ident(user,rid,kind,parent,len(docs),text)
            docs.append({"id":did,"kind":kind,"text":text,"parent":parent,"sources":list(sources),
                         "entities":list(entities),"meta":meta or {},"sigvec":None})
        for m in messages:
            for text,start,end in chunks(m["raw"]):
                doc("raw",text,m["id"],[m["id"]],meta={"start":start,"end":end,"role":m["role"],"timestamp":m["timestamp"]})
        for im in touched:
            if im['id'] in new_image_ids:
                for text,start,end in chunks(im["caption"]):
                    doc("caption",text,im["id"],[im["message"]],im["entities"],{"start":start,"end":end,"image":im["id"]})
            context=im["context"]
            # Even if extraction omits an image update, direct nearby user text remains searchable.
            nearby=[m["raw"] for m in history+messages if abs(m["seq"]-all_msg[im["message"]]["seq"])<=2 and m["raw"]]
            context="\n".join(dict.fromkeys([context,*nearby])).strip()
            if context:
                for n,(text,start,end) in enumerate(chunks(context)):
                    doc("image_context",text,im["id"],im["sources"],im["entities"],{"image":im["id"]},stable=ident(im["id"],"context",n))
        continued=False
        for i,cell in enumerate(extracted):
            cid=ident(user,rid,"cell",i)
            continuing=bool(previous and not continued and cell.get('continue_cell')==previous['id']
                and len(set(cell['sources']+previous['sources']))<=12
                and (set(cell['sources']) & set(previous['sources']) or
                     {canonical(e) for e in cell['entities']} & {canonical(e) for e in previous['entities']}))
            if continuing:
                cid=previous['id'];continued=True
                cell['sources']=list(dict.fromkeys(previous['sources']+cell['sources']))
                cell['entities']=list(dict.fromkeys(previous['entities']+cell['entities']))
                cell['facts']=previous['facts']+cell['facts']
                cell['scene']=previous['scene']
            cell.update(id=cid,session=session,observed=max((all_msg[x]["timestamp"] or created for x in cell["sources"]),default=created),
                        status="provisional" if i==len(extracted)-1 and len(cell['sources'])<12 else "finalized",
                        revision=previous['revision']+1 if continuing else 1)
            if not cell["summary"]:
                cell["summary"]="\n".join(clip(all_msg[x]["raw"],500) for x in cell["sources"] if x in all_msg)
            cell["summary"]=clip(cell["summary"],1600)
            cell["title"]=clip(cell["title"],100)
            for j,f in enumerate(cell["facts"]):
                f["id"]=f.get('id') or ident(cid,"fact",rid,j,f["text"])
                f.setdefault('observed',cell['observed'])
                if f["valid_from"] and not f["time_basis"]:f["valid_from"]=""
                if f["valid_to"] and not f["time_basis"]:f["valid_to"]=""
                if f["kind"]=="state" and f["assertion"]!="actual":
                    f["action"]="assert"  # preserve plans/uncertain reports without replacing actual state
                doc("fact",clip(" ".join([f["subject"],f["predicate"],f["text"],f["valid_from"],f["time_basis"],
                    f.get('scope',''),f.get('time_semantics',''),f['kind'],f['assertion'],f['action'],f['polarity']]),1000),
                    cid,f["sources"],cell["entities"],{**f,"cell":cid},stable=f["id"])
            doc("cell",cell["title"]+"\n"+cell["summary"],cid,cell["sources"],cell["entities"],{"cell":cid,"observed":cell["observed"]},stable=ident(cid,"episode"))
            cells.append(cell)
        vectors=await self.p.embed([d["text"] for d in docs])
        for d,v in zip(docs,vectors):d["vec"]=v
        scene_map={sc["id"]:sc for sc in scene_candidates};changed={}
        for cell in cells:
            d=next(d for d in docs if d["kind"]=="cell" and d["parent"]==cell["id"])
            cv=d["vec"]; entities={canonical(x) for x in cell["entities"] if canonical(x)}
            selected=None;best=-1
            if previous and cell['id']==previous['id']:
                selected=scene_map.get(previous['scene'])
            for sc in scene_map.values():
                if previous and cell['id']==previous['id'] and selected:break
                overlap=bool(entities & {canonical(x) for x in sc["entities"]})
                score=float(normvec(sc["centroid"])@cv)
                nominated=cell["scene"]==sc["id"]
                age=abs(cell["observed"]-sc["updated"])/86400000
                scope_ok=sc["scope"]=="topic" or age<=14
                if scope_ok and ((score>=0.80) or (overlap and score>=0.58) or (nominated and score>=0.58)) and score>best:
                    selected=sc;best=score
            if selected is None:
                sid=ident(user,cell["id"],"scene")
                selected={"id":sid,"title":cell["title"],"summary":"","scope":cell["scope"],"entities":[],
                          "centroid":np.zeros_like(cv),"count":0,"updated":cell["observed"]}
                scene_map[sid]=selected
            n=selected["count"]
            replacing=bool(previous and cell['id']==previous['id'] and selected['id']==previous['scene'] and '_old_vec' in previous)
            if replacing:
                selected['centroid']=(selected['centroid']*n-previous['_old_vec']+cv)/max(1,n)
            else:
                selected["centroid"]=(selected["centroid"]*n+cv)/(n+1)
                selected["count"]=n+1
            selected["updated"]=max(selected["updated"],cell["observed"])
            selected["entities"]=list(dict.fromkeys(selected["entities"]+cell["entities"]))[:100]
            cell["scene"]=selected["id"];d["meta"]["scene"]=selected["id"]
            changed[selected["id"]]=selected
            for fd in docs:
                if fd["parent"]==cell["id"]:fd["meta"]["scene"]=selected["id"]
        before=len(docs)
        for sc in changed.values():
            current=[c for c in cells if c['scene']==sc['id']]
            earlier=self.store.scene_summaries(user,sc['id'],{c['id'] for c in current})
            members=sorted(current+earlier,key=lambda c:c['observed'],reverse=True)[:12]
            sc['summary']='\n'.join(c['title']+': '+clip(c['summary'],160) for c in members)
            doc("scene",sc["title"]+"\n"+sc["summary"],sc["id"],[],sc["entities"],{"scene":sc["id"]},stable=ident(sc["id"],"summary"))
        if len(docs)>before:
            vs=await self.p.embed([d["text"] for d in docs[before:]])
            for d,v in zip(docs[before:],vs):d["vec"]=v
        if self.s.profile=="A":
            if images:
                _,iv=await self.p.vision(images=[asset_uri(im["metadata"]) for im in images])
                if len(iv)!=len(images):raise ValueError("image vector count mismatch")
                for im,v in zip(images,iv):im["vision"]=v
            # Worker segments by the SigLIP tokenizer; one node per short original-text chunk.
            sigdocs=[]
            for d in list(docs):
                if d["kind"]!="raw":continue
                r=await self.p.worker.post("/siglip-text-chunks",json={"text":d["text"]})
                r.raise_for_status()
                for j,row in enumerate(r.json()["chunks"]):
                    sigdocs.append({**d,"id":ident(d["id"],"siglip",j),"kind":"sigtext","text":row["text"],
                                    "sigvec":normvec(row["vector"])})
            docs.extend(sigdocs)
        response={"success":True,"request_id":rid,"user_id":user,"session_id":session}
        return self.store.publish(user,rid,ph,messages,assets,touched,cells,docs,list(changed.values()),response)
