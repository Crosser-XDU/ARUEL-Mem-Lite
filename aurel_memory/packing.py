import json
import math
from .media import asset_uri
from .util import dumps,clip,tokens,iso,terms


def source_line(m):
    when=iso(m["timestamp"]) or "source time unknown"
    return f"[source={m['id']} role={m['role']} time={when}]\n{m['raw']}"


def candidate_text(c,snapshot):
    parts=[]
    if c["kind"]=="image":
        im=snapshot["images"][c["parent"]]
        parts.append("Image occurrence "+im["id"]+"; Caption source: "+im["caption_source"])
        # A short ranking view selects relevant text. Rendering later uses full Caption.
        parts.append(clip(c["view"] or im["caption"],350))
        if im["context"]:parts.append("Image context: "+clip(im["context"],120))
    else:parts.append(clip(c["view"],470))
    source_budget=240
    for sid in c["sources"]:
        m=snapshot["messages"].get(sid)
        if m and m["raw"] and source_budget>0:
            piece=clip(source_line(m),min(120,source_budget));parts.append(piece);source_budget-=tokens(piece)
    return "\n".join(dict.fromkeys(x for x in parts if x))


def image_tokens(asset,s):
    w,h=asset["width"],asset["height"]
    scale=min(1,2048/max(w,h));w*=scale;h*=scale
    scale=min(1,768/min(w,h));w*=scale;h*=scale
    return s.image_base_tokens+s.image_tile_tokens*math.ceil(w/512)*math.ceil(h/512)


def selected_excerpt(text,spans,query,budget):
    spans=list(spans)
    if not spans:
        for word in sorted(set(terms(query)),key=len,reverse=True):
            if len(word)<2:continue
            at=text.casefold().find(word)
            if at>=0:spans.append((max(0,at-250),min(len(text),at+650)))
            if len(spans)>=8:break
    if not spans:spans=[(0,min(len(text),budget*4))]
    pieces=[];used=[];left=max(1,budget-80)
    for start,end in spans:
        start=max(0,start-100);end=min(len(text),end+100)
        if any(a<=start and end<=b for a,b in used):continue
        segment=clip(text[start:end],max(1,left-32));piece=f'[source chars {start}:{start+len(segment)}]\n'+segment
        if tokens(piece)>left:break
        pieces.append(piece);used.append((start,start+len(segment)));left-=tokens(piece)
        if left<100:break
    return '\n'.join(pieces)+'\n[Other source spans omitted due to budget; full source retained in memory.]'


def pack(candidates,snapshot,req,s,image_eligible,pin):
    # Conservative configurable image accounting. Never count base64 characters as model tokens.
    q=req.model_dump(exclude_none=True)
    qtext=q["query"] if isinstance(q["query"],str) else "\n".join(p["text"] for p in q["query"] if p["type"]=="text")
    query_image_cost=0
    if isinstance(q["query"],list):
        from .media import decode
        for p in q["query"]:
            if p["type"]=="image_url":
                _,_,(w,h)=decode(p["image_url"]["url"])
                query_image_cost+=image_tokens({"width":w,"height":h},s)
    remaining=max(0,s.input_budget-s.template_reserve-query_image_cost-tokens(qtext+dumps(q.get("options",[]))))
    result=[];rendered_sources=set();rendered_occ=set();assets=set();image_bytes=0;pixels=set();ids=set();incomplete=set();rendered_captions={}
    messages=snapshot["messages"]
    sessions={}
    for m in messages.values():sessions.setdefault(m["session"],{})[m["seq"]]=m
    image_candidates=[c for c in candidates if c["kind"]=="image" and c["id"] in image_eligible]
    chosen=[];chosen_assets=set()
    # Keep the image-specific list globally ranked; later rounds can replace earlier candidates.
    for c in image_candidates:
        asset=snapshot["images"][c["parent"]]["asset"]
        if asset in chosen_assets:continue
        chosen.append(c);chosen_assets.add(asset)
        if len(chosen)>=s.response_image_limit:break
    chosen_ids={c["id"] for c in chosen}
    core=[c for c in candidates if c["id"] in pin or c["id"] in chosen_ids]
    # Direct text/relationship support is core, before supplementary image captions.
    core += [c for c in candidates if c not in core and c["kind"]!="image"][:24]
    extra=sorted([c for c in candidates if c["kind"]=="image" and c["id"] not in {x["id"] for x in core}],key=lambda c:c.get('caption_priority',0),reverse=True)[:s.caption_limit]
    rest=[c for c in candidates if c["kind"]!="image" and c["id"] not in {x["id"] for x in core}]

    def emit(c,allow_image,context_extra=False):
        nonlocal remaining,image_bytes
        if len(result)>=req.top_k or c["id"] in ids:return False
        parts=[];im=None;source_ids=list(c["sources"])
        abbreviated=False
        def without_repeated_caption(text):
            for oid in rendered_occ:
                cap=snapshot['images'][oid]['caption']
                if len(cap)>=24:text=text.replace(cap,'[Caption already included above]')
            return text
        if c["kind"]=="image":
            im=snapshot["images"][c["parent"]]
            if im["id"] in rendered_occ:return False
            capkey=(im['asset'],im['caption'])
            caption=im['caption']
            if tokens(caption)>max(1024,remaining-512):
                caption=selected_excerpt(caption,c.get('caption_spans',[]),qtext,max(256,min(8000,remaining//2)));abbreviated=True
            label='Image Caption: '+caption if capkey not in rendered_captions else 'Same image and Caption as occurrence '+rendered_captions[capkey]
            parts.append(f"[image occurrence={im['id']} caption_source={im['caption_source']}]\n"+label)
            if im["context"]:parts.append("Source-grounded image context: "+im["context"])
        elif c["kind"]=="cell":
            record=snapshot["cells"].get(c["parent"])
            if record:
                cell=json.loads(record["body"])
                facts=cell.get("facts",[])
                lines=[]
                for f in facts:
                    lines.append("[derived from "+",".join(f["sources"])+"; "+f["kind"]+"; "+f["assertion"]+"] "+without_repeated_caption(f["text"])+
                                 ''.join('; '+k+'='+str(f[k]) for k in ('valid_from','valid_to','time_basis','scope','polarity','action') if f.get(k)))
                parts.append("Event: "+cell["title"]+"\n"+without_repeated_caption(cell["summary"])+"\n"+"\n".join(lines))
                source_ids=list(dict.fromkeys(source_ids+cell["sources"]))
        # Required explicit dependencies always travel with the main evidence.
        new_sources=[]
        for sid in source_ids:
            if sid in rendered_sources:continue
            m=messages.get(sid)
            if m and m["raw"]:
                line=source_line(m)
                limit=max(512,min(12000,remaining//max(1,len(source_ids))))
                if tokens(line)>limit:
                    spans=c.get('source_spans',{}).get(sid,[])
                    excerpt=selected_excerpt(m['raw'],spans,qtext,max(256,min(8000,limit-100)))
                    line=source_line({**m,'raw':excerpt});abbreviated=True
                parts.append(line);new_sources.append(sid)
        if not parts:
            # Do not duplicate an already delivered raw source just to fill result count.
            return False
        text="\n\n".join(parts).strip()
        image_cost=0;image_part=None
        if im and allow_image and im["asset"] not in assets and len(assets)<s.response_image_limit:
            a=im["metadata"];cost=image_tokens(a,s)
            if image_bytes+a["bytes"]<=30*1024*1024 and tokens(text)+cost+64<=remaining:
                image_cost=cost;image_part={"type":"image_url","image_url":{"url":asset_uri(a),"detail":"high"}}
        cost=tokens(text)+64+image_cost
        truncated=abbreviated
        if abbreviated:incomplete.add(c['id'])
        if cost>remaining:
            # Keep source boundaries explicit for a single oversized piece; never silently cut.
            if remaining<400:return False
            text=clip(text,max(1,remaining-128))+"\n[Additional source text omitted due to input budget; full source retained in memory.]"
            image_part=None;image_cost=0;cost=tokens(text)+64
            if cost>remaining:return False
            truncated=True;incomplete.add(c["id"])
        if not context_extra and not truncated:
            # Nearby context is optional; explicit dependencies above are not.
            neighbor_lines=[];neighbor_cost=0
            for sid in source_ids:
                m=messages.get(sid)
                if not m:continue
                for n in range(max(0,m["seq"]-2),m["seq"]+3):
                    nm=sessions.get(m["session"],{}).get(n)
                    if nm and nm["id"] not in rendered_sources and nm["id"] not in new_sources and nm["raw"]:
                        line=source_line(nm)
                        nc=tokens(line)+16
                        if nc<=1500 and neighbor_cost+nc<=max(0,remaining-cost)//8:
                            neighbor_lines.append(line);new_sources.append(nm["id"]);neighbor_cost+=nc
            if neighbor_lines:
                text+="\n\nAdjacent source context:\n"+"\n".join(neighbor_lines)
                cost=tokens(text)+64+image_cost
        content=[{"type":"text","text":text}]
        if image_part:
            content.append(image_part);assets.add(im["asset"]);image_bytes+=im["metadata"]["bytes"];pixels.add(c["id"])
        if im:
            rendered_occ.add(im["id"])
            if not truncated:rendered_captions[(im['asset'],im['caption'])]=im['id']
        # Scores reflect the final delivery order; do not expose incomparable channel scores.
        entry={"id":c["id"],"content":content if im else text,"score":1/(1+len(result))}
        ts=[messages[x]["timestamp"] for x in source_ids if x in messages and messages[x]["timestamp"] is not None]
        if ts:entry["created_at"]=iso(min(ts))
        result.append(entry);ids.add(c["id"])
        if not truncated:rendered_sources.update(new_sources)
        remaining-=cost
        return True
    for c in core:emit(c,c["id"] in chosen_ids)
    for c in extra:emit(c,False)
    reserved=int(remaining*.2);remaining-=reserved
    for c in rest:emit(c,False,True)
    return {"data":result},{"ids":ids,"complete_ids":ids-incomplete,"pixels":pixels,"remaining_tokens":remaining+reserved}
