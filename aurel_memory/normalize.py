"""Lenient syntax/shape repair, strict source attribution. Never retries the model."""
import json
import re
from json_repair import repair_json
from .util import as_list, strings, canonical, iso


def parse_output(text):
    if isinstance(text,(dict,list)):
        return text, "native"
    text = str(text or "").strip()
    if not text:
        return {}, "empty"
    try:
        return json.loads(text), "json"
    except ValueError:
        pass
    if not any(c in text for c in "{["):
        return {}, "unstructured"
    try:
        value = repair_json(text, return_objects=True)
        return value if isinstance(value,(dict,list)) else {}, "repaired"
    except Exception:
        return {}, "unstructured"


def source_ids(value, allowed, default=()):
    out = [v for v in strings(value) if v in allowed]
    return list(dict.fromkeys(out or [v for v in default if v in allowed]))


def caption_text(value):
    if isinstance(value,str):return value.strip()
    if not isinstance(value,dict):return ''
    text='\n'.join(strings(value.get('caption',value.get('description',value.get('image_description','')))))
    ocr=[x for x in strings(value.get('ocr')) if x not in text]
    if ocr:text+='\nVisible text: '+'; '.join(ocr)
    uncertainty=strings(value.get('uncertainties'))
    if uncertainty:text+='\nUncertain details: '+'; '.join(uncertainty)
    return text.strip()


def alias_input(payload):
    """Short prompt IDs reduce copying errors; persistent source identifiers stay unchanged."""
    forward={}
    def add(value,prefix):
        if value and value not in forward:forward[value]=prefix+str(1+sum(v.startswith(prefix) for v in forward.values()))
    for m in payload['earlier_context']+payload['new_messages']:
        add(m['id'],'M')
        for im in m.get('images',[]):add(im['id'],'I')
    for sc in payload.get('scene_candidates',[]):add(sc['id'],'S')
    if payload.get('open_cell'):add(payload['open_cell']['id'],'C')
    def walk(value,key=''):
        if isinstance(value,dict):return {k:walk(v,k) for k,v in value.items()}
        if isinstance(value,list):return [walk(v,key) for v in value]
        if isinstance(value,str):
            if key in {'id','sources','source_ids','scene','continue_cell'}:return forward.get(value,value)
            if key=='text':
                for old,new in forward.items():
                    if new.startswith('I'):value=value.replace('Image '+old+' Caption:','Image '+new+' Caption:')
        return value
    return walk(payload),{v.upper():k for k,v in forward.items()}


def restore_ids(value,aliases,key=''):
    if isinstance(value,dict):return {k:restore_ids(v,aliases,k) for k,v in value.items()}
    if isinstance(value,list):
        out=[restore_ids(v,aliases,key) for v in value]
        return [x for v in out for x in (v if isinstance(v,list) else [v])] if key in {'sources','source_ids'} else out
    if isinstance(value,str) and key in {'id','sources','source_ids','scene','continue_cell'}:
        if value.strip().upper() in aliases:return aliases[value.strip().upper()]
        if key in {'sources','source_ids'}:
            found=[aliases[x.upper()] for x in re.findall(r'\bM\d+\b',value,re.I) if x.upper() in aliases]
            if found:return found
    return value


def grounded_dates(text,evidence,allowed_years=()):
    def chinese(m):return m[0] if m[1] in evidence or m[1] in allowed_years else ''
    def numeric(m):return m[0] if m[1] in evidence or m[1] in allowed_years else m[2]+'-'+m[3]+' [year unspecified]'
    text=re.sub(r'(?<!\d)((?:19|20|21)\d{2})年(?=\d{1,2}月)',chinese,text)
    return re.sub(r'(?<!\d)((?:19|20|21)\d{2})-(\d{2})-(\d{2})\b',numeric,text)


def normalize(value, messages, occurrences):
    allowed = {m["id"] for m in messages}
    new = [m["id"] for m in messages if not m.get("history")]
    if isinstance(value,list):
        value = {"cells":value}
    if not isinstance(value,dict):
        value = {}
    cells, covered = [], set()
    for item in as_list(value.get("cells", value.get("records", value.get("events")))):
        if isinstance(item,str):
            item = {"summary":item}
        if not isinstance(item,dict):
            continue
        refs = source_ids(item.get("sources",item.get("source_ids")),allowed, new if len(new)==1 else ())
        # A valid new fact can repair an omitted parent-cell source; never discard that evidence.
        child_refs=[s for f in as_list(item.get('facts')) if isinstance(f,dict)
                    for s in source_ids(f.get('sources',f.get('source_ids')),allowed)]
        refs=list(dict.fromkeys(refs+child_refs))
        if not refs or not set(refs).intersection(new):
            continue
        facts = []
        for f in as_list(item.get("facts")):
            if isinstance(f,str):
                f = {"text":f}
            if not isinstance(f,dict):
                continue
            text = str(f.get("text",f.get("fact","")) or "").strip()
            if not text:
                text = " ".join(str(f.get(k,"")) for k in ("subject","predicate","object")).strip()
            if not text:
                continue
            refs_f = source_ids(f.get("sources",f.get("source_ids")),allowed,refs)
            clean = {k:str(f.get(k,"") or "") for k in ("subject","predicate","object","quote",
                         "kind","assertion","polarity","valid_from","valid_to","time_basis","action","facet","scope","time_semantics","state_slot")}
            clean.update(text=text,sources=refs_f)
            evidence='\n'.join(str(m.get('text',''))+'\n'+'\n'.join(str(x.get('caption','')) for x in m.get('images',[])) for m in messages if m['id'] in refs_f)
            basis=clean['time_basis']
            anchors=[iso(m.get('timestamp')) for m in messages if m['id'] in refs_f and m.get('timestamp') is not None]
            for field in ('valid_from','valid_to'):
                time_value=clean[field]
                year=re.match(r'^(\d{4})-',time_value)
                # Unsupported time normalization is discarded; verbatim temporal evidence remains.
                if time_value and (not basis or canonical(basis) not in canonical(evidence)):
                    clean[field]=''
                elif year and year[1] not in evidence:
                    relative=bool(re.search(r'今天|昨天|前天|上周|本周|去年|今年|明年|today|yesterday|last |this |next |ago',basis,re.I))
                    anchor_years={int(a[:4]) for a in anchors}
                    if not relative or not any(abs(int(year[1])-y)<=1 for y in anchor_years):clean[field]=''
            valid_years={clean[k][:4] for k in ('valid_from','valid_to') if re.match(r'^\d{4}-',clean[k])}
            for field in ('text','quote','time_basis'):clean[field]=grounded_dates(clean[field],evidence,valid_years)
            clean['kind']={'状态':'state','偏好':'preference','行为':'behavior','关系':'relation','事实':'fact'}.get(clean['kind'],clean['kind'].lower())
            clean["kind"] = clean["kind"] if clean["kind"] in {"fact","state","preference","behavior","relation"} else "fact"
            clean["assertion"] = clean["assertion"] if clean["assertion"] in {"actual","plan","hypothetical","reported","inferred"} else "reported"
            # Accept semantic fields even when the model uses the generic kind='fact'.
            if clean['kind']=='fact' and canonical(clean['time_semantics']) in {'currentstate','historicalstate','当前状态','历史状态'} and clean['subject'] and clean['predicate'] and clean['object']:
                clean['kind']='state'
            predicate=canonical(clean['predicate'])
            if clean['kind']=='state' and clean['scope'] and predicate in {'拥有','持有','更换','换成','退掉','退还','owns','has','replaced','replace','returned'}:
                clean['state_slot']='current_possession'
                if predicate in {'更换','换成','replaced','replace'}:clean['action']='replace'
                elif predicate in {'退掉','退还','returned'}:clean['action']='retract';clean['polarity']='negative'
            # No automatic current-state/profile mutation based on assistant-only evidence.
            roles = {m["role"] for m in messages if m["id"] in refs_f}
            if clean["kind"] in {"state","preference","behavior"} and "user" not in roles:
                clean["kind"] = "fact"
                clean["assertion"] = "reported"
                clean['state_slot']=''
            facts.append(clean)
        source_text='\n'.join(str(m.get('text',''))+'\n'+'\n'.join(str(x.get('caption','')) for x in m.get('images',[])) for m in messages if m['id'] in refs)
        resolved_years={f[k][:4] for f in facts for k in ('valid_from','valid_to') if re.match(r'^\d{4}-',f[k])}
        cells.append({"title":grounded_dates(str(item.get("title",item.get("topic","")) or ""),source_text,resolved_years),
                      "summary":grounded_dates(str(item.get("summary",item.get("description","")) or ""),source_text,resolved_years),
                      "sources":refs,"entities":strings(item.get("entities")),"facts":facts,
                      "scene":str(item.get("scene","") or ""),
                      "continue_cell":str(item.get("continue_cell","") or ""),
                      "scope":"topic" if item.get("scope")=="topic" else "event"})
        covered.update(refs)
    # Omitted / malformed records keep immutable raw evidence and a searchable source cell.
    for m in messages:
        if m["id"] in new and m["id"] not in covered:
            cells.append({"title":"source memory","summary":"","sources":[m["id"]],
                          "entities":[],"facts":[],"scene":"","scope":"event","fallback":True})
            covered.add(m["id"])
    occ_ids = {o["id"] for o in occurrences}
    images = []
    for item in as_list(value.get("images")):
        if not isinstance(item,dict) or str(item.get("id")) not in occ_ids:
            continue
        refs = source_ids(item.get("sources"),allowed)
        images.append({"id":str(item["id"]),"sources":refs,
                       "context":str(item.get("context","") or "") if refs else "",
                       "entities":strings(item.get("entities")),"ocr":strings(item.get("ocr"))})
    return {"cells":cells,"images":images}


def decision(value, visible):
    if not isinstance(value,dict):
        value = {}
    d = value.get("decision",value.get("is_sufficient"))
    if isinstance(d,bool):
        d = "sufficient" if d else "insufficient"
    d = str(d or "unknown").lower().strip()
    d = {"true":"sufficient","false":"insufficient","yes":"sufficient","no":"insufficient"}.get(d,d)
    if d not in {"sufficient","insufficient","unknown"}:
        d = "unknown"
    support = source_ids(value.get("support_ids",value.get("evidence_ids")),visible)
    if d=="sufficient" and not support:
        d="unknown"
    return {"decision":d,"support_ids":support,"missing":strings(value.get("missing",value.get("missing_information"))),
            "query":str(value.get("query",value.get("next_query","")) or ""),
            "need_pixels":source_ids(value.get("need_pixels"),visible)}
