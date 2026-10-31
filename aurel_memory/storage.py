import json
import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager
import numpy as np
from .util import dumps, ident, terms, canonical, normvec


class Conflict(Exception):pass


class Store:
    def __init__(self,root):
        self.root=Path(root)
        self.path=self.root/"memory.sqlite3"
        self.root.mkdir(parents=True,exist_ok=True)
        with self.connect() as c:
            c.executescript('''
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS requests(user TEXT,rid TEXT,hash TEXT,response TEXT,PRIMARY KEY(user,rid));
            CREATE TABLE IF NOT EXISTS versions(user TEXT PRIMARY KEY,version INTEGER);
            CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY,user TEXT,session TEXT,seq INTEGER,role TEXT,
              ts INTEGER,content TEXT,raw TEXT,created INTEGER);
            CREATE INDEX IF NOT EXISTS msg_session ON messages(user,session,seq);
            CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY,metadata TEXT);
            CREATE TABLE IF NOT EXISTS images(id TEXT PRIMARY KEY,user TEXT,message TEXT,asset TEXT,caption TEXT,
              context TEXT,sources TEXT,entities TEXT,caption_source TEXT,vision BLOB);
            CREATE INDEX IF NOT EXISTS image_user ON images(user);
            CREATE INDEX IF NOT EXISTS image_message ON images(message);
            CREATE TABLE IF NOT EXISTS scenes(id TEXT PRIMARY KEY,user TEXT,title TEXT,summary TEXT,scope TEXT,
              entities TEXT,centroid BLOB,count INTEGER,updated INTEGER);
            CREATE INDEX IF NOT EXISTS scene_user ON scenes(user);
            CREATE TABLE IF NOT EXISTS cells(id TEXT PRIMARY KEY,user TEXT,session TEXT,scene TEXT,body TEXT,sources TEXT,entities TEXT);
            CREATE INDEX IF NOT EXISTS cell_scene ON cells(user,scene);
            CREATE INDEX IF NOT EXISTS cell_session ON cells(user,session);
            CREATE TABLE IF NOT EXISTS cell_revisions(id TEXT,revision INTEGER,body TEXT,PRIMARY KEY(id,revision));
            CREATE TABLE IF NOT EXISTS docs(id TEXT PRIMARY KEY,user TEXT,kind TEXT,text TEXT,parent TEXT,
              sources TEXT,entities TEXT,meta TEXT,vec BLOB,sigvec BLOB);
            CREATE INDEX IF NOT EXISTS doc_user ON docs(user);
            CREATE INDEX IF NOT EXISTS doc_parent ON docs(parent);
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(tokens,docid UNINDEXED,user UNINDEXED,tokenize='unicode61');
            CREATE TABLE IF NOT EXISTS fts_rows(docid TEXT PRIMARY KEY,fts_rowid INTEGER);
            CREATE TABLE IF NOT EXISTS states(id TEXT PRIMARY KEY,user TEXT,subject TEXT,predicate TEXT,value TEXT,
              valid_from TEXT,valid_to TEXT,observed INTEGER,action TEXT,sources TEXT,assertion TEXT);
            CREATE INDEX IF NOT EXISTS state_lookup ON states(user,subject,predicate,observed);
            CREATE TABLE IF NOT EXISTS preferences(id TEXT PRIMARY KEY,user TEXT,facet TEXT,claim TEXT,sources TEXT,event TEXT,kind TEXT);
            CREATE INDEX IF NOT EXISTS preference_lookup ON preferences(user,facet);
            ''')
            c.execute('INSERT OR IGNORE INTO fts_rows SELECT docid,rowid FROM docs_fts')

    @contextmanager
    def connect(self):
        c=sqlite3.connect(self.path,timeout=30)
        c.row_factory=sqlite3.Row
        c.execute("PRAGMA busy_timeout=30000")
        try:
            yield c
            c.commit()
        except BaseException:
            c.rollback()
            raise
        finally:c.close()

    def replay(self,user,rid,payload_hash):
        with self.connect() as c:
            row=c.execute("SELECT hash,response FROM requests WHERE user=? AND rid=?",(user,rid)).fetchone()
        if not row:return None
        if row["hash"]!=payload_hash:raise Conflict("request_id reused with different payload")
        return json.loads(row["response"])

    def tail(self,user,session,n=6):
        with self.connect() as c:
            rows=c.execute("SELECT * FROM messages WHERE user=? AND session=? ORDER BY seq DESC LIMIT ?",(user,session,n)).fetchall()
        return [self.message(r) for r in reversed(rows)]

    def open_cell(self,user,session):
        with self.connect() as c:
            row=c.execute("SELECT body FROM cells WHERE user=? AND session=? AND json_extract(body,'$.status')='provisional' ORDER BY rowid DESC LIMIT 1",(user,session)).fetchone()
            if not row:return None
            cell=json.loads(row[0])
            vec=c.execute("SELECT vec FROM docs WHERE user=? AND parent=? AND kind='cell'",(user,cell['id'])).fetchone()
        if vec:cell['_old_vec']=np.frombuffer(vec[0],dtype=np.float32).copy()
        return cell

    def scene_summaries(self,user,scene,exclude=()):
        with self.connect() as c:
            rows=c.execute("SELECT id,body FROM cells WHERE user=? AND scene=? ORDER BY rowid DESC LIMIT 12",(user,scene)).fetchall()
        return [json.loads(r['body']) for r in rows if r['id'] not in exclude]

    @staticmethod
    def message(r):
        d=dict(r)
        d["timestamp"]=d.pop("ts")
        d["content"]=json.loads(d["content"])
        return d

    def get_messages(self,user,ids):
        if not ids:return {}
        with self.connect() as c:
            rows=c.execute("SELECT * FROM messages WHERE user=? AND id IN ("+",".join("?" for _ in ids)+")",[user,*ids]).fetchall()
        return {r["id"]:self.message(r) for r in rows}

    def neighbors(self,user,session,seq,radius=2):
        with self.connect() as c:
            rows=c.execute("SELECT * FROM messages WHERE user=? AND session=? AND seq BETWEEN ? AND ? ORDER BY seq",
                           (user,session,max(0,seq-radius),seq+radius)).fetchall()
        return [self.message(r) for r in rows]

    def scene_candidates(self,user,limit=256):
        with self.connect() as c:
            rows=c.execute("SELECT * FROM scenes WHERE user=? ORDER BY updated DESC LIMIT ?",(user,limit)).fetchall()
        return [{**dict(r),"entities":json.loads(r["entities"]),"centroid":np.frombuffer(r["centroid"],dtype=np.float32)} for r in rows]

    def image_records(self,user):
        with self.connect() as c:
            rows=c.execute("SELECT i.*,a.metadata FROM images i JOIN assets a ON i.asset=a.id WHERE i.user=?",(user,)).fetchall()
        out={}
        for r in rows:
            d=dict(r)
            for field in ("sources","entities","metadata"):d[field]=json.loads(d[field])
            d["vision"]=np.frombuffer(d["vision"],dtype=np.float32) if d["vision"] else None
            out[d["id"]]=d
        return out

    def images_for_messages(self,user,ids):
        if not ids:return []
        with self.connect() as c:
            rows=c.execute("SELECT i.*,a.metadata FROM images i JOIN assets a ON i.asset=a.id WHERE i.user=? AND i.message IN ("+",".join("?" for _ in ids)+")",[user,*ids]).fetchall()
        result=[]
        for r in rows:
            d=dict(r)
            for field in ("sources","entities","metadata"):d[field]=json.loads(d[field])
            d["vision"]=np.frombuffer(d["vision"],dtype=np.float32).copy() if d["vision"] else None
            result.append(d)
        return result

    def snapshot(self,user):
        # One SQLite read transaction: all candidates belong to the same committed version.
        with self.connect() as c:
            c.execute("BEGIN")
            vr=c.execute("SELECT version FROM versions WHERE user=?",(user,)).fetchone()
            rows=c.execute("SELECT * FROM docs WHERE user=?",(user,)).fetchall()
            cells={r["id"]:dict(r) for r in c.execute("SELECT * FROM cells WHERE user=?",(user,))}
            image_rows=c.execute("SELECT i.*,a.metadata FROM images i JOIN assets a ON i.asset=a.id WHERE i.user=?",(user,)).fetchall()
            msg_rows=c.execute("SELECT * FROM messages WHERE user=? ORDER BY session,seq",(user,)).fetchall()
        docs=[]
        for r in rows:
            d=dict(r)
            for f in ("sources","entities","meta"):d[f]=json.loads(d[f])
            d["vec"]=np.frombuffer(d["vec"],dtype=np.float32)
            d["sigvec"]=np.frombuffer(d["sigvec"],dtype=np.float32) if d["sigvec"] else None
            docs.append(d)
        images={}
        for r in image_rows:
            d=dict(r)
            for f in ("sources","entities","metadata"):d[f]=json.loads(d[f])
            d["vision"]=np.frombuffer(d["vision"],dtype=np.float32) if d["vision"] else None
            images[d["id"]]=d
        return {"version":vr[0] if vr else 0,"docs":docs,"cells":cells,"images":images,
                "messages":{r["id"]:self.message(r) for r in msg_rows}}

    def version(self,user):
        with self.connect() as c:
            r=c.execute("SELECT version FROM versions WHERE user=?",(user,)).fetchone()
        return r[0] if r else 0

    def publish(self,user,rid,payload_hash,messages,assets,images,cells,docs,scenes,response):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            def remove_fts(did):
                row=c.execute('SELECT fts_rowid FROM fts_rows WHERE docid=?',(did,)).fetchone()
                if row:c.execute('DELETE FROM docs_fts WHERE rowid=?',(row[0],))
                c.execute('DELETE FROM fts_rows WHERE docid=?',(did,))
            old=c.execute("SELECT hash,response FROM requests WHERE user=? AND rid=?",(user,rid)).fetchone()
            if old:
                if old["hash"]!=payload_hash:raise Conflict("request_id reused")
                return json.loads(old["response"])
            for m in messages:
                c.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?)",(m["id"],user,m["session"],m["seq"],m["role"],m["timestamp"],dumps(m["content"]),m["raw"],m["created"]))
            for a in assets:
                c.execute("INSERT OR IGNORE INTO assets VALUES (?,?)",(a["id"],dumps(a)))
            for im in images:
                # Superseded context chunks must not survive a shorter replacement.
                oldids=[r[0] for r in c.execute("SELECT id FROM docs WHERE user=? AND parent=? AND kind='image_context'",(user,im['id']))]
                for did in oldids:
                    remove_fts(did)
                c.execute("DELETE FROM docs WHERE user=? AND parent=? AND kind='image_context'",(user,im['id']))
                c.execute("INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?,?,?,?)",(im["id"],user,im["message"],im["asset"],im["caption"],im["context"],dumps(im["sources"]),dumps(im["entities"]),im["caption_source"],im["vision"].tobytes() if im.get("vision") is not None else None))
            for sc in scenes:
                c.execute("INSERT OR REPLACE INTO scenes VALUES (?,?,?,?,?,?,?,?,?)",(sc["id"],user,sc["title"],sc["summary"],sc["scope"],dumps(sc["entities"]),sc["centroid"].astype(np.float32).tobytes(),sc["count"],sc["updated"]))
            if messages:
                # The current last event remains open. Earlier events are finalized without LLM calls.
                active={x['id'] for x in cells if x['status']=='provisional'}
                for r in c.execute("SELECT id,body FROM cells WHERE user=? AND session=? AND json_extract(body,'$.status')='provisional'",(user,messages[0]['session'])).fetchall():
                    if r['id'] not in active:
                        oldcell=json.loads(r['body']);oldcell['status']='finalized'
                        c.execute("UPDATE cells SET body=? WHERE id=?",(dumps(oldcell),r['id']))
            for cell in cells:
                if cell['revision']>1:
                    oldids=[r[0] for r in c.execute("SELECT id FROM docs WHERE user=? AND parent=? AND kind IN ('cell','fact')",(user,cell['id']))]
                    for did in oldids:remove_fts(did)
                    c.execute("DELETE FROM docs WHERE user=? AND parent=? AND kind IN ('cell','fact')",(user,cell['id']))
                c.execute("INSERT OR REPLACE INTO cells VALUES (?,?,?,?,?,?,?)",(cell["id"],user,cell["session"],cell["scene"],dumps(cell),dumps(cell["sources"]),dumps(cell["entities"])))
                c.execute("INSERT OR IGNORE INTO cell_revisions VALUES (?,?,?)",(cell['id'],cell['revision'],dumps(cell)))
                for f in cell["facts"]:
                    fid=f["id"]
                    if f["kind"]=="state" and f["subject"] and f["predicate"]:
                        slot=canonical(f.get('state_slot') or f['predicate'])+'::'+canonical(f.get('scope',''))
                        c.execute("INSERT OR IGNORE INTO states VALUES (?,?,?,?,?,?,?,?,?,?,?)",(fid,user,canonical(f["subject"]),slot,f["object"],f["valid_from"],f["valid_to"],f.get('observed',cell["observed"]),f["action"],dumps(f["sources"]),f["assertion"]))
                    if f["kind"] in {"preference","behavior"}:
                        c.execute("INSERT OR IGNORE INTO preferences VALUES (?,?,?,?,?,?,?)",(fid,user,f["facet"] or f["predicate"],f["text"],dumps(f["sources"]),cell["id"],f["kind"]))
            for d in docs:
                # A scene summary is replaceable derived state; source/cell docs remain immutable.
                remove_fts(d['id'])
                c.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?,?,?,?)",(d["id"],user,d["kind"],d["text"],d["parent"],dumps(d["sources"]),dumps(d["entities"]),dumps(d["meta"]),d["vec"].astype(np.float32).tobytes(),d["sigvec"].tobytes() if d.get("sigvec") is not None else None))
                rowid=c.execute("INSERT INTO docs_fts(tokens,docid,user) VALUES (?,?,?)",(" ".join(terms(d["text"])),d["id"],user)).lastrowid
                c.execute('INSERT INTO fts_rows VALUES (?,?)',(d['id'],rowid))
            c.execute("INSERT INTO versions VALUES (?,1) ON CONFLICT(user) DO UPDATE SET version=version+1",(user,))
            c.execute("INSERT INTO requests VALUES (?,?,?,?)",(user,rid,payload_hash,dumps(response)))
        return response

    def stats(self):
        with self.connect() as c:
            return {table:c.execute("SELECT COUNT(*) FROM "+table).fetchone()[0] for table in ("requests","messages","images","cells","scenes","docs","states","preferences")}
