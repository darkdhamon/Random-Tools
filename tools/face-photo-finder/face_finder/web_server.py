from __future__ import annotations

import io
import json
import secrets
import subprocess
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps

from .catalog import FaceCatalog, default_catalog_path

HOST = "127.0.0.1"
PORT = 8765
APP_ROOT = Path(__file__).resolve().parents[1]


PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Face Photo Finder Gallery</title><style>
:root{color-scheme:dark;background:#111;color:#eee;font:15px system-ui}body{margin:0}header{position:sticky;top:0;z-index:2;background:#181818;padding:12px;display:flex;flex-wrap:wrap;gap:10px;align-items:center;box-shadow:0 2px 8px #000}input,select,textarea,button{background:#292929;color:#eee;border:1px solid #555;border-radius:6px;padding:8px}button{cursor:pointer}.timeline{position:relative;padding:18px 18px 18px 86px}.timeline:before{content:'';position:absolute;left:48px;top:0;bottom:0;width:3px;background:#3a6f78}.year-group{position:relative;margin-bottom:30px}.year-group h2{position:sticky;top:68px;z-index:1;margin:0 0 12px -72px;width:58px;text-align:center;background:#164d57;border:2px solid #59d7e6;border-radius:18px;padding:6px;font-size:16px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}.card{background:#202020;border-radius:9px;overflow:hidden;cursor:pointer;box-shadow:0 3px 12px #0008}.card img{width:100%;height:180px;object-fit:cover}.info{padding:9px}.muted{color:#aaa;font-size:12px}.stars{color:#ffd65a}.modal{position:fixed;inset:0;background:#000d;display:none;z-index:5;padding:3vh}.panel{max-width:1100px;height:94vh;margin:auto;background:#202020;border-radius:10px;display:grid;grid-template-columns:2fr 1fr;overflow:hidden}.viewer{background:#090909;display:flex;align-items:center;justify-content:center}.viewer img{max-width:100%;max-height:94vh}.editor{padding:16px;overflow:auto}.editor input,.editor textarea,.editor select{width:100%;box-sizing:border-box;margin:5px 0 12px}.faces div{padding:7px;background:#292929;margin:5px 0;border-radius:5px}.close{float:right}.save-state{min-height:20px;color:#69dfbd;margin:8px 0}@media(max-width:750px){.panel{grid-template-columns:1fr}.viewer{height:48vh}.viewer img{max-height:48vh}.timeline{padding-left:58px}.timeline:before{left:28px}.year-group h2{margin-left:-50px}}
</style></head><body><header><strong>Photo Timeline</strong><input id=q placeholder="Search paths, titles, tags"><select id=person><option value="">All people</option></select><input id=year type=number placeholder="Year" min=1900 style="width:90px"><label><input id=hideNsfw type=checkbox checked onchange="load()"> Hide NSFW</label><label><input id=hideScreenshots type=checkbox checked onchange="load()"> Hide screenshots</label><label><input id=hideDocuments type=checkbox checked onchange="load()"> Hide documents</label><select id=contentFilter onchange="load()"><option value=normal>Normal browsing</option><option value=nsfw>NSFW only</option><option value=all>Include NSFW</option><option value=screenshot>Screenshots only</option><option value=document>Documents only</option></select><button onclick="load()">Search</button><button onclick="post('/api/scan',{})">Scan now</button><button onclick="post('/api/open-desktop',{})">Desktop app</button></header><main id=timeline class=timeline></main>
<div style="text-align:center;padding:15px"><button id=more onclick="load(false)">Load more</button></div><div id=modal class=modal><div class=panel><div class=viewer><img id=full></div><div class=editor><button class=close onclick="modal.style.display='none'">Close</button><h2 id=name></h2><label>Title<input id=title></label><label>Description<textarea id=description rows=4></textarea></label><label>Tags<input id=tags placeholder="family, vacation"></label><label>Rating<select id=rating><option value=0>Unrated</option><option value=1>★</option><option value=2>★★</option><option value=3>★★★</option><option value=4>★★★★</option><option value=5>★★★★★</option></select></label><label>Capture year<input id=captureYear type=number min=1900></label><label>Content type<select id=mediaKindOverride><option value="">Automatic</option><option value=photo>Photo</option><option value=screenshot>Screenshot</option><option value=document>Document</option></select></label><label>NSFW classification<select id=nsfwOverride><option value="">Automatic</option><option value=0>Mark safe</option><option value=1>Mark NSFW</option></select></label><div id=nsfwScore class=muted></div><div id=saveState class=save-state></div><h3>Detected faces</h3><div id=faces class=faces></div></div></div></div>
<script>const token='__TOKEN__';let current=null,identities=[],offset=0,saveTimer=null;async function api(u,o){let r=await fetch(u,o);if(!r.ok)throw Error(await r.text());return r.json()}async function post(u,d){let x=await api(u,{method:'POST',headers:{'Content-Type':'application/json','X-Gallery-Token':token},body:JSON.stringify(d)});if(x.message)alert(x.message);return x}async function people(){identities=await api('/api/identities');for(let p of identities){let o=document.createElement('option');o.value=p.id;o.textContent=p.name;person.append(o)}}function yearGrid(value){let label=value||'Unknown date',id='year-'+String(label).replace(/\W/g,'-'),grid=document.getElementById(id);if(!grid){let section=document.createElement('section');section.className='year-group';section.innerHTML=`<h2>${esc(String(label))}</h2><div id="${id}" class="grid"></div>`;timeline.append(section);grid=document.getElementById(id)}return grid}async function load(reset=true){if(reset){offset=0;timeline.innerHTML=''}let filter=contentFilter.value==='nsfw'?'nsfw':(contentFilter.value==='all'||!hideNsfw.checked?'all':'safe');let media=['screenshot','document'].includes(contentFilter.value)?contentFilter.value:'';let excluded=[];if(!media&&hideScreenshots.checked)excluded.push('screenshot');if(!media&&hideDocuments.checked)excluded.push('document');let p=new URLSearchParams({q:q.value,identity_id:person.value,year:year.value,nsfw:filter,media_kind:media,exclude_kinds:excluded.join(','),limit:100,offset});let a=await api('/api/photos?'+p);offset+=a.length;more.style.display=a.length<100?'none':'';for(let x of a){let c=document.createElement('article');c.className='card';c.innerHTML=`<img loading=lazy src="/media?id=${x.id}&thumb=1"><div class=info><b>${esc(x.title||x.name)}</b><div class=muted>${x.identified_count}/${x.face_count} faces · ${x.media_kind}${x.is_nsfw?' · NSFW':''}</div><div class=stars>${'★'.repeat(x.rating)}</div></div>`;c.onclick=()=>openPhoto(x.id);yearGrid(x.year).append(c)}}async function openPhoto(id){clearTimeout(saveTimer);current=await api('/api/photo?id='+id);full.src='/media?id='+id;name.textContent=current.name;title.value=current.title||'';description.value=current.description||'';tags.value=current.tags||'';rating.value=current.rating||0;captureYear.value=current.year||'';mediaKindOverride.value=current.media_kind_override==null?'':current.media_kind_override;nsfwOverride.value=current.nsfw_override==null?'':current.nsfw_override;saveState.textContent='';nsfwScore.textContent=`Detected type: ${current.media_kind}. `+(current.nsfw_score==null?'Not NSFW-classified yet':`Automatic NSFW confidence: ${(current.nsfw_score*100).toFixed(1)}%`);faces.innerHTML=(current.faces||[]).map(f=>`<div>${esc(f.name||(f.unknown?'Unknown person':'Unprocessed'))}${f.art?' · artwork':''}${f.estimated_age!=null?' · age '+f.estimated_age:''}<br><select onchange="assignFace(${f.id},this.value)"><option value="">Reassign…</option>${identities.map(p=>`<option value="${p.id}">${esc(p.name)}</option>`).join('')}</select></div>`).join('');modal.style.display='block'}async function assignFace(face,id){if(!id)return;await post('/api/face',{face_id:face,identity_id:+id});saveState.textContent='Face assignment saved';await openPhoto(current.id)}function queueSave(immediate=false){if(!current)return;clearTimeout(saveTimer);saveState.textContent='Saving…';saveTimer=setTimeout(save,immediate?0:700)}async function save(){if(!current)return;try{await post('/api/photo',{id:current.id,title:title.value,description:description.value,tags:tags.value,rating:+rating.value,capture_year:captureYear.value?+captureYear.value:null,nsfw_override:nsfwOverride.value===''?null:+nsfwOverride.value,media_kind_override:mediaKindOverride.value||null});saveState.textContent='Saved automatically';}catch(e){saveState.textContent='Save failed: '+e.message}}function esc(s){let d=document.createElement('div');d.textContent=s;return d.innerHTML}for(let id of ['title','description','tags','captureYear'])document.getElementById(id).addEventListener('input',()=>queueSave(false));for(let id of ['rating','mediaKindOverride','nsfwOverride'])document.getElementById(id).addEventListener('change',()=>queueSave(true));people().then(load);</script></body></html>'''


class GalleryHandler(BaseHTTPRequestHandler):
    token = secrets.token_urlsafe(24)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, value: object, status: int = 200) -> None:
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _catalog(self) -> FaceCatalog:
        return FaceCatalog(default_catalog_path())

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            data = PAGE.replace("__TOKEN__", self.token).encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
        catalog = self._catalog()
        try:
            if parsed.path == "/api/photos":
                identity = query.get("identity_id", [""])[0]
                year = query.get("year", [""])[0]
                self._json(catalog.gallery_photos(
                    query.get("q", [""])[0], int(identity) if identity else None,
                    int(year) if year else None, int(query.get("limit", ["100"])[0]),
                    int(query.get("offset", ["0"])[0]), query.get("nsfw", ["all"])[0],
                    tuple(filter(None, query.get("exclude_kinds", [""])[0].split(","))),
                    query.get("media_kind", [""])[0] or None,
                ))
            elif parsed.path == "/api/photo":
                photo = catalog.gallery_photo(int(query["id"][0]))
                self._json(photo if photo else {"error": "not found"}, 200 if photo else 404)
            elif parsed.path == "/api/identities":
                self._json([{"id": x.identity_id, "name": x.name, "birth_year": x.birth_year} for x in catalog.identities()])
            elif parsed.path == "/media":
                path = catalog.image_path(int(query["id"][0]))
                if path is None: self.send_error(404); return
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    if query.get("thumb") == ["1"]: image.thumbnail((500, 360), Image.Resampling.LANCZOS)
                    output = io.BytesIO(); image.save(output, "JPEG", quality=88); data = output.getvalue()
                self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "private, max-age=3600")
                self.end_headers(); self.wfile.write(data)
            else: self.send_error(404)
        except (ValueError, KeyError) as exc: self._json({"error": str(exc)}, 400)
        finally: catalog.close()

    def do_POST(self) -> None:
        if self.headers.get("X-Gallery-Token") != self.token:
            self._json({"error": "invalid request token"}, 403); return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 1_000_000)
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/photo":
                catalog = self._catalog()
                try:
                    catalog.update_gallery_metadata(int(body["id"]), str(body.get("title", "")), str(body.get("description", "")), str(body.get("tags", "")), int(body.get("rating", 0)), body.get("capture_year"), body.get("nsfw_override"), body.get("media_kind_override"))
                finally: catalog.close()
                self._json({"ok": True})
            elif self.path == "/api/face":
                catalog = self._catalog()
                try: catalog.assign_face(int(body["face_id"]), int(body["identity_id"]))
                finally: catalog.close()
                self._json({"ok": True})
            elif self.path == "/api/scan":
                subprocess.run(["schtasks", "/Run", "/TN", "Face Photo Finder Background Catalog"], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
                self._json({"ok": True, "message": "Background catalog scan started."})
            elif self.path == "/api/open-desktop":
                subprocess.Popen([sys.executable, str(APP_ROOT / "run.py")], cwd=APP_ROOT, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
                self._json({"ok": True})
            else: self._json({"error": "not found"}, 404)
        except Exception as exc: self._json({"error": str(exc)}, 400)


def main(open_browser: bool = False) -> None:
    server = ThreadingHTTPServer((HOST, PORT), GalleryHandler)
    if open_browser: webbrowser.open(f"http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main("--open" in sys.argv)
