from __future__ import annotations

import io
import json
import os
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
PICTURES_ROOT = Path(os.environ.get("OneDrive", Path.home() / "OneDrive")) / "Pictures"
GENERAL_ARCHIVE = PICTURES_ROOT / "Hidden Pictures" / "GeneralArchive.zip"


PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Face Photo Finder Gallery</title><style>
:root{color-scheme:dark;background:#111;color:#eee;font:15px system-ui}body{margin:0}header{position:sticky;top:0;z-index:2;background:#181818;padding:12px;display:flex;flex-wrap:wrap;gap:10px;align-items:center;box-shadow:0 2px 8px #000}input,select,textarea,button{background:#292929;color:#eee;border:1px solid #555;border-radius:6px;padding:8px}button{cursor:pointer}.timeline{position:relative;padding:18px 18px 18px 86px}.timeline:before{content:'';position:absolute;left:48px;top:0;bottom:0;width:3px;background:#3a6f78}.year-group{position:relative;margin-bottom:30px}.year-group h2{position:sticky;top:68px;z-index:1;margin:0 0 12px -72px;width:58px;text-align:center;background:#164d57;border:2px solid #59d7e6;border-radius:18px;padding:6px;font-size:16px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}.card{background:#202020;border-radius:9px;overflow:hidden;cursor:pointer;box-shadow:0 3px 12px #0008}.card img{width:100%;height:180px;object-fit:cover}.info{padding:9px}.muted{color:#aaa;font-size:12px}.stars{color:#ffd65a}.modal{position:fixed;inset:0;background:#000d;display:none;z-index:5;padding:3vh}.panel{max-width:1100px;height:94vh;margin:auto;background:#202020;border-radius:10px;display:grid;grid-template-columns:2fr 1fr;overflow:hidden}.viewer{background:#090909;display:flex;align-items:center;justify-content:center}.viewer img{max-width:100%;max-height:94vh}.editor{padding:16px;overflow:auto}.editor input,.editor textarea,.editor select{width:100%;box-sizing:border-box;margin:5px 0 12px}.faces div{padding:7px;background:#292929;margin:5px 0;border-radius:5px}.close{float:right}.save-state{min-height:20px;color:#69dfbd;margin:8px 0}.map{position:relative;height:150px;border:1px solid #49656b;border-radius:8px;margin:6px 0 8px;overflow:hidden;background-color:#18323a;background-image:linear-gradient(#ffffff14 1px,transparent 1px),linear-gradient(90deg,#ffffff14 1px,transparent 1px);background-size:25% 50%}.map:after{content:'Offline world map';position:absolute;right:7px;bottom:5px;color:#9eb7bc;font-size:11px}.pin{display:none;position:absolute;width:14px;height:14px;background:#ff4e67;border:3px solid white;border-radius:50% 50% 50% 0;transform:translate(-50%,-100%) rotate(-45deg);box-shadow:0 2px 6px #000}.location-row{display:grid;grid-template-columns:1fr 1fr;gap:8px}.map-link{display:none;color:#72ddea}@media(max-width:750px){.panel{grid-template-columns:1fr}.viewer{height:48vh}.viewer img{max-height:48vh}.timeline{padding-left:58px}.timeline:before{left:28px}.year-group h2{margin-left:-50px}}
</style></head><body><header><strong>Photo Timeline</strong><input id=q placeholder="Search paths, titles, tags"><select id=person><option value="">All people</option></select><input id=year type=number placeholder="Year" min=1900 style="width:90px"><label><input id=hideNsfw type=checkbox checked onchange="load()"> Hide NSFW</label><label><input id=hideScreenshots type=checkbox checked onchange="load()"> Hide screenshots</label><label><input id=hideDocuments type=checkbox checked onchange="load()"> Hide documents</label><select id=contentFilter onchange="load()"><option value=normal>Normal browsing</option><option value=nsfw>NSFW only</option><option value=all>Include NSFW</option><option value=screenshot>Screenshots only</option><option value=document>Documents only</option></select><button onclick="load()">Search</button><button onclick="post('/api/scan',{})">Scan now</button><button onclick="post('/api/open-desktop',{})">Desktop app</button></header><main id=timeline class=timeline></main>
<div style="text-align:center;padding:15px"><button id=more onclick="load(false)">Load more</button></div><div id=modal class=modal><div class=panel><div class=viewer><img id=full></div><div class=editor><button class=close onclick="modal.style.display='none'">Close</button><h2 id=name></h2><label>Title<input id=title></label><label>Description<textarea id=description rows=4></textarea></label><label>Tags<input id=tags placeholder="family, vacation"></label><label>Rating<select id=rating><option value=0>Unrated</option><option value=1>★</option><option value=2>★★</option><option value=3>★★★</option><option value=4>★★★★</option><option value=5>★★★★★</option></select></label><label>Capture year<input id=captureYear type=number min=1900></label><h3>Location</h3><label>Place name<input id=locationName placeholder="Home, Chicago, Grand Canyon…"></label><div class=location-row><label>Latitude<input id=latitude type=number min=-90 max=90 step=any></label><label>Longitude<input id=longitude type=number min=-180 max=180 step=any></label></div><div id=map class=map><div id=pin class=pin></div></div><a id=mapLink class=map-link target=_blank rel="noopener noreferrer">Open in OpenStreetMap</a><label>Content type<select id=mediaKindOverride><option value="">Automatic</option><option value=photo>Photo</option><option value=screenshot>Screenshot</option><option value=document>Document</option></select></label><label>NSFW classification<select id=nsfwOverride><option value="">Automatic</option><option value=0>Mark safe</option><option value=1>Mark NSFW</option></select></label><div id=nsfwScore class=muted></div><div id=saveState class=save-state></div><h3>Detected faces</h3><div id=faces class=faces></div></div></div></div>
<script>const token='__TOKEN__';let current=null,identities=[],offset=0,saveTimer=null;async function api(u,o){let r=await fetch(u,o);if(!r.ok)throw Error(await r.text());return r.json()}async function post(u,d){let x=await api(u,{method:'POST',headers:{'Content-Type':'application/json','X-Gallery-Token':token},body:JSON.stringify(d)});if(x.message)alert(x.message);return x}async function people(){identities=await api('/api/identities');for(let p of identities){let o=document.createElement('option');o.value=p.id;o.textContent=p.name;person.append(o)}}function yearGrid(value){let label=value||'Unknown date',id='year-'+String(label).replace(/\W/g,'-'),grid=document.getElementById(id);if(!grid){let section=document.createElement('section');section.className='year-group';section.innerHTML=`<h2>${esc(String(label))}</h2><div id="${id}" class="grid"></div>`;timeline.append(section);grid=document.getElementById(id)}return grid}async function load(reset=true){if(reset){offset=0;timeline.innerHTML=''}let filter=contentFilter.value==='nsfw'?'nsfw':(contentFilter.value==='all'||!hideNsfw.checked?'all':'safe');let media=['screenshot','document'].includes(contentFilter.value)?contentFilter.value:'';let excluded=[];if(!media&&hideScreenshots.checked)excluded.push('screenshot');if(!media&&hideDocuments.checked)excluded.push('document');let p=new URLSearchParams({q:q.value,identity_id:person.value,year:year.value,nsfw:filter,media_kind:media,exclude_kinds:excluded.join(','),limit:100,offset});let a=await api('/api/photos?'+p);offset+=a.length;more.style.display=a.length<100?'none':'';for(let x of a){let c=document.createElement('article');c.className='card';c.innerHTML=`<img loading=lazy src="/media?id=${x.id}&thumb=1"><div class=info><b>${esc(x.title||x.name)}</b><div class=muted>${x.identified_count}/${x.face_count} faces · ${x.media_kind}${x.location_name?' · '+esc(x.location_name):''}${x.is_nsfw?' · NSFW':''}</div><div class=stars>${'★'.repeat(x.rating)}</div></div>`;c.onclick=()=>openPhoto(x.id);yearGrid(x.year).append(c)}}function updateMap(){let lat=parseFloat(latitude.value),lon=parseFloat(longitude.value),valid=Number.isFinite(lat)&&Number.isFinite(lon)&&lat>=-90&&lat<=90&&lon>=-180&&lon<=180;pin.style.display=valid?'block':'none';mapLink.style.display=valid?'inline-block':'none';if(valid){pin.style.left=((lon+180)/360*100)+'%';pin.style.top=((90-lat)/180*100)+'%';mapLink.href=`https://www.openstreetmap.org/?mlat=${lat}&mlon=${lon}#map=13/${lat}/${lon}`}}async function openPhoto(id){clearTimeout(saveTimer);current=await api('/api/photo?id='+id);full.src='/media?id='+id;name.textContent=current.name;title.value=current.title||'';description.value=current.description||'';tags.value=current.tags||'';rating.value=current.rating||0;captureYear.value=current.year||'';locationName.value=current.location_name||'';latitude.value=current.latitude==null?'':current.latitude;longitude.value=current.longitude==null?'':current.longitude;updateMap();mediaKindOverride.value=current.media_kind_override==null?'':current.media_kind_override;nsfwOverride.value=current.nsfw_override==null?'':current.nsfw_override;saveState.textContent='';nsfwScore.textContent=`Detected type: ${current.media_kind}. `+(current.nsfw_score==null?'Not NSFW-classified yet':`Automatic NSFW confidence: ${(current.nsfw_score*100).toFixed(1)}%`);faces.innerHTML=(current.faces||[]).map(f=>`<div>${esc(f.name||(f.unknown?'Unknown person':'Unprocessed'))}${f.art?' · artwork':''}${f.estimated_age!=null?' · age '+f.estimated_age:''}<br><select onchange="assignFace(${f.id},this.value)"><option value="">Reassign…</option>${identities.map(p=>`<option value="${p.id}">${esc(p.name)}</option>`).join('')}</select></div>`).join('');modal.style.display='block'}async function assignFace(face,id){if(!id)return;await post('/api/face',{face_id:face,identity_id:+id});saveState.textContent='Face assignment saved';await openPhoto(current.id)}function queueSave(immediate=false){if(!current)return;clearTimeout(saveTimer);updateMap();saveState.textContent='Saving…';saveTimer=setTimeout(save,immediate?0:700)}async function save(){if(!current)return;saveTimer=null;try{await post('/api/photo',{id:current.id,title:title.value,description:description.value,tags:tags.value,rating:+rating.value,capture_year:captureYear.value?+captureYear.value:null,location_name:locationName.value,latitude:latitude.value===''?null:+latitude.value,longitude:longitude.value===''?null:+longitude.value,nsfw_override:nsfwOverride.value===''?null:+nsfwOverride.value,media_kind_override:mediaKindOverride.value||null});saveState.textContent='Saved automatically';}catch(e){saveState.textContent='Save failed: '+e.message}}function esc(s){let d=document.createElement('div');d.textContent=s;return d.innerHTML}for(let id of ['title','description','tags','captureYear','locationName','latitude','longitude'])document.getElementById(id).addEventListener('input',()=>queueSave(false));for(let id of ['rating','mediaKindOverride','nsfwOverride'])document.getElementById(id).addEventListener('change',()=>queueSave(true));people().then(load);</script></body></html>'''

PAGE = PAGE.replace(
    "</body>",
    r'''<script>
const openPhotoWithNsfwDetails = openPhoto;
openPhoto = async function(id) {
  await openPhotoWithNsfwDetails(id);
  let box = document.getElementById('nsfwFindings');
  if (!box) {
    box = document.createElement('div');
    box.id = 'nsfwFindings';
    box.className = 'faces';
    nsfwScore.insertAdjacentElement('afterend', box);
  }
  const findings = current.nsfw_detections || [];
  if (current.nsfw_review_required) {
    nsfwScore.textContent += ' Models disagree — manual review recommended.';
  }
  box.innerHTML = findings.length
    ? '<b>Model detections</b>' + findings.map(item => {
        const label = String(item.label).toLowerCase().replaceAll('_', ' ')
          .replace(/\b\w/g, value => value.toUpperCase());
        const reason = item.explicit ? ' · contributes to NSFW score' : '';
        return `<div>${esc(label)} — ${(item.score * 100).toFixed(1)}%${reason}</div>`;
      }).join('')
    : '<div class="muted">No anatomical detections above 20% confidence.</div>';
};
</script></body>''',
)
PAGE = PAGE.replace(
    '<option value=nsfw>NSFW only</option>',
    '<option value=nsfw>NSFW only</option><option value=review>Needs NSFW review</option>',
).replace(
    "let filter=contentFilter.value==='nsfw'?'nsfw':",
    "let filter=['nsfw','review'].includes(contentFilter.value)?contentFilter.value:",
)
PAGE = PAGE.replace(
    "</style>",
    r'''.danger-zone{border-top:1px solid #633;margin-top:22px;padding-top:16px}.danger-button{background:#9d1c25;border-color:#ef5963;font-weight:700}.danger-button:hover{background:#c32632}.danger-dialog{position:fixed;inset:0;z-index:20;display:none;align-items:center;justify-content:center;background:#000c;padding:20px}.danger-dialog.open{display:flex}.danger-box{width:min(480px,100%);background:#251b1c;border:2px solid #e04450;border-radius:10px;padding:20px;box-shadow:0 12px 45px #000}.danger-box h2{color:#ff7b84;margin-top:0}.danger-actions{display:flex;justify-content:flex-end;gap:10px;margin-top:18px}</style>''',
).replace(
    "</body>",
    r'''<div id=deleteDialog class=danger-dialog role=dialog aria-modal=true aria-labelledby=deleteTitle><div class=danger-box><h2 id=deleteTitle>Danger: permanently delete photo?</h2><p>This will permanently delete <strong id=deleteName></strong> from your computer and remove it from the catalog.</p><p><strong>This action cannot be undone.</strong></p><div id=deleteError class=save-state></div><div class=danger-actions><button onclick=cancelDelete()>Cancel</button><button id=confirmDeleteButton class=danger-button onclick=confirmDelete()>Delete permanently</button></div></div></div><script>
const openPhotoWithDelete = openPhoto;
let deleteScrollPosition = 0;
openPhoto = async function(id) {
  await openPhotoWithDelete(id);
  let zone = document.getElementById('dangerZone');
  if (!zone) {
    zone = document.createElement('div');
    zone.id = 'dangerZone';
    zone.className = 'danger-zone';
    zone.innerHTML = '<h3>Danger zone</h3><button class="danger-button" onclick="requestDelete()">Delete this photo...</button>';
    document.querySelector('.editor').append(zone);
  }
};
function requestDelete() {
  if (!current) return;
  clearTimeout(saveTimer);
  deleteScrollPosition = window.scrollY;
  deleteName.textContent = current.name;
  deleteError.textContent = '';
  confirmDeleteButton.disabled = false;
  deleteDialog.classList.add('open');
}
function cancelDelete() { deleteDialog.classList.remove('open'); }
async function confirmDelete() {
  if (!current) return;
  const deletedId = current.id;
  confirmDeleteButton.disabled = true;
  deleteError.textContent = 'Deleting...';
  try {
    await post('/api/delete-photo', {id: current.id});
    deleteDialog.classList.remove('open');
    modal.style.display = 'none';
    current = null;
    removeDeletedCard(deletedId);
    requestAnimationFrame(() => window.scrollTo(0, Math.min(
      deleteScrollPosition, Math.max(0, document.documentElement.scrollHeight - window.innerHeight)
    )));
  } catch (error) {
    deleteError.textContent = 'Delete failed: ' + error.message;
    confirmDeleteButton.disabled = false;
  }
}
function removeDeletedCard(id) {
  const card = document.querySelector(`.card[data-photo-id="${id}"]`);
  if (!card) return;
  const day = card.closest('.day-group');
  const month = card.closest('.month-group');
  const yearSection = card.closest('.year-group');
  card.remove();
  offset = Math.max(0, offset - 1);
  if (day && !day.querySelector('.card')) day.remove();
  if (month && !month.querySelector('.card')) month.remove();
  if (yearSection && !yearSection.querySelector('.card')) yearSection.remove();
}
</script></body>''',
)
PAGE = PAGE.replace(
    "let current=null,identities=[],offset=0,saveTimer=null;",
    "let current=null,identities=[],offset=0,saveTimer=null,loading=false,hasMore=true,reloadAfterLoad=false;",
).replace(
    "async function load(reset=true){if(reset){offset=0;timeline.innerHTML=''}",
    "async function load(reset=true){if(loading){if(reset)reloadAfterLoad=true;return}if(!reset&&!hasMore)return;loading=true;try{if(reset){offset=0;hasMore=true;timeline.innerHTML=''}",
).replace(
    "yearGrid(x.year).append(c)}}function updateMap()",
    "yearGrid(x.year).append(c)}hasMore=a.length===100}finally{loading=false;if(reloadAfterLoad){reloadAfterLoad=false;load(true)}}}function updateMap()",
).replace(
    "people().then(load);</script>",
    "const infiniteObserver=new IntersectionObserver(entries=>{if(entries.some(entry=>entry.isIntersecting))load(false)},{rootMargin:'600px 0px'});infiniteObserver.observe(more);people().then(load);</script>",
)
PAGE = PAGE.replace(
    "c.className='card';c.innerHTML=",
    "c.className='card';c.dataset.photoId=x.id;c.innerHTML=",
).replace(
    "</style>",
    r'''.month-group{margin:0 0 28px}.month-group>h3{margin:12px 0;padding:8px 12px;background:#18363c;border-left:4px solid #49aeb9;border-radius:5px}.day-group{margin:0 0 20px}.day-group>h4{margin:10px 0 8px;color:#9de9f1;font-size:14px}</style>''',
).replace(
    "yearGrid(x.year).append(c)",
    "timelineGrid(x).append(c)",
).replace(
    "const infiniteObserver=new IntersectionObserver",
    r'''function timelineGrid(photo){let yearLabel=photo.year||'Unknown date',yearId='year-'+String(yearLabel).replace(/\W/g,'-'),yearSection=document.getElementById(yearId);if(!yearSection){yearSection=document.createElement('section');yearSection.id=yearId;yearSection.className='year-group';yearSection.innerHTML=`<h2>${esc(String(yearLabel))}</h2><div class="months"></div>`;timeline.append(yearSection)}let date=photo.capture_date?new Date(photo.capture_date+'T12:00:00'):null,monthKey=date?String(date.getMonth()+1).padStart(2,'0'):'unknown',monthLabel=date?date.toLocaleDateString(undefined,{month:'long'}):'Unknown month',monthId=yearId+'-month-'+monthKey,month=document.getElementById(monthId);if(!month){month=document.createElement('section');month.id=monthId;month.className='month-group';month.innerHTML=`<h3>${esc(monthLabel)}</h3><div class="days"></div>`;yearSection.querySelector('.months').append(month)}let dayKey=date?String(date.getDate()).padStart(2,'0'):'unknown',dayLabel=date?date.toLocaleDateString(undefined,{weekday:'long',month:'long',day:'numeric'}):'Unknown day',dayId=monthId+'-day-'+dayKey,grid=document.getElementById(dayId);if(!grid){let day=document.createElement('section');day.className='day-group';day.innerHTML=`<h4>${esc(dayLabel)}</h4><div id="${dayId}" class="grid"></div>`;month.querySelector('.days').append(day);grid=document.getElementById(dayId)}return grid}const infiniteObserver=new IntersectionObserver''',
)
PAGE = PAGE.replace(
    "</style>",
    r'''.card{position:relative}.select-box{display:none;position:absolute;z-index:1;top:9px;left:9px;width:34px;height:34px;border-radius:50%;background:#171717dd;border:2px solid #ddd;font-size:20px;line-height:1}.selection-mode .select-box{display:block}.card.selected{outline:4px solid #5de0ee;outline-offset:-4px}.card.selected .select-box{display:block;background:#168594;border-color:#baf8ff}.bulk-bar{display:none;position:fixed;z-index:12;left:50%;bottom:18px;transform:translateX(-50%);align-items:center;gap:9px;background:#152a2e;border:1px solid #55c9d6;border-radius:10px;padding:10px 14px;box-shadow:0 8px 30px #000}.selection-mode .bulk-bar{display:flex}.archive-button{background:#17627a;border-color:#56bad5;font-weight:700}</style>''',
).replace(
    "c.onclick=()=>openPhoto(x.id);",
    "attachSelection(c,x);",
).replace(
    "</header>",
    r'''<button id=selectModeButton onclick=toggleSelectionMode()>Select photos</button></header>''',
).replace(
    "</body>",
    r'''<div id=bulkBar class=bulk-bar><strong id=selectedCount>0 selected</strong><button class=archive-button onclick="requestBulkAction('archive')">Archive selected</button><button class=danger-button onclick="requestBulkAction('delete')">Delete selected</button><button onclick=clearSelection()>Done</button></div><div id=bulkDialog class=danger-dialog role=dialog aria-modal=true aria-labelledby=bulkTitle><div class=danger-box><h2 id=bulkTitle></h2><p id=bulkMessage></p><div id=bulkError class=save-state></div><div class=danger-actions><button onclick=cancelBulkAction()>Cancel</button><button id=bulkConfirmButton onclick=executeBulkAction()></button></div></div></div><script>
const selectedPhotos = new Map();
let selectionMode = false;
let pendingBulkAction = null;
function attachSelection(card, photo) {
  const picker = document.createElement('button');
  picker.className = 'select-box';
  picker.textContent = '✓';
  picker.setAttribute('aria-label', `Select ${photo.name}`);
  picker.onclick = event => { event.stopPropagation(); toggleSelected(card, photo); };
  card.prepend(picker);
  card.onclick = () => selectionMode ? toggleSelected(card, photo) : openPhoto(photo.id);
}
function toggleSelectionMode() {
  selectionMode = !selectionMode;
  document.body.classList.toggle('selection-mode', selectionMode);
  selectModeButton.textContent = selectionMode ? 'Selecting photos' : 'Select photos';
  if (!selectionMode) clearSelection();
}
function toggleSelected(card, photo) {
  if (!selectionMode) {
    selectionMode = true;
    document.body.classList.add('selection-mode');
    selectModeButton.textContent = 'Selecting photos';
  }
  if (selectedPhotos.has(photo.id)) selectedPhotos.delete(photo.id);
  else selectedPhotos.set(photo.id, photo.name);
  card.classList.toggle('selected', selectedPhotos.has(photo.id));
  updateSelectedCount();
}
function updateSelectedCount() { selectedCount.textContent = `${selectedPhotos.size} selected`; }
function clearSelection() {
  selectedPhotos.clear();
  document.querySelectorAll('.card.selected').forEach(card => card.classList.remove('selected'));
  selectionMode = false;
  document.body.classList.remove('selection-mode');
  selectModeButton.textContent = 'Select photos';
  updateSelectedCount();
}
function requestBulkAction(action) {
  if (!selectedPhotos.size) return;
  pendingBulkAction = action;
  const count = selectedPhotos.size;
  bulkError.textContent = '';
  bulkConfirmButton.disabled = false;
  if (action === 'delete') {
    bulkTitle.textContent = `Danger: permanently delete ${count} photo${count === 1 ? '' : 's'}?`;
    bulkMessage.innerHTML = '<strong>This action cannot be undone.</strong> The selected source files and their catalog records will be permanently removed.';
    bulkConfirmButton.textContent = 'Delete permanently';
    bulkConfirmButton.className = 'danger-button';
  } else {
    bulkTitle.textContent = `Archive ${count} photo${count === 1 ? '' : 's'}?`;
    bulkMessage.textContent = 'The selected originals will be moved into Hidden Pictures\\GeneralArchive.zip and removed from the active catalog.';
    bulkConfirmButton.textContent = 'Move to archive';
    bulkConfirmButton.className = 'archive-button';
  }
  bulkDialog.classList.add('open');
}
function cancelBulkAction() { bulkDialog.classList.remove('open'); pendingBulkAction = null; }
async function executeBulkAction() {
  if (!pendingBulkAction || !selectedPhotos.size) return;
  const action = pendingBulkAction;
  const ids = [...selectedPhotos.keys()];
  const preservedScroll = window.scrollY;
  bulkConfirmButton.disabled = true;
  bulkError.textContent = action === 'archive' ? 'Archiving...' : 'Deleting...';
  try {
    await post(action === 'archive' ? '/api/archive-photos' : '/api/delete-photos', {ids});
    ids.forEach(removeDeletedCard);
    bulkDialog.classList.remove('open');
    pendingBulkAction = null;
    clearSelection();
    requestAnimationFrame(() => window.scrollTo(0, Math.min(
      preservedScroll, Math.max(0, document.documentElement.scrollHeight - window.innerHeight)
    )));
  } catch (error) {
    bulkError.textContent = `${action === 'archive' ? 'Archive' : 'Delete'} failed: ${error.message}`;
    bulkConfirmButton.disabled = false;
  }
}
</script></body>''',
)


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
                    catalog.update_gallery_metadata(
                        int(body["id"]), str(body.get("title", "")),
                        str(body.get("description", "")), str(body.get("tags", "")),
                        int(body.get("rating", 0)), body.get("capture_year"),
                        body.get("nsfw_override"), body.get("media_kind_override"),
                        str(body.get("location_name", "")), body.get("latitude"),
                        body.get("longitude"),
                    )
                finally: catalog.close()
                self._json({"ok": True})
            elif self.path == "/api/face":
                catalog = self._catalog()
                try: catalog.assign_face(int(body["face_id"]), int(body["identity_id"]))
                finally: catalog.close()
                self._json({"ok": True})
            elif self.path == "/api/delete-photo":
                catalog = self._catalog()
                try: deleted = catalog.delete_photo(int(body["id"]))
                finally: catalog.close()
                self._json({"ok": True, "deleted": deleted.name})
            elif self.path == "/api/delete-photos":
                catalog = self._catalog()
                try: deleted = catalog.delete_photos([int(value) for value in body["ids"]])
                finally: catalog.close()
                self._json({"ok": True, "deleted": len(deleted)})
            elif self.path == "/api/archive-photos":
                catalog = self._catalog()
                try:
                    archived = catalog.archive_photos(
                        [int(value) for value in body["ids"]], GENERAL_ARCHIVE, PICTURES_ROOT
                    )
                finally: catalog.close()
                self._json({"ok": True, "archived": len(archived), "archive": str(GENERAL_ARCHIVE)})
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
