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


def archive_path_for_name(value: object = None) -> Path:
    """Resolve a user-selected archive name without allowing paths outside Hidden Pictures."""
    name = str(value or "").strip() or GENERAL_ARCHIVE.name
    if Path(name).name != name or any(character in name for character in '<>:"/\\|?*'):
        raise ValueError("Archive names cannot contain a folder path or reserved characters.")
    candidate = Path(name)
    if not candidate.suffix:
        candidate = candidate.with_suffix(".zip")
    if candidate.suffix.lower() not in {".zip", ".hide"} or not candidate.stem.strip(". "):
        raise ValueError("Archive names must end in .zip or .hide.")
    return GENERAL_ARCHIVE.parent / candidate.name


def available_archives() -> list[str]:
    names = {GENERAL_ARCHIVE.name}
    if GENERAL_ARCHIVE.parent.is_dir():
        names.update(
            path.name for path in GENERAL_ARCHIVE.parent.iterdir()
            if path.is_file() and path.suffix.lower() in {".zip", ".hide"}
        )
    return [GENERAL_ARCHIVE.name, *sorted(names - {GENERAL_ARCHIVE.name}, key=str.casefold)]


PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Face Photo Finder Gallery</title><style>
:root{color-scheme:dark;background:#111;color:#eee;font:15px system-ui}body{margin:0}header{position:sticky;top:0;z-index:2;background:#181818;padding:12px;display:flex;flex-wrap:wrap;gap:10px;align-items:center;box-shadow:0 2px 8px #000}input,select,textarea,button{background:#292929;color:#eee;border:1px solid #555;border-radius:6px;padding:8px}button{cursor:pointer}.timeline{position:relative;padding:18px 18px 18px 86px}.timeline:before{content:'';position:absolute;left:48px;top:0;bottom:0;width:3px;background:#3a6f78}.year-group{position:relative;margin-bottom:30px}.year-group h2{position:sticky;top:68px;z-index:1;margin:0 0 12px -72px;width:58px;text-align:center;background:#164d57;border:2px solid #59d7e6;border-radius:18px;padding:6px;font-size:16px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}.card{background:#202020;border-radius:9px;overflow:hidden;cursor:pointer;box-shadow:0 3px 12px #0008}.card img{width:100%;height:180px;object-fit:cover}.info{padding:9px}.muted{color:#aaa;font-size:12px}.stars{color:#ffd65a}.modal{position:fixed;inset:0;background:#000d;display:none;z-index:5;padding:0}.panel{width:100vw;max-width:none;height:100vh;margin:0;background:#202020;border-radius:0;display:grid;grid-template-columns:minmax(0,1fr) clamp(340px,25vw,460px);overflow:hidden}.viewer{min-width:0;background:#090909;display:flex;align-items:center;justify-content:center}.viewer img{max-width:100%;max-height:100vh}.editor{padding:16px;overflow:auto}.editor input,.editor textarea,.editor select{width:100%;box-sizing:border-box;margin:5px 0 12px}.faces div{padding:7px;background:#292929;margin:5px 0;border-radius:5px}.close{float:right}.save-state{min-height:20px;color:#69dfbd;margin:8px 0}.map{position:relative;height:150px;border:1px solid #49656b;border-radius:8px;margin:6px 0 8px;overflow:hidden;background-color:#18323a;background-image:linear-gradient(#ffffff14 1px,transparent 1px),linear-gradient(90deg,#ffffff14 1px,transparent 1px);background-size:25% 50%}.map:after{content:'Offline world map';position:absolute;right:7px;bottom:5px;color:#9eb7bc;font-size:11px}.pin{display:none;position:absolute;width:14px;height:14px;background:#ff4e67;border:3px solid white;border-radius:50% 50% 50% 0;transform:translate(-50%,-100%) rotate(-45deg);box-shadow:0 2px 6px #000}.location-row{display:grid;grid-template-columns:1fr 1fr;gap:8px}.map-link{display:none;color:#72ddea}@media(max-width:750px){.panel{grid-template-columns:1fr;grid-template-rows:minmax(42vh,1fr) minmax(0,1fr)}.viewer{height:auto;min-height:42vh}.viewer img{max-height:58vh}.timeline{padding-left:58px}.timeline:before{left:28px}.year-group h2{margin-left:-50px}}
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
        const reason = item.label === 'WHOLE_IMAGE_NSFW' && item.explicit
          ? ' · automatic NSFW decision'
          : (item.explicit ? ' · explicit-content evidence' : '');
        return `<div>${esc(label)} — ${(item.score * 100).toFixed(1)}%${reason}</div>`;
      }).join('')
    : '<div class="muted">No anatomical detections above 20% confidence.</div>';
};
</script></body>''',
)

PAGE = PAGE.replace(
    "</style>",
    r'''.app-tabs{position:sticky;top:0;z-index:15;display:flex;gap:6px;background:#101719;padding:8px 14px;border-bottom:1px solid #31535a}.app-tabs button{border-color:transparent;background:transparent;font-weight:700}.app-tabs button.active{background:#176679;border-color:#63d7e8}.app-tabs+header{top:49px}.identity-view{display:none;padding:22px;max-width:1400px;margin:auto}.identity-toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:15px}.identity-toolbar input{min-width:280px}.identity-table-wrap{overflow:auto;border:1px solid #3e555a;border-radius:9px}.identity-table{width:100%;border-collapse:collapse;background:#1c1f20}.identity-table th,.identity-table td{padding:10px;border-bottom:1px solid #353d3f;text-align:left;white-space:nowrap}.identity-table th{position:sticky;top:0;background:#18363c;color:#bdf5fb}.identity-table input{box-sizing:border-box;width:100%}.identity-table .name-input{min-width:220px}.identity-table .year-input{width:100px}.identity-status{min-height:22px;color:#72dfbd}.reference-strip{display:flex;gap:6px;max-width:390px;overflow-x:auto;padding:3px}.reference-thumb{padding:0;border:2px solid #47656b;background:#101010;flex:0 0 auto}.reference-thumb:hover{border-color:#62dbe8}.reference-thumb img{display:block;width:58px;height:58px;object-fit:cover;border-radius:4px}.reference-viewer{width:min(620px,95vw);background:#1c2325;border-color:#5dd4e2}.reference-viewer img{display:block;max-width:100%;max-height:72vh;margin:auto;border-radius:7px}.reference-viewer h2{color:#baf6fb}@media(max-width:750px){.app-tabs+header{top:49px}.identity-view{padding:12px}.identity-toolbar input{min-width:0;width:100%}.reference-strip{max-width:250px}}</style>''',
).replace(
    "<body><header>",
    r'''<body><nav class=app-tabs aria-label="Gallery sections"><button id=timelineTabButton class=active onclick="showAppTab('timeline')">Timeline</button><button id=identityTabButton onclick="showAppTab('identity')">Identity</button></nav><header id=timelineHeader>''',
).replace(
    '<div style="text-align:center;padding:15px"><button id=more',
    '<div id=timelineMore style="text-align:center;padding:15px"><button id=more',
).replace(
    "<div id=modal class=modal>",
    r'''<section id=identityView class=identity-view><div class=identity-toolbar><h1>Identity management</h1><button id=recognizedPeopleTab class=active onclick="showIdentityKind('recognized')">Recognized people</button><button id=unidentifiedPeopleTab onclick="showIdentityKind('unidentified')">Unidentified people</button><input id=identitySearch placeholder="Search identities or IDs" oninput=renderIdentityTable()><button onclick=createNewIdentity()>New identity</button><button id=mergeIdentityButton disabled onclick=requestIdentityMerge()>Merge selected…</button><button onclick=loadIdentityTable()>Refresh</button></div><div id=identityStatus class=identity-status></div><div id=recognizedPeoplePanel class=identity-table-wrap><table class=identity-table><thead><tr><th>Select</th><th>ID</th><th>Name</th><th>Reference images</th><th>Birth year</th><th>Approx. age</th><th>Photos</th><th>Faces</th><th>Profile samples</th><th>Timeline</th></tr></thead><tbody id=identityTableBody></tbody></table></div><div id=unidentifiedPeoplePanel class=identity-table-wrap style="display:none"><table class=identity-table><thead><tr><th>Anonymous group</th><th>Appearances</th><th>Photos</th><th>Faces</th><th>Timeline</th></tr></thead><tbody id=unidentifiedTableBody></tbody></table></div></section><div id=modal class=modal>''',
).replace(
    "</body>",
    r'''<div id=referenceDialog class=danger-dialog role=dialog aria-modal=true aria-labelledby=referenceTitle><div class="danger-box reference-viewer"><button class=close onclick=closeReferencePreview()>Close</button><h2 id=referenceTitle>Reference image</h2><img id=referenceFull alt="Identity reference image"><div class=reference-actions><button class="danger-button reference-not-face" onclick=removeCurrentReferenceFace()>Not a face</button></div></div></div><script>
let identityManagementRows = [], unidentifiedManagementRows = [], identityKind = 'recognized', unknownGroupFilter = '', selectedIdentityIds = new Set();
function showAppTab(tabName, preservePersonFilter=false) {
  const identityActive = tabName === 'identity';
  if (!identityActive && !preservePersonFilter) unknownGroupFilter = '';
  timelineTabButton.classList.toggle('active', !identityActive);
  identityTabButton.classList.toggle('active', identityActive);
  timelineHeader.style.display = identityActive ? 'none' : '';
  timeline.style.display = identityActive ? 'none' : '';
  timelineMore.style.display = identityActive ? 'none' : '';
  identityView.style.display = identityActive ? 'block' : 'none';
  if (identityActive) { clearSelection(); loadIdentityTable(); }
}
async function loadIdentityTable() {
  identityStatus.textContent = 'Loading identities...';
  try {
    [identityManagementRows, unidentifiedManagementRows] = await Promise.all([api('/api/identity-summaries'),api('/api/unidentified-summaries')]);
    identityStatus.textContent = identityKind === 'recognized' ? `${identityManagementRows.length} recognized people` : `${unidentifiedManagementRows.length} unidentified people`;
    renderIdentityTable();
  } catch (error) { identityStatus.textContent = 'Load failed: ' + error.message; }
}
function renderIdentityTable() {
  if (identityKind === 'unidentified') { renderUnidentifiedTable(); return; }
  const query = identitySearch.value.trim().toLowerCase();
  identityTableBody.innerHTML = '';
  for (const identity of identityManagementRows.filter(item => item.name.toLowerCase().includes(query) || String(item.id).includes(query))) {
    const row = document.createElement('tr');
    const selectCell = row.insertCell(); const selector = document.createElement('input'); selector.type = 'checkbox';
    selector.checked = selectedIdentityIds.has(identity.id); selector.setAttribute('aria-label', `Select ${identity.name} identity ${identity.id}`);
    selector.onchange = () => { selector.checked ? selectedIdentityIds.add(identity.id) : selectedIdentityIds.delete(identity.id); updateIdentityMergeButton(); };
    selectCell.append(selector);
    row.insertCell().textContent = identity.id;
    const nameCell = row.insertCell();
    const nameInput = document.createElement('input');
    nameInput.className = 'name-input';
    nameInput.value = identity.name;
    nameCell.append(nameInput);
    const referenceCell = row.insertCell();
    const referenceStrip = document.createElement('div');
    referenceStrip.className = 'reference-strip';
    for (const faceId of identity.reference_face_ids || []) {
      const referenceItem = document.createElement('div'); referenceItem.className = 'reference-item';
      const previewButton = document.createElement('button');
      previewButton.className = 'reference-thumb';
      previewButton.title = `Open reference for ${identity.name}`;
      const preview = document.createElement('img');
      preview.loading = 'lazy';
      preview.alt = `Reference for ${identity.name}`;
      preview.src = `/api/identity-reference?id=${faceId}`;
      previewButton.append(preview);
      previewButton.onclick = () => openReferencePreview(faceId, identity.name);
      referenceItem.append(previewButton); referenceStrip.append(referenceItem);
    }
    if (!referenceStrip.children.length) referenceStrip.textContent = 'No eligible references';
    referenceCell.append(referenceStrip);
    const yearCell = row.insertCell();
    const yearInput = document.createElement('input');
    yearInput.className = 'year-input';
    yearInput.type = 'number';
    yearInput.min = '1900';
    yearInput.value = identity.birth_year == null ? '' : identity.birth_year;
    yearCell.append(yearInput);
    for (const value of [identity.age ?? '—', identity.photo_count, identity.face_count, identity.profile_sample_count]) {
      const cell = row.insertCell(); cell.textContent = value;
    }
    const actionCell = row.insertCell();
    const viewButton = document.createElement('button');
    viewButton.textContent = 'View photos';
    viewButton.onclick = () => viewIdentityTimeline(identity.id);
    actionCell.append(viewButton);
    const saveIdentityName = () => saveIdentityRow(identity, nameInput, yearInput, false);
    nameInput.addEventListener('input', () => {
      clearTimeout(nameInput.identitySaveTimer);
      identityStatus.textContent = 'Name changed — saving...';
      nameInput.identitySaveTimer = setTimeout(saveIdentityName, 700);
    });
    nameInput.addEventListener('change', () => {
      clearTimeout(nameInput.identitySaveTimer);
      saveIdentityName();
    });
    yearInput.addEventListener('change', () => saveIdentityRow(identity, nameInput, yearInput, true));
    identityTableBody.append(row);
  }
}
function updateIdentityMergeButton() { mergeIdentityButton.disabled = selectedIdentityIds.size < 2; mergeIdentityButton.textContent = selectedIdentityIds.size < 2 ? 'Merge selected…' : `Merge ${selectedIdentityIds.size} selected…`; }
async function createNewIdentity() {
  const name = prompt('Name for the new identity (duplicate names are allowed):'); if (!name || !name.trim()) return;
  try { await post('/api/create-identity',{name}); await people(true); await loadIdentityTable(); identityStatus.textContent = `Created a distinct identity named ${name.trim()}`; }
  catch (error) { identityStatus.textContent = 'Create failed: ' + error.message; }
}
function requestIdentityMerge() {
  const selected = identityManagementRows.filter(item => selectedIdentityIds.has(item.id)); if (selected.length < 2) return;
  mergeIdentityTarget.innerHTML = selected.map(item => `<option value="${item.id}">${esc(item.name)} (#${item.id})</option>`).join('');
  mergeIdentitySummary.textContent = `Merge ${selected.length} identities. All faces and manual person tags will move to the identity you keep; the other identity IDs will be permanently removed.`;
  mergeIdentityError.textContent = ''; mergeIdentityDialog.classList.add('open');
}
function cancelIdentityMerge() { mergeIdentityDialog.classList.remove('open'); }
async function confirmIdentityMerge() {
  confirmIdentityMergeButton.disabled = true; mergeIdentityError.textContent = 'Merging…';
  try { await post('/api/merge-identities',{identity_ids:[...selectedIdentityIds],target_identity_id:+mergeIdentityTarget.value}); selectedIdentityIds.clear(); cancelIdentityMerge(); await people(true); await loadIdentityTable(); identityStatus.textContent = 'Identities merged successfully'; }
  catch (error) { mergeIdentityError.textContent = 'Merge failed: ' + error.message; }
  finally { confirmIdentityMergeButton.disabled = false; updateIdentityMergeButton(); }
}
function showIdentityKind(kind) {
  identityKind = kind;
  recognizedPeopleTab.classList.toggle('active', kind === 'recognized');
  unidentifiedPeopleTab.classList.toggle('active', kind === 'unidentified');
  recognizedPeoplePanel.style.display = kind === 'recognized' ? '' : 'none';
  unidentifiedPeoplePanel.style.display = kind === 'unidentified' ? '' : 'none';
  mergeIdentityButton.style.display = kind === 'recognized' ? '' : 'none';
  identitySearch.placeholder = kind === 'recognized' ? 'Search identities or IDs' : 'Search unidentified groups';
  renderIdentityTable();
}
function renderUnidentifiedTable() {
  const query = identitySearch.value.trim().toLowerCase();
  unidentifiedTableBody.innerHTML = '';
  const sortedGroups = unidentifiedManagementRows.filter(item => item.label.toLowerCase().includes(query));
  sortedGroups.sort((left,right) => identitySort.value === 'photo_count'
    ? right.photo_count-left.photo_count || right.id-left.id
    : identitySort.value === 'name'
      ? left.label.localeCompare(right.label,undefined,{sensitivity:'base'}) || right.id-left.id
      : (right.last_seen||'').localeCompare(left.last_seen||'') || right.id-left.id);
  for (const group of sortedGroups) {
    const row = document.createElement('tr');
    row.insertCell().textContent = group.label;
    row.insertCell().textContent = group.last_seen || 'Unknown';
    const referenceCell = row.insertCell();
    const strip = document.createElement('div'); strip.className = 'reference-strip';
    for (const faceId of group.reference_face_ids || []) {
      const referenceItem = document.createElement('div'); referenceItem.className = 'reference-item';
      const button = document.createElement('button'); button.className = 'reference-thumb';
      const preview = document.createElement('img'); preview.loading = 'lazy'; preview.alt = group.label;
      preview.src = `/api/unidentified-reference?id=${faceId}`; button.append(preview);
      button.onclick = () => openUnidentifiedPreview(faceId, group.label);
      referenceItem.append(button); strip.append(referenceItem);
    }
    referenceCell.append(strip);
    row.insertCell().textContent = group.photo_count;
    row.insertCell().textContent = group.face_count;
    const identityCell = row.insertCell(); const identitySelect = document.createElement('select');
    identitySelect.innerHTML = `<option value="">Choose known identity…</option>${identities.map(item=>`<option value="${item.id}">${esc(item.name)} (#${item.id})</option>`).join('')}`;
    const assignButton = document.createElement('button'); assignButton.textContent = 'Add to known identity';
    assignButton.onclick = () => assignUnknownCluster(group, identitySelect.value);
    identityCell.append(identitySelect, assignButton);
    const actionCell = row.insertCell(); const viewButton = document.createElement('button');
    viewButton.textContent = 'View photos'; viewButton.onclick = () => viewUnidentifiedTimeline(group.group_ids); actionCell.append(viewButton);
    unidentifiedTableBody.append(row);
  }
  identityStatus.textContent = `${unidentifiedManagementRows.length} unidentified review clusters`;
}
async function assignUnknownCluster(group, identityId) {
  if (!identityId) { identityStatus.textContent = 'Choose a known identity first.'; return; }
  const target = identities.find(item => item.id === +identityId);
  if (!confirm(`Assign all ${group.face_count} face(s) in this similar-identity cluster to ${target.name} (#${target.id})?`)) return;
  try { await post('/api/assign-unknown-groups',{group_ids:group.group_ids,identity_id:+identityId}); await people(true); await loadIdentityTable(); identityStatus.textContent = `Added cluster to ${target.name} (#${target.id})`; }
  catch (error) { identityStatus.textContent = 'Assignment failed: ' + error.message; }
}
async function saveIdentityRow(identity, nameInput, yearInput, rerender) {
  identityStatus.textContent = 'Saving identity...';
  try {
    await post('/api/identity', {id:identity.id,name:nameInput.value,birth_year:yearInput.value||null});
    identity.name = nameInput.value.trim().replace(/\s+/g, ' ');
    identity.birth_year = yearInput.value ? +yearInput.value : null;
    identity.age = identity.birth_year ? new Date().getFullYear() - identity.birth_year : null;
    const option = person.querySelector(`option[value="${identity.id}"]`);
    if (option) option.textContent = identity.name;
    identityStatus.textContent = 'Identity saved automatically';
    if (rerender) renderIdentityTable();
  } catch (error) { identityStatus.textContent = 'Save failed: ' + error.message; }
}
function viewIdentityTimeline(identityId) {
  unknownGroupFilter = '';
  person.value = String(identityId);
  showAppTab('timeline', true);
  load();
}
function viewUnidentifiedTimeline(groupIds) { person.value = ''; unknownGroupFilter = groupIds.join(','); showAppTab('timeline', true); load(); }
function openReferencePreview(faceId, identityName) {
  referenceTitle.textContent = `${identityName} reference`;
  referenceFull.src = `/api/identity-reference?id=${faceId}`;
  referenceDialog.dataset.faceId = faceId;
  referenceDialog.dataset.label = identityName;
  referenceDialog.classList.add('open');
}
function closeReferencePreview() {
  referenceDialog.classList.remove('open');
  referenceFull.removeAttribute('src');
  delete referenceDialog.dataset.faceId;
  delete referenceDialog.dataset.label;
}
function openUnidentifiedPreview(faceId, label) { referenceTitle.textContent = label; referenceFull.src = `/api/unidentified-reference?id=${faceId}`; referenceDialog.dataset.faceId = faceId; referenceDialog.dataset.label = label; referenceDialog.classList.add('open'); }
function removeCurrentReferenceFace() { const faceId = +referenceDialog.dataset.faceId; if (faceId) removeReferenceFace(faceId, referenceDialog.dataset.label || 'selected'); }
async function removeReferenceFace(faceId, label) {
  if (!confirm(`Mark this ${label} reference as not a face? It will be removed from the catalog and biometric matching.`)) return;
  try { await post('/api/face',{face_id:faceId,action:'remove'}); closeReferencePreview(); await loadIdentityTable(); identityStatus.textContent = 'False reference face removed'; }
  catch (error) { identityStatus.textContent = 'Removal failed: ' + error.message; }
}
</script></body>''',
)
PAGE = PAGE.replace(
    '<option value=nsfw>NSFW only</option>',
    '<option value=nsfw>NSFW only</option><option value=conflict>NSFW conflicts</option>',
).replace(
    "let filter=contentFilter.value==='nsfw'?'nsfw':",
    "let filter=['nsfw','conflict'].includes(contentFilter.value)?contentFilter.value:",
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
    await people(true);
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
    r'''.card{position:relative}.select-box{display:none;position:absolute;z-index:1;top:9px;left:9px;width:34px;height:34px;border-radius:50%;background:#171717dd;border:2px solid #ddd;font-size:20px;line-height:1}.selection-mode .select-box{display:block}.card.selected{outline:4px solid #5de0ee;outline-offset:-4px}.card.selected .select-box{display:block;background:#168594;border-color:#baf8ff}.bulk-bar{display:none;position:fixed;z-index:12;left:50%;bottom:18px;transform:translateX(-50%);align-items:center;justify-content:center;flex-wrap:wrap;max-width:calc(100vw - 36px);gap:9px;background:#152a2e;border:1px solid #55c9d6;border-radius:10px;padding:10px 14px;box-shadow:0 8px 30px #000}.selection-mode .bulk-bar{display:flex}.archive-button{background:#17627a;border-color:#56bad5;font-weight:700}.safe-button{background:#176b50;border-color:#55d3a5;font-weight:700}.archive-options{display:none;background:#162529;padding:12px;border-radius:7px}.archive-options label{display:block;margin:8px 0}.archive-options select,.archive-options input{display:block;width:100%;box-sizing:border-box;margin-top:5px}</style>''',
).replace(
    "c.onclick=()=>openPhoto(x.id);",
    "attachSelection(c,x);",
).replace(
    "</header>",
    r'''<button id=selectModeButton onclick=toggleSelectionMode()>Select photos</button></header>''',
).replace(
    "</body>",
    r'''<div id=bulkBar class=bulk-bar><strong id=selectedCount>0 selected</strong><button class=safe-button onclick="bulkSetNsfw(0)">Mark safe</button><button class=danger-button onclick="bulkSetNsfw(1)">Mark NSFW</button><button class=archive-button onclick="requestBulkAction('archive')">Archive selected</button><button class=danger-button onclick="requestBulkAction('delete')">Delete selected</button><button onclick=clearSelection()>Done</button></div><div id=bulkDialog class=danger-dialog role=dialog aria-modal=true aria-labelledby=bulkTitle><div class=danger-box><h2 id=bulkTitle></h2><p id=bulkMessage></p><div id=archiveOptions class=archive-options><label>Existing archive<select id=archiveExisting></select></label><label>Or create a new archive<input id=archiveNew placeholder="Example: Vacation.zip"></label><div class=muted>Leave the new name blank to use the selected existing archive. The default is GeneralArchive.zip.</div></div><div id=bulkError class=save-state></div><div class=danger-actions><button onclick=cancelBulkAction()>Cancel</button><button id=bulkConfirmButton onclick=executeBulkAction()></button></div></div></div><script>
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
async function bulkSetNsfw(value) {
  if (!selectedPhotos.size) return;
  const ids = [...selectedPhotos.keys()];
  const preservedScroll = window.scrollY;
  try { await post('/api/nsfw-override', {ids, value}); }
  catch (error) { alert(`NSFW update failed: ${error.message}`); return; }
  const filter = contentFilter.value;
  const remove = filter === 'conflict'
    || (value === 0 && filter === 'nsfw')
    || (value === 1 && hideNsfw.checked && !['all','nsfw'].includes(filter));
  if (remove) ids.forEach(removeDeletedCard);
  clearSelection();
  requestAnimationFrame(() => window.scrollTo(0, Math.min(
    preservedScroll, Math.max(0, document.documentElement.scrollHeight - window.innerHeight)
  )));
}
async function requestBulkAction(action) {
  if (!selectedPhotos.size) return;
  pendingBulkAction = action;
  const count = selectedPhotos.size;
  bulkError.textContent = '';
  bulkConfirmButton.disabled = false;
  if (action === 'delete') {
    archiveOptions.style.display = 'none';
    bulkTitle.textContent = `Danger: permanently delete ${count} photo${count === 1 ? '' : 's'}?`;
    bulkMessage.innerHTML = '<strong>This action cannot be undone.</strong> The selected source files and their catalog records will be permanently removed.';
    bulkConfirmButton.textContent = 'Delete permanently';
    bulkConfirmButton.className = 'danger-button';
  } else {
    archiveOptions.style.display = 'block';
    archiveNew.value = '';
    const archives = await api('/api/archives');
    archiveExisting.innerHTML = archives.map(name => `<option value="${esc(name)}">${esc(name)}</option>`).join('');
    bulkTitle.textContent = `Archive ${count} photo${count === 1 ? '' : 's'}?`;
    bulkMessage.textContent = 'Choose an existing archive or enter a new archive name. If no destination is designated, GeneralArchive.zip is used.';
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
    const payload = {ids};
    if (action === 'archive') payload.archive_name = archiveNew.value.trim() || archiveExisting.value || 'GeneralArchive.zip';
    await post(action === 'archive' ? '/api/archive-photos' : '/api/delete-photos', payload);
    ids.forEach(removeDeletedCard);
    await people(true);
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


PAGE = PAGE.replace(
    "</style>",
    r'''.face-entry,.person-tag,.pet-tag{display:grid;grid-template-columns:1fr auto;gap:7px;align-items:center}.face-assignment{grid-column:1/2;display:flex;gap:7px}.face-assignment input{margin:0!important;min-width:0;flex:1}.face-assignment button{white-space:nowrap}.face-entry .remove-face{grid-column:2;background:#70242a;border-color:#bd5058}.person-tag-actions,.pet-tag-actions{grid-column:2;display:flex;gap:6px}.person-tag-actions .remove-tag,.pet-tag-actions .remove-tag{background:#70242a;border-color:#bd5058}.add-person-tag,.add-pet-tag{display:flex;gap:7px;margin-top:9px;flex-wrap:wrap}.add-person-tag select{margin:0!important;flex:1}.add-pet-tag input{margin:0!important;flex:1;min-width:100px}.viewer.manual-targeting img{cursor:crosshair}.viewer.manual-targeting .face-reticle{pointer-events:none}.empty-faces{padding:8px 0;color:#aaa}</style>''',
).replace(
    "</body>",
    r'''<script>
const openPhotoWithFaceTags = openPhoto;
let pendingManualTagIdentityId = null, pendingPetTarget = null;
openPhoto = async function(id) { pendingManualTagIdentityId = null; pendingPetTarget = null; full.parentElement.classList.remove('manual-targeting'); await openPhotoWithFaceTags(id); renderFaceTagControls(); };
function assignmentBaseLabel(suggestion) { return `${suggestion.name} (#${suggestion.id})`; }
function assignmentMatchLabel(suggestion) { return `${suggestion.same_day?'Same day · ':''}${suggestion.match_score==null?'No biometric profile':`${(suggestion.match_score*100).toFixed(1)}% match`}`; }
function htmlAttribute(value) { return esc(value).replaceAll('"','&quot;'); }
function faceDisplayName(face) { const number = Math.max(1,(current.faces || []).findIndex(item=>item.id===face.id)+1); return face.name || (face.unknown ? `Unknown person ${number}` : `Unprocessed face ${number}`); }
function renderFaceAssignment(face) {
  const inputId = `faceAssignment${face.id}`;
  const suggestions = face.assignment_suggestions || identities.map(identity => ({...identity,same_day:false,match_score:null}));
  return `<div class="face-assignment"><input id="${inputId}" list="${inputId}Options" placeholder="Type or choose an identity" onfocus="activateFaceReticle(${face.id})" onkeydown="if(event.key==='Enter'){event.preventDefault();assignFaceFromInput(${face.id},'${inputId}')}" aria-label="Assign ${htmlAttribute(faceDisplayName(face))}"><datalist id="${inputId}Options">${suggestions.map(suggestion=>`<option value="${htmlAttribute(assignmentBaseLabel(suggestion))}" label="${htmlAttribute(assignmentMatchLabel(suggestion))}"></option>`).join('')}</datalist><button onclick="assignFaceFromInput(${face.id},'${inputId}')">Assign</button></div>`;
}
async function assignFaceFromInput(faceId, inputId) {
  const input = document.getElementById(inputId); const value = input.value.trim(); if (!value) return;
  const face = (current.faces || []).find(item => item.id === faceId); const suggestions = face?.assignment_suggestions || [];
  let identity = suggestions.find(item => assignmentBaseLabel(item).toLocaleLowerCase() === value.toLocaleLowerCase());
  if (!identity) { const matches = identities.filter(item => item.name.toLocaleLowerCase() === value.toLocaleLowerCase()); if (matches.length === 1) identity = matches[0]; }
  let identityId = identity?.id;
  if (!identityId) {
    if (!confirm(`Create a new identity named "${value}" and assign this face to it?`)) return;
    const created = await post('/api/create-identity',{name:value}); identityId = created.id; await people(true);
  }
  await assignFace(faceId, identityId);
}
function renderFaceTagControls() {
  const detected = current.faces || [];
  const tags = current.face_tags || [];
  faces.innerHTML = detected.length ? detected.map(face => `<div id="faceEntry${face.id}" class="face-entry"><span>${esc(faceDisplayName(face))}${face.art?' · artwork':''}${face.estimated_age!=null?' · age '+face.estimated_age:''}</span>${renderFaceAssignment(face)}<button class="remove-face" onclick="removeDetectedFace(${face.id})">Not a face / remove</button></div>`).join('') : '<div class="empty-faces">No faces detected.</div>';
  faces.insertAdjacentHTML('beforeend', `<button id="showFaceTagsButton" class="show-tags-button" onclick="toggleFaceTags()" ${detected.some(face=>face.bbox)||tags.some(tag=>tag.target_x!=null)||(current.pet_tags||[]).length?'':'disabled'}>${faceTagsVisible?'Hide tags':'Show tags'}</button>`);
  let tagBox = document.getElementById('personTags');
  if (!tagBox) { tagBox = document.createElement('div'); tagBox.id = 'personTags'; faces.insertAdjacentElement('afterend', tagBox); }
  tagBox.innerHTML = '<h3>Manual person tags</h3>' + (tags.length ? tags.map(tag => `<div id="manualTag${tag.identity_id}" class="person-tag"><span>${esc(tag.name)} <span class="muted">(non-biometric${tag.target_x==null?'':', targeted'})</span></span><div class="person-tag-actions"><button onclick="startManualPersonTarget(${tag.identity_id})">${tag.target_x==null?'Target':'Retarget'}</button><button class="remove-tag" onclick="removePersonTag(${tag.identity_id})">Remove tag</button></div></div>`).join('') : '<div class="empty-faces">No manual person tags.</div>') + `<div class="add-person-tag"><select id="newPersonTag"><option value="">Choose a person…</option>${identities.map(person=>`<option value="${person.id}">${esc(person.name)}</option>`).join('')}</select><button onclick="addPersonTag()">Add tag</button><button onclick="targetNewPersonTag()">Target on photo</button></div>`;
  let petBox = document.getElementById('petTags'); if (!petBox) { petBox=document.createElement('div'); petBox.id='petTags'; tagBox.insertAdjacentElement('afterend',petBox); }
  const pets = current.pet_tags || [];
  petBox.innerHTML = '<h3>Pets</h3><button onclick="detectPets()">Auto-detect cats and dogs</button>' + (pets.length ? pets.map(pet=>`<div class="pet-tag"><input value="${htmlAttribute(pet.name)}" aria-label="Pet name" onchange="renamePet(${pet.id},this.value)"><span class="muted">${esc(pet.species)}${pet.confidence==null?'':` · ${(pet.confidence*100).toFixed(1)}%`}${pet.automatic?' · automatic':''}</span><div class="pet-tag-actions"><button onclick="startPetRetarget(${pet.id})">Retarget</button><button class="remove-tag" onclick="removePet(${pet.id})">Remove</button></div></div>`).join('') : '<div class="empty-faces">No pets tagged.</div>') + '<div class="add-pet-tag"><input id="newPetName" placeholder="Pet name"><select id="newPetSpecies"><option>Dog</option><option>Cat</option><option>Other pet</option></select><button onclick="startNewPetTarget()">Target on photo</button></div>';
}
async function removeDetectedFace(faceId) {
  if (!confirm('Mark this detection as not a face and remove its person assignment?')) return;
  await post('/api/face', {face_id:faceId,action:'remove'}); await openPhoto(current.id);
  saveState.textContent = 'False face detection removed';
}
async function addPersonTag() {
  const identityId = +document.getElementById('newPersonTag').value; if (!identityId) return;
  await post('/api/photo-identity-tag', {image_id:current.id,identity_id:identityId}); await openPhoto(current.id);
  saveState.textContent = 'Person tag added (not used for biometric matching)';
}
function targetNewPersonTag() { const identityId = +document.getElementById('newPersonTag').value; if (identityId) startManualPersonTarget(identityId); }
function startManualPersonTarget(identityId) {
  const identity = identities.find(item=>item.id===identityId); if (!identity) return;
  pendingManualTagIdentityId = identityId; full.parentElement.classList.add('manual-targeting');
  saveState.textContent = `Click ${identity.name}'s location in the photo, or press Escape to cancel.`;
}
function cancelManualPersonTarget() { pendingManualTagIdentityId = null; full.parentElement.classList.remove('manual-targeting'); saveState.textContent = 'Manual tag targeting cancelled'; }
full.addEventListener('click', async event => {
  if (!pendingManualTagIdentityId) return;
  const bounds = full.getBoundingClientRect();
  const targetX = Math.max(0,Math.min(1,(event.clientX-bounds.left)/bounds.width));
  const targetY = Math.max(0,Math.min(1,(event.clientY-bounds.top)/bounds.height));
  const identityId = pendingManualTagIdentityId, imageId = current.id;
  pendingManualTagIdentityId = null; full.parentElement.classList.remove('manual-targeting');
  await post('/api/photo-identity-tag',{image_id:imageId,identity_id:identityId,target_x:targetX,target_y:targetY});
  await openPhoto(imageId); faceTagsVisible = true; renderFaceTagControls(); renderFaceReticles();
  saveState.textContent = 'Targeted manual person tag saved (not used for biometric matching)';
});
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&pendingManualTagIdentityId)cancelManualPersonTarget();if(event.key==='Escape'&&pendingPetTarget){pendingPetTarget=null;full.parentElement.classList.remove('manual-targeting');saveState.textContent='Pet targeting cancelled';}});
function startNewPetTarget() { const name=newPetName.value.trim(); if (!name) { saveState.textContent='Enter a pet name first.'; return; } pendingPetTarget={name,species:newPetSpecies.value}; full.parentElement.classList.add('manual-targeting'); saveState.textContent=`Click ${name}'s location in the photo.`; }
function startPetRetarget(id) { const pet=(current.pet_tags||[]).find(item=>item.id===id); if (!pet) return; pendingPetTarget={id,name:pet.name,species:pet.species}; full.parentElement.classList.add('manual-targeting'); saveState.textContent=`Click ${pet.name}'s location in the photo.`; }
full.addEventListener('click',async event=>{ if (!pendingPetTarget) return; const bounds=full.getBoundingClientRect(); const x=Math.max(0,Math.min(.88,(event.clientX-bounds.left)/bounds.width-.06)), y=Math.max(0,Math.min(.88,(event.clientY-bounds.top)/bounds.height-.06)); const target=pendingPetTarget,imageId=current.id; pendingPetTarget=null; full.parentElement.classList.remove('manual-targeting'); if(target.id) await post('/api/pet-tag',{action:'update',id:target.id,bbox:[x,y,.12,.12]}); else await post('/api/pet-tag',{image_id:imageId,name:target.name,species:target.species,bbox:[x,y,.12,.12]}); await openPhoto(imageId); faceTagsVisible=true; renderFaceReticles(); saveState.textContent='Pet tag saved'; });
async function detectPets() { saveState.textContent='Detecting cats and dogs locally…'; const result=await post('/api/detect-pets',{image_id:current.id}); await openPhoto(current.id); faceTagsVisible=true; renderFaceReticles(); saveState.textContent=`Detected ${result.count} pet${result.count===1?'':'s'}`; }
async function renamePet(id,value) { await post('/api/pet-tag',{action:'update',id,name:value}); saveState.textContent='Pet name saved'; }
async function removePet(id) { await post('/api/pet-tag',{action:'remove',id}); await openPhoto(current.id); saveState.textContent='Pet tag removed'; }
async function removePersonTag(identityId) {
  await post('/api/photo-identity-tag', {image_id:current.id,identity_id:identityId,action:'remove'}); await openPhoto(current.id);
  saveState.textContent = 'Person tag removed';
}
</script></body>''',
)

PAGE = PAGE.replace(
    "identity_id:person.value,year:year.value",
    "identity_id:person.value,unknown_group_id:unknownGroupFilter,year:year.value",
)

PAGE = PAGE.replace(
    "async function people(){identities=await api('/api/identities');for(let p of identities)",
    "async function people(refresh=false){identities=await api('/api/identities');if(refresh){person.innerHTML='<option value=\"\">All people</option>'}for(let p of identities.filter(item=>item.has_library_photos))",
).replace(
    "o.textContent=p.name;",
    "o.textContent=`${p.name} (#${p.id})`;",
).replace(
    "${esc(p.name)}</option>",
    "${esc(p.name)} (#${p.id})</option>",
).replace(
    "${esc(person.name)}</option>",
    "${esc(person.name)} (#${person.id})</option>",
).replace(
    "</body>",
    r'''<div id=mergeIdentityDialog class=danger-dialog role=dialog aria-modal=true aria-labelledby=mergeIdentityTitle><div class=danger-box><h2 id=mergeIdentityTitle>Merge identities?</h2><p id=mergeIdentitySummary></p><label>Identity to keep<select id=mergeIdentityTarget></select></label><p><strong>The removed identity IDs cannot be restored.</strong></p><div id=mergeIdentityError class=save-state></div><div class=danger-actions><button onclick=cancelIdentityMerge()>Cancel</button><button id=confirmIdentityMergeButton class=danger-button onclick=confirmIdentityMerge()>Merge identities</button></div></div></div></body>''',
)

PAGE = PAGE.replace(
    '<div class=viewer><img id=full>',
    '<div class=viewer><img id=full><div id=faceReticleLayer class=face-reticle-layer></div>',
).replace(
    "</style>",
    r'''.viewer{position:relative}.face-reticle-layer{position:absolute;inset:0;pointer-events:none;display:none}.face-reticle{position:absolute;border:3px solid var(--reticle-color);box-sizing:border-box;filter:drop-shadow(0 1px 2px #000);pointer-events:auto;cursor:pointer}.face-reticle:before,.face-reticle:after{content:'';position:absolute;background:var(--reticle-color)}.face-reticle:before{width:18px;height:2px;left:50%;top:50%;transform:translate(-50%,-50%)}.face-reticle:after{width:2px;height:18px;left:50%;top:50%;transform:translate(-50%,-50%)}.face-reticle.recognized{--reticle-color:#35e287}.face-reticle.unknown{--reticle-color:#a4aeb2}.face-reticle.unprocessed{--reticle-color:#ff4f5f}.face-reticle.active{--reticle-color:#21d9ee}.face-reticle.manual-tag{--reticle-color:#ffd65a;pointer-events:none}.face-reticle.pet{--reticle-color:#b978ff;pointer-events:none}.face-reticle-label{position:absolute;left:-3px;top:-27px;max-width:220px;padding:3px 6px;background:var(--reticle-color);color:#07110d;border-radius:4px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.show-tags-button{margin:8px 0;width:100%}</style>''',
).replace(
    "</body>",
    r'''<script>
let faceTagsVisible = false, activeFaceReticleId = null;
const openPhotoWithReticles = openPhoto;
openPhoto = async function(id) { faceTagsVisible = false; activeFaceReticleId = null; faceReticleLayer.style.display = 'none'; await openPhotoWithReticles(id); };
function activateFaceReticle(faceId) {
  activeFaceReticleId = faceId; faceTagsVisible = true;
  const button = document.getElementById('showFaceTagsButton'); if (button) button.textContent = 'Hide tags';
  renderFaceReticles();
}
function focusFaceAssignment(faceId) {
  activateFaceReticle(faceId);
  const input = document.getElementById(`faceAssignment${faceId}`);
  if (input) { input.scrollIntoView({behavior:'smooth',block:'center'}); input.focus({preventScroll:true}); }
}
function toggleFaceTags() {
  faceTagsVisible = !faceTagsVisible;
  const button = document.getElementById('showFaceTagsButton');
  if (button) button.textContent = faceTagsVisible ? 'Hide tags' : 'Show tags';
  renderFaceReticles();
}
function renderFaceReticles() {
  faceReticleLayer.innerHTML = '';
  faceReticleLayer.style.display = faceTagsVisible ? 'block' : 'none';
  if (!faceTagsVisible || !full.naturalWidth || !current) return;
  const imageRect = full.getBoundingClientRect(), viewerRect = full.parentElement.getBoundingClientRect();
  const scaleX = imageRect.width / full.naturalWidth, scaleY = imageRect.height / full.naturalHeight;
  for (const face of (current.faces || []).filter(item => item.bbox)) {
    const [x,y,width,height] = face.bbox;
    const reticle = document.createElement('div');
    reticle.className = 'face-reticle ' + (face.name ? 'recognized' : (face.unknown ? 'unknown' : 'unprocessed')) + (face.id===activeFaceReticleId?' active':'');
    reticle.setAttribute('role','button'); reticle.tabIndex = 0; reticle.setAttribute('aria-label',`Focus assignment for ${faceDisplayName(face)}`);
    reticle.onclick = () => focusFaceAssignment(face.id); reticle.onkeydown = event => { if (event.key==='Enter' || event.key===' ') { event.preventDefault(); focusFaceAssignment(face.id); } };
    reticle.style.left = `${imageRect.left-viewerRect.left+x*scaleX}px`;
    reticle.style.top = `${imageRect.top-viewerRect.top+y*scaleY}px`;
    reticle.style.width = `${Math.max(18,width*scaleX)}px`; reticle.style.height = `${Math.max(18,height*scaleY)}px`;
    const label = document.createElement('span'); label.className = 'face-reticle-label';
    label.textContent = faceDisplayName(face);
    reticle.append(label); faceReticleLayer.append(reticle);
  }
  for (const tag of (current.face_tags || []).filter(item=>item.target_x!=null&&item.target_y!=null)) {
    const reticle = document.createElement('div'); reticle.className = 'face-reticle manual-tag';
    reticle.style.left = `${imageRect.left-viewerRect.left+tag.target_x*imageRect.width-22}px`;
    reticle.style.top = `${imageRect.top-viewerRect.top+tag.target_y*imageRect.height-22}px`;
    reticle.style.width = '44px'; reticle.style.height = '44px';
    const label = document.createElement('span'); label.className = 'face-reticle-label'; label.textContent = `${tag.name} · manual tag`;
    reticle.append(label); faceReticleLayer.append(reticle);
  }
  for (const pet of (current.pet_tags || [])) {
    const [x,y,width,height]=pet.bbox, reticle=document.createElement('div'); reticle.className='face-reticle pet';
    reticle.style.left=`${imageRect.left-viewerRect.left+x*imageRect.width}px`; reticle.style.top=`${imageRect.top-viewerRect.top+y*imageRect.height}px`;
    reticle.style.width=`${Math.max(24,width*imageRect.width)}px`; reticle.style.height=`${Math.max(24,height*imageRect.height)}px`;
    const label=document.createElement('span'); label.className='face-reticle-label'; label.textContent=`${pet.name} · ${pet.species}`; reticle.append(label); faceReticleLayer.append(reticle);
  }
}
full.addEventListener('load', renderFaceReticles);
window.addEventListener('resize', renderFaceReticles);
</script></body>''',
)

PAGE = PAGE.replace(
    '<input id=identitySearch placeholder="Search identities or IDs" oninput=renderIdentityTable()>',
    '<input id=identitySearch placeholder="Search identities or IDs" oninput=renderIdentityTable()><label id=identitySortLabel>Sort <select id=identitySort onchange=renderIdentityTable()><option value=last_seen>Date last seen</option><option value=name>Name</option><option value=photo_count>Number of photos</option></select></label>',
).replace(
    '<th>Name</th><th>Reference images</th>',
    '<th>Name</th><th>Last seen</th><th>Reference images</th>',
).replace(
    "for (const identity of identityManagementRows.filter(item => item.name.toLowerCase().includes(query) || String(item.id).includes(query))) {",
    """const sortedIdentities = identityManagementRows.filter(item => item.name.toLowerCase().includes(query) || String(item.id).includes(query));
  sortedIdentities.sort((left,right) => identitySort.value === 'name'
    ? left.name.localeCompare(right.name,undefined,{sensitivity:'base'}) || left.id-right.id
    : identitySort.value === 'photo_count'
      ? right.photo_count-left.photo_count || left.name.localeCompare(right.name) || left.id-right.id
      : (right.last_seen||'').localeCompare(left.last_seen||'') || left.name.localeCompare(right.name) || left.id-right.id);
  for (const identity of sortedIdentities) {""",
).replace(
    "nameCell.append(nameInput);\n    const referenceCell",
    "nameCell.append(nameInput);\n    row.insertCell().textContent = identity.last_seen || 'Unknown';\n    const referenceCell",
).replace(
    "mergeIdentityButton.style.display = kind === 'recognized' ? '' : 'none';",
    "mergeIdentityButton.style.display = kind === 'recognized' ? '' : 'none';\n  identitySortLabel.style.display = '';",
).replace(
    '<th>Photos</th><th>Faces</th><th>Timeline</th>',
    '<th>Photos</th><th>Faces</th><th>Add to known identity</th><th>Timeline</th>',
).replace(
    '<div id=unidentifiedPeoplePanel class=identity-table-wrap style="display:none"><table',
    '<div id=unidentifiedPeoplePanel class=identity-table-wrap style="display:none"><p class=muted style="padding:0 10px">Similar unidentified people are grouped by biometric resemblance for review.</p><table',
).replace(
    '<th>Anonymous group</th><th>Appearances</th>',
    '<th>Anonymous group</th><th>Last seen</th><th>Appearances</th>',
).replace(
    '</style>',
    r'''.reference-item{display:flex;flex:0 0 auto;width:64px}.reference-item .reference-thumb{width:64px}.reference-actions{display:flex;justify-content:center;margin-top:16px}.reference-not-face{background:#9d1c25;border-color:#ef5963;color:#fff;font-weight:700}.reference-not-face:hover{background:#c32632}</style>''',
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
                unknown_group = query.get("unknown_group_id", [""])[0]
                unknown_groups = tuple(
                    int(value) for value in unknown_group.split(",") if value.strip()
                )
                year = query.get("year", [""])[0]
                self._json(catalog.gallery_photos(
                    query.get("q", [""])[0], int(identity) if identity else None,
                    int(year) if year else None, int(query.get("limit", ["100"])[0]),
                    int(query.get("offset", ["0"])[0]), query.get("nsfw", ["all"])[0],
                    tuple(filter(None, query.get("exclude_kinds", [""])[0].split(","))),
                    query.get("media_kind", [""])[0] or None,
                    unknown_group_ids=unknown_groups,
                ))
            elif parsed.path == "/api/photo":
                photo = catalog.gallery_photo(int(query["id"][0]))
                self._json(photo if photo else {"error": "not found"}, 200 if photo else 404)
            elif parsed.path == "/api/identities":
                visible_ids = catalog.gallery_identity_ids()
                self._json([{"id": x.identity_id, "name": x.name, "birth_year": x.birth_year,
                             "has_library_photos": x.identity_id in visible_ids}
                            for x in catalog.identities()])
            elif parsed.path == "/api/identity-summaries":
                self._json(catalog.identity_summaries())
            elif parsed.path == "/api/unidentified-summaries":
                self._json(catalog.unidentified_summaries())
            elif parsed.path == "/api/identity-reference":
                preview = catalog.identity_reference_preview(int(query["id"][0]))
                if preview is None: self.send_error(404); return
                self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(preview)))
                self.send_header("Cache-Control", "private, max-age=3600")
                self.end_headers(); self.wfile.write(preview)
            elif parsed.path == "/api/unidentified-reference":
                preview = catalog.unidentified_reference_preview(int(query["id"][0]))
                if preview is None: self.send_error(404); return
                self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(preview)))
                self.send_header("Cache-Control", "private, max-age=3600")
                self.end_headers(); self.wfile.write(preview)
            elif parsed.path == "/api/archives":
                self._json(available_archives())
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
                try:
                    action = str(body.get("action", "assign"))
                    if action == "remove": catalog.remove_face(int(body["face_id"]))
                    else: catalog.assign_face(int(body["face_id"]), int(body["identity_id"]))
                finally: catalog.close()
                self._json({"ok": True})
            elif self.path == "/api/photo-identity-tag":
                catalog = self._catalog()
                try:
                    if body.get("action") == "remove":
                        catalog.remove_photo_identity_tag(int(body["image_id"]), int(body["identity_id"]))
                    else:
                        catalog.add_photo_identity_tag(
                            int(body["image_id"]), int(body["identity_id"]),
                            float(body["target_x"]) if body.get("target_x") is not None else None,
                            float(body["target_y"]) if body.get("target_y") is not None else None,
                        )
                finally: catalog.close()
                self._json({"ok": True})
            elif self.path == "/api/pet-tag":
                catalog = self._catalog()
                tag_id = None
                try:
                    action = str(body.get("action", "add"))
                    if action == "remove": catalog.remove_pet_tag(int(body["id"]))
                    elif action == "update":
                        bbox = tuple(float(value) for value in body["bbox"]) if body.get("bbox") else None
                        catalog.update_pet_tag(int(body["id"]), body.get("name"), bbox)  # type: ignore[arg-type]
                    else:
                        tag_id = catalog.add_pet_tag(
                            int(body["image_id"]), str(body.get("name", "Pet")),
                            str(body.get("species", "Pet")),
                            tuple(float(value) for value in body["bbox"]),  # type: ignore[arg-type]
                        )
                finally: catalog.close()
                self._json({"ok": True, "id": tag_id})
            elif self.path == "/api/detect-pets":
                from .pet_detector import PetDetector
                catalog = self._catalog()
                try:
                    image_id = int(body["image_id"]); path = catalog.image_path(image_id)
                    if path is None: raise ValueError("Photo was not found.")
                    detections = PetDetector(APP_ROOT / "models").detect(path)
                    catalog.clear_automatic_pet_tags(image_id)
                    for index, detection in enumerate(detections, 1):
                        catalog.add_pet_tag(
                            image_id, f"{detection.species} {index}", detection.species,
                            detection.bbox, detection.confidence, True,
                        )
                finally: catalog.close()
                self._json({"ok": True, "count": len(detections)})
            elif self.path == "/api/identity":
                catalog = self._catalog()
                try:
                    birth_year = body.get("birth_year")
                    catalog.update_identity(
                        int(body["id"]), str(body.get("name", "")),
                        int(birth_year) if birth_year not in (None, "") else None,
                    )
                finally: catalog.close()
                self._json({"ok": True})
            elif self.path == "/api/create-identity":
                catalog = self._catalog()
                try: identity_id = catalog.create_identity(str(body.get("name", "")))
                finally: catalog.close()
                self._json({"ok": True, "id": identity_id})
            elif self.path == "/api/merge-identities":
                catalog = self._catalog()
                try:
                    faces, tags = catalog.merge_identities(
                        [int(value) for value in body["identity_ids"]],
                        int(body["target_identity_id"]),
                    )
                finally: catalog.close()
                self._json({"ok": True, "faces": faces, "tags": tags})
            elif self.path == "/api/assign-unknown-groups":
                catalog = self._catalog()
                try:
                    faces, photos = catalog.assign_unknown_groups(
                        [int(value) for value in body["group_ids"]], int(body["identity_id"])
                    )
                finally: catalog.close()
                self._json({"ok": True, "faces": faces, "photos": photos})
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
            elif self.path == "/api/nsfw-override":
                catalog = self._catalog()
                try:
                    updated = catalog.set_nsfw_overrides(
                        [int(value) for value in body["ids"]], int(body["value"])
                    )
                finally: catalog.close()
                self._json({"ok": True, "updated": updated})
            elif self.path == "/api/archive-photos":
                catalog = self._catalog()
                try:
                    archived = catalog.archive_photos(
                        [int(value) for value in body["ids"]],
                        archive_path_for_name(body.get("archive_name")), PICTURES_ROOT
                    )
                finally: catalog.close()
                destination = archive_path_for_name(body.get("archive_name"))
                self._json({"ok": True, "archived": len(archived), "archive": str(destination)})
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
