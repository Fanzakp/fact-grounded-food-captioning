/* Static archive explorer. Uses only verified saved outputs, never invokes a model. */
(() => {
  'use strict';
  const data = window.DEMO_DATA;
  if (!data || !Array.isArray(data.rows) || data.rows.length !== 507) {
    console.error('Archive data is unavailable; showing pre-rendered preview instead.');
    return;
  }
  const rows = data.rows, models = data.models;
  const byId = new Map(rows.map(r => [r.id,r]));
  const cols = [
    ['Calories','kcal'],['Protein','g'],['Fat','g'],['Carbs','g'],['Mass','g']
  ];
  const state = {selected:'dish_1566416701',filter:'all',q:'',sort:'impact',a:'grounded',b:'conventional',limit:24};
  const idx = code => models.findIndex(m => m.code === code);
  const txt = v => String(v ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const score = (r, code) => r.n[idx(code)].reduce((acc,value,j) => acc + (value !== null && Math.abs(value-r.f[j]) <= 1 ? 1 : 0),0);
  const gain = r => score(r,'grounded')-score(r,'conventional');
  const calErr = r => Math.abs(r.f[0]-r.gt[0]);
  const nice = v => Number.isInteger(v)? String(v):Number(v).toFixed(1).replace(/\.0$/,'');
  const optionHTML = (chosen) => models.map(m => `<option value="${m.code}" ${m.code===chosen?'selected':''}>${txt(m.label)}</option>`).join('');
  const title = r => r.i.split(',')[0].trim().replace(/\band\b.*/, '').slice(0,46);

  // Photos are bundled as relative URLs; an embedded-image fallback is also supported.
  const imageSrc = r => {
    if (window.DEMO_IMAGES && window.DEMO_IMAGES[r.id]) {
      return 'data:image/webp;base64,' + window.DEMO_IMAGES[r.id];
    }
    return r.photo;
  };
  function gallery() {
    document.querySelectorAll('#gallery .gallery-item').forEach(el => {
      const selected = el.dataset.dish === state.selected;
      el.classList.toggle('active', selected);
      el.setAttribute('aria-pressed', String(selected));
    });
  }
  let allPhotoLimit = 48;
  let allPhotoExpanded = false;
  function renderFullPhotoGallery() {
    if (!allPhotoExpanded) return;
    const matches=sorted();
    const node=document.getElementById('photoGrid');
    document.getElementById('galleryPhotoCount').textContent = `${Math.min(allPhotoLimit,matches.length)} / ${matches.length}`;
    node.innerHTML=matches.slice(0,allPhotoLimit).map(r => `<button type="button" class="photo-grid-card ${r.id===state.selected?'active':''}" data-dish="${r.id}" aria-label="View ${txt(r.id)}"><img src="${imageSrc(r)}" loading="lazy" decoding="async" alt="Original Nutrition5k RGB image for ${txt(r.id)}"><span>${txt(title(r))}</span></button>`).join('');
    node.querySelectorAll('button[data-dish]').forEach(el=>el.addEventListener('click',()=>choose(el.dataset.dish)));
    document.getElementById('galleryLoadMore').hidden = allPhotoLimit >= matches.length;
  }
  function sorted() {
    const q = state.q.toLowerCase().trim();
    return rows.filter(r => (!q || r.i.toLowerCase().includes(q) || r.id.toLowerCase().includes(q)) &&
      (state.filter === 'all' ||
      (state.filter === 'photos' && !!r.photo) ||
      (state.filter === 'advantage' && gain(r)>0) ||
      (state.filter === 'caution' && score(r,'grounded')>=4 && calErr(r)>=30)))
      .sort((a,b) => state.sort === 'impact' ? gain(b)-gain(a) || calErr(b)-calErr(a) || a.id.localeCompare(b.id):
      state.sort === 'calError' ? calErr(b)-calErr(a) || a.id.localeCompare(b.id) :
      state.sort === 'name' ? a.i.localeCompare(b.i) || a.id.localeCompare(b.id) : a.id.localeCompare(b.id));
  }
  function list() {
    const items = sorted(), node=document.querySelector('#dishList');
    document.querySelector('#resultCount').textContent = `${items.length} of ${rows.length} records`;
    document.querySelector('#loadMore').hidden = items.length <= state.limit;
    if (!items.length) {node.innerHTML='<p class="empty-state">No matching dishes. Try a different ingredient, dish ID, or filter.</p>';return;}
    node.innerHTML=items.slice(0,state.limit).map(r => {
      const preview = `<img src="${imageSrc(r)}" alt="" loading="lazy" decoding="async">`;
      return `<button type="button" role="listitem" class="dish-item ${r.id===state.selected?'active':''}" data-dish="${r.id}" aria-pressed="${r.id===state.selected}"><span class="dish-mini">${preview}</span><span class="dish-content"><strong>${txt(r.i)}</strong><small>${txt(r.id)} · ${nice(r.gt[0])} kcal measured</small></span><span class="dish-gain">${gain(r)>0?'+':''}${gain(r)}</span></button>`;
    }).join('');
    node.querySelectorAll('button[data-dish]').forEach(el=>el.addEventListener('click',()=>choose(el.dataset.dish)));
  }
  function choose(id) {
    if (!byId.has(id)) return;
    state.selected=id;
    gallery();list();detail();
    if (allPhotoExpanded) document.querySelectorAll('.photo-grid-card').forEach(el=>el.classList.toggle('active',el.dataset.dish===id));
    if (window.innerWidth < 780) document.getElementById('detail').scrollIntoView({behavior:'smooth',block:'start'});
  }
  function captionPanel(r,code,side){
    const j=idx(code),m=models[j],s=score(r,code);
    return `<article class="caption-card ${side==='a'?'grounded-card':''}"><div class="caption-type"><h5>${txt(m.label)}</h5><span class="count">${s}/5 faithful</span></div><p>${txt(r.c[j])}</p><div class="caption-note">${s===5?'All five numeric fields match the supplied fact values.':`${5-s} field(s) missing or outside ±1 of supplied facts.`} Historical regex extraction.</div></article>`;
  }
  function attrCell(value,fact){
    if(value === null) return '<span class="score-omit" title="Not extracted by the historical parser">—</span>';
    const cls=Math.abs(value-fact)<=1?'score-ok':'score-bad';
    return `<span class="${cls}">${nice(value)}</span>`;
  }
  function detail() {
    const r=byId.get(state.selected);
    const node=document.getElementById('detail');
    const a=idx(state.a), b=idx(state.b);
    const alertNum = r.f.map((val,j)=>({field:cols[j][0],difference:Math.abs(val-r.gt[j]),unit:cols[j][1]})).sort((x,y)=>y.difference-x.difference)[0];
    const caution = alertNum.difference>=20 ? `<div class="analysis-note"><strong>Watch the distinction:</strong> For this dish, the supplied estimate differs from the measured ${txt(alertNum.field.toLowerCase())} by ${nice(alertNum.difference)} ${alertNum.unit}. A caption can faithfully repeat an inaccurate estimate.</div>` : `<div class="analysis-note"><strong>Interpretation:</strong> Green means the caption repeats the supplied estimate within ±1, not that the estimated nutrition is physically accurate.</div>`;
    const image = `<img src="${imageSrc(r)}" alt="Original Nutrition5k RGB image for ${r.id}" decoding="async"/><span class="photo-label">NUTRITION5K · REAL RGB INPUT</span>`;
    node.innerHTML=`
      <div class="detail-top"><div><p class="eyebrow">SELECTED TEST EXAMPLE</p><h3>${txt(title(r))}</h3></div><div class="detail-id"><span>${r.id}</span><button type="button" class="jump" id="random" title="Choose a random dish" aria-label="Choose random dish">⤨</button></div></div>
      <div class="photo-and-facts"><div class="big-photo">${image}</div>
      <div class="fact-box"><p class="box-kicker">SUPPLIED NUTRITION FACT STRING</p><p class="fact-heading">${txt(r.i)}</p><div class="nutrients-grid">
      ${r.f.map((v,j)=>`<div class="nutrition-tile"><b>${nice(v)} <em>${cols[j][1]}</em></b><small>${cols[j][0].toUpperCase()}</small></div>`).join('')}
      </div><div class="fact-under">These are rounded nutrition estimates passed into grounded caption models—not guaranteed ground-truth values.</div></div></div>
      <div class="comparison-head"><h4>Side-by-side captions</h4><small>ARCHIVED MODEL OUTPUTS</small></div>
      <div class="comparison-controls"><div><label for="modelA">MODEL A</label><select id="modelA">${optionHTML(state.a)}</select></div><div><label for="modelB">MODEL B</label><select id="modelB">${optionHTML(state.b)}</select></div></div>
      <div class="caption-grid">${captionPanel(r,state.a,'a')}${captionPanel(r,state.b,'b')}</div>
      <div class="comparison-head"><h4>Numbers under the microscope</h4><small>GREEN = MATCHES INPUT ±1</small></div>
      <div class="field-table-wrap"><table class="field-table"><thead><tr><th>ATTRIBUTE</th><th>SUPPLIED</th><th>MEASURED</th><th>MODEL A</th><th>MODEL B</th></tr></thead><tbody>
      ${cols.map(([name,unit],j)=>`<tr><td>${name} <span class="quiet">(${unit})</span></td><td>${nice(r.f[j])}</td><td>${nice(r.gt[j])}</td><td>${attrCell(r.n[a][j],r.f[j])}</td><td>${attrCell(r.n[b][j],r.f[j])}</td></tr>`).join('')}
      </tbody></table></div><p class="legend"><span class="ok">●</span> Within ±1 of supplied fact &nbsp;&nbsp; <span class="bad">●</span> Outside ±1 &nbsp;&nbsp; — Not parsed. “Measured” comes from Nutrition5k metadata.</p>${caution}`;
    node.querySelector('#modelA').addEventListener('change',e => {state.a=e.target.value;detail();});
    node.querySelector('#modelB').addEventListener('change',e => {state.b=e.target.value;detail();});
    node.querySelector('#random').addEventListener('click',()=>choose(rows[Math.floor(Math.random()*rows.length)].id));
  }
  function scores(){
    document.querySelector('#modelStats').innerHTML = models.map(m=>`<div class="model-row ${m.code==='grounded'?'primary':''}"><div class="model-name">${txt(m.label)}<small>${m.code==='grounded'?'PRIMARY RESEARCH MODEL':m.code.startsWith('conventional')?'IMAGE ONLY':m.code==='template'?'SHARED TEMPLATE RULE':m.code==='t5'?'TEXT ONLY':'IMAGE + FACTS'}</small></div><div class="progress-wrap"><div class="progress-track"><div class="progress-value" style="width:${m.nutacc}%"></div></div><b>${m.nutacc.toFixed(2)}%</b></div><span class="model-number">${m.bleu.toFixed(4)}</span><span class="model-number">${m.cider.toFixed(4)}</span></div>`).join('');
  }
  document.getElementById('search').addEventListener('input',e=>{state.q=e.target.value;state.limit=24;allPhotoLimit=48;list();renderFullPhotoGallery();});
  document.getElementById('sort').addEventListener('change',e=>{state.sort=e.target.value;state.limit=24;allPhotoLimit=48;list();renderFullPhotoGallery();});
  document.querySelectorAll('[data-filter]').forEach(btn=>btn.addEventListener('click',()=>{
    state.filter=btn.dataset.filter;state.limit=24;allPhotoLimit=48;
    document.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('active',b===btn));list();renderFullPhotoGallery();
  }));
  document.getElementById('loadMore').addEventListener('click',()=>{state.limit+=30;list();});
  document.querySelectorAll('#gallery .gallery-item').forEach(el => {
    el.dataset.dish=el.dataset.dish || el.getAttribute('href').replace('#preview-','');
    el.addEventListener('click',e=>{e.preventDefault();choose(el.dataset.dish);});
  });
  document.getElementById('photoGalleryToggle').addEventListener('click',()=>{
    allPhotoExpanded=!allPhotoExpanded;
    allPhotoLimit=48;
    document.getElementById('fullPhotoGallery').hidden=!allPhotoExpanded;
    document.getElementById('photoGalleryToggle').setAttribute('aria-expanded',String(allPhotoExpanded));
    document.getElementById('photoGalleryToggle').textContent=allPhotoExpanded?'Hide complete photo gallery ↑':'Browse all 507 photos ↗';
    renderFullPhotoGallery();
  });
  document.getElementById('galleryLoadMore').addEventListener('click',()=>{allPhotoLimit+=48;renderFullPhotoGallery();});
  gallery();list();detail();scores();
  document.body.classList.remove('no-js');
  document.body.classList.add('js-ready');
})();
