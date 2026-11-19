import {createScene,tint} from './renderer.mjs';
import {alignedView} from './alignment.mjs';
import {createLibrary} from './library.mjs';
import {parseXYZ,makePython} from './chemistry.mjs';
import {createAnalysis,formatTime} from './analysis.mjs';
import {glyph,mountGlyphs} from './glyphs.mjs';

const byId = name => document.getElementById(name);
const taskLabel = {energy:'Single-point energy',relax:'Geometry optimization',md:'Molecular dynamics',trajectory:'Imported trajectory'};
const atomNames = {C:'Carbon',H:'Hydrogen',O:'Oxygen',N:'Nitrogen',F:'Fluorine',S:'Sulfur',P:'Phosphorus',Cl:'Chlorine'};
const exampleRuns = [];
const sourceFrames = new Map();
const journal = [];
let activeJob=null;
let molecule, engine = false, working = false, shown = null;
let library, focusIndices=null;
let selectedAtom = null, playback = 0, toastTimeout = 0;
let smilesRequest = null, trajectoryRequest = null, hostStatus = null, engineCheck;

function node(name,text,cls) {
  const element = document.createElement(name);
  if (text!==undefined) element.textContent = text;
  if (cls) element.className = cls;
  return element;
}
function prettyFormula(text) {
  return (text==='H3N'?'NH3':text).replace(/\d/g,n=>'₀₁₂₃₄₅₆₇₈₉'[n]);
}
function flash(text) {
  clearTimeout(toastTimeout);
  byId('notice').textContent = text;
  byId('notice').hidden = false;
  toastTimeout = setTimeout(()=>byId('notice').hidden=true,4500);
}
function download(name,content,type) {
  const url = URL.createObjectURL(new Blob([content],{type}));
  const link = node('a');
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),5000);
}
async function request(route,options) {
  const response = await fetch(route,options);
  if (!response.ok) {
    let reason;
    try { reason = (await response.json()).error; } catch {}
    throw new Error(reason || 'Request failed ('+response.status+').');
  }
  return response.json();
}
mountGlyphs();
const scene = createScene(byId('structure'),{onSelect(index){selectedAtom=index==null?null:(focusIndices?.[index]??index);describeAtom();}});

const analysis=createAnalysis(byId('trajectory-analysis'),{
  onFrame(index){if(shown){haltPlayback();changeFrame(index);}},
});

function settings() {
  return {
    task:byId('method').value,
    model:document.querySelector('input[name=potential]:checked').value,
    charge:byId('charge').valueAsNumber,
    steps:byId(byId('method').value==='md'?'md-steps':'max-steps').valueAsNumber,
    temperature:byId('md-temp').valueAsNumber,
    timestep:byId('md-dt').valueAsNumber,
    device:byId('device').value,
    precision:byId('precision').value,
    fmax:byId('force-limit').valueAsNumber,
  };
}
function exampleFor(config) {
  if (!molecule || (molecule.source_frame??1)!==1) return;
  return exampleRuns.find(row=>{
    if(row.id!==molecule.id || row.task!==config.task || row.model!==config.model || row.charge!==config.charge)return false;
    if((row.result.precision??'mixed')!==config.precision)return false;
    if(config.device!=='auto' && config.device!==(row.result.device??'cpu'))return false;
    return row.task==='relax' && row.fmax===config.fmax && row.steps===config.steps;
  });
}
function canCalculate(structure) {
  const atoms=structure.atom_count??structure.elements?.length;
  return structure.calculation_allowed!==false && atoms>=2 && atoms<=256;
}
function updateAction() {
  if (!molecule || working) return;
  const config=settings(), saved=exampleFor(config);
  if(!canCalculate(molecule)){
    byId('action').disabled=true;byId('action').textContent='Structure preview';
    byId('python').hidden=true;
    byId('engine-note').textContent=molecule.calculation_note||'Local calculations support up to 256 atoms.';
    return;
  }
  const label=engine?(config.task==='md'?'Run dynamics':'Run calculation'):saved?'View example result':'Download Python job';
  byId('action').replaceChildren(glyph(engine||saved?'play':'code'),node('span',label));
  byId('action').disabled=false;
  byId('engine-note').textContent=engine?'Local engine':saved?'Precomputed example':'';
  byId('python').hidden=!engine&&!saved;
}
function describeAtom() {
  const popup=byId('atom-pick');
  popup.hidden=selectedAtom==null || !molecule;
  if (popup.hidden) return;
  const name=molecule.elements[selectedAtom];
  popup.textContent=(atomNames[name]||name)+' '+(selectedAtom+1);
}
function haltPlayback() {
  clearInterval(playback); playback=0;
  byId('animate').replaceChildren(glyph('play'));
  byId('animate').setAttribute('aria-label','Play trajectory');
  byId('animate').title='Play trajectory';
}
function clearOutput() {
  haltPlayback();
  shown=null; selectedAtom=null;analysis.clear();
  byId('save-trajectory').hidden=true;
  byId('view-status').hidden=false;
  byId('view-kind').textContent='Structure view';
  byId('dipole-control').hidden=true;byId('dipole-toggle').checked=false;
  byId('atom-pick').hidden=true; byId('result').hidden=true; byId('viewport-player').hidden=true; byId('job-error').hidden=true;
}
function configureMethod() {
  byId('charge').readOnly=!!molecule?.cell;
  const method=byId('method').value;
  for (const group of document.querySelectorAll('#setup [data-method]')) {
    const enabled=group.dataset.method===method;
    group.hidden=!enabled;
    group.querySelectorAll('input').forEach(input=>input.disabled=!enabled);
  }
  const steps=byId('md-steps').valueAsNumber, dt=byId('md-dt').valueAsNumber, duration=steps*dt;
  byId('duration').textContent=Number.isFinite(duration)?formatTime(duration):'—';
}
function visiblePositions(entry,coords,cell) {
  if(!focusIndices)return coords;
  return focusIndices.map((i,n)=>{
    const shift=entry.focus_images?.[n];
    return shift?coords[i].map((v,k)=>v+shift.reduce((sum,s,j)=>sum+s*cell[j][k],0)):coords[i];
  });
}
function displayStructure(entry){
  const focused=byId('structure-scope').value==='solute';
  const range=entry.focus_range;
  focusIndices=focused?(entry.focus_indices??(range?Array.from({length:range[1]-range[0]},(_,i)=>range[0]+i):null)):null;
  const elements=focusIndices?focusIndices.map(i=>entry.elements[i]):entry.elements, large=elements.length>2500;
  const mode=byId('mode');
  if(large)mode.value='points';
  mode.disabled=large;
  const cell=focusIndices?null:entry.cell;
  byId('cell-info').hidden=!entry.cell;
  byId('cell-info').textContent=focusIndices?(entry.focus_label??'Solute')+' · '+elements.length.toLocaleString()+' of '+entry.elements.length.toLocaleString()+' atoms':'Unit cell · periodic in 3D';
  scene.load(elements,visiblePositions(entry,entry.coordinates,entry.cell),cell);scene.style(mode.value);
  const counts=[...new Set(elements)].sort();
  byId('elements').replaceChildren(...counts.map(symbol=>{
    const element=node('span'),swatch=node('i');swatch.style.background=tint(symbol);
    element.title=atomNames[symbol]||symbol;
    element.append(swatch,document.createTextNode(symbol));return element;
  }));
  selectedAtom=null;describeAtom();
}
byId('structure-scope').onchange=()=>{displayStructure(molecule);if(shown)changeFrame(+byId('scrub').value);};
function loadMolecule(entry,config) {
  molecule=entry; clearOutput();
  byId('structure-title').textContent=entry.name;
  byId('selected-structure').textContent=entry.conformer_count>1
    ? entry.name+' · geometry '+(entry.source_frame??1)
    : entry.name;
  byId('structure-title').title=entry.source?.replace('Frame 1 of','Frame '+(entry.source_frame??1)+' of')||'';
  const bar=byId('source-bar'),framePicker=byId('frame-picker'),picker=byId('starting-frame'),sourceLink=byId('source-xyz');
  bar.hidden=!(entry.xyz_file||entry.structure_file||entry.source_url||entry.recording);
  sourceLink.hidden=!(entry.xyz_file||entry.structure_file);
  sourceLink.onclick=null;
  const description=byId('example-description');
  description.hidden=!entry.description;description.textContent=entry.description||'';
  if(entry.structure_file){
    sourceLink.href='#';sourceLink.removeAttribute('download');
    sourceLink.textContent='Save full structure';
    sourceLink.onclick=e=>{
      e.preventDefault();
      const header=['Properties=species:S:1:pos:R:3','pbc="'+(entry.cell?'T T T':'F F F')+'"'];
      if(entry.cell)header.push('Lattice="'+entry.cell.flat().join(' ')+'"');
      if(Number.isInteger(entry.charge))header.push('charge='+entry.charge);
      const body=entry.elements.map((el,i)=>[el,...entry.coordinates[i]].join(' ')).join('\n');
      download(entry.id+'.extxyz',entry.elements.length+'\n'+header.join(' ')+'\n'+body+'\n','text/plain');
    };
  }
  const source=getSourceLink(entry);
  const dataset=byId('dataset-source');dataset.hidden=!source;
  if(source){dataset.href=source;dataset.title=[entry.source||entry.description,entry.source_file].filter(Boolean).join(' · ')||'Dataset source';}
  framePicker.hidden=!(entry.conformer_count>1);
  const replay=byId('replay-peptide');
  replay.textContent=entry.recording_label??'Open 50 fs example';
  replay.hidden=!(entry.recording||(entry.id==='ala4' && (entry.source_frame??1)===1));
  if(entry.xyz_file){
    sourceLink.href=entry.xyz_file;sourceLink.setAttribute('download','');sourceLink.textContent='Source XYZ';
    if(entry.conformer_count>1){
      if(picker.options.length!==entry.conformer_count)picker.replaceChildren(...Array.from({length:entry.conformer_count},(_,i)=>new Option(String(i+1),String(i+1))));
      picker.value=entry.source_frame??1;
      byId('source-count').textContent='of '+entry.conformer_count;
    }
  }
  byId('molecular-info').replaceChildren(document.createTextNode(prettyFormula(entry.formula)),node('span','·'),document.createTextNode(entry.elements.length+' atoms'));
  byId('charge').value=config?.charge??entry.charge;
  byId('method').querySelector('[value=md]').disabled=!!entry.cell;
  const requestedTask=config?.task??entry.default_task;
  byId('method').value=requestedTask==='md' && !entry.cell?'md':'relax';
  byId('environment').closest('.property').hidden=!entry.cell;
  byId('scope-control').hidden=!(entry.focus_range||entry.focus_indices);
  byId('structure-scope').options[0].textContent=entry.focus_label??'Solute';
  byId('structure-scope').value=entry.focus_range||entry.focus_indices?'solute':'cell';
  byId('force-limit').value=config?.fmax??0.05;
  byId('max-steps').value=(config?.task==='relax'?config.steps:null)??entry.relax_steps??100;
  byId('md-steps').value=config?.task==='md'?(config.steps??100):100;
  byId('md-temp').value=config?.temperature??300;
  byId('md-dt').value=config?.timestep??byId('md-dt').defaultValue;
  if(config?.precision)byId('precision').value=config.precision;
  const radio=Array.from(document.querySelectorAll('input[name=potential]')).find(el=>el.value===config?.model);
  (radio??document.querySelector('input[name=potential][value="so3lr-tauri-m"]')).checked=true;
  setupDevices(hostStatus,config?.device??byId('device').value);
  configureMethod();
  displayStructure(entry);
  library?.select(entry.id);
  updateAction();
}
function getSourceLink(entry){
  return entry.source_url?.startsWith('https://')?entry.source_url:null;
}
function liveConfig(entry){
  if(entry.result.recording?.production_duration_fs || entry.config.steps>1000000)
    return {task:'md',steps:100,temperature:entry.config.temperature,model:entry.config.model};
  return entry.config;
}
async function recording(structure){
  const saved=structure.recording?await request(structure.recording):exampleRuns.find(row=>row.id===structure.id && row.task==='md' && row.steps===100);
  if(!saved)throw Error('The recorded calculation could not be loaded.');
  const {result,id,...config}=saved;
  const entry={result,config,structure,provenance:result.recording?.figure||structure.recording_provenance||'Recorded example'};
  loadMolecule(structure,liveConfig(entry));
  addResult(entry,0);
}
function buildFromSmiles(smiles,signal) {
  return request('api/smiles',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({smiles}),signal});
}
async function selectExample(item,{replay=!!item.recording,scroll=true,play=false}={}){
  if(working)return;
  lock(true);
  try{
    let structure=item.structure_file?{...item,...await request(item.structure_file)}:item;
    if(item.smiles && (await engineCheck)?.engine==='so3lr'){
      const built=await buildFromSmiles(item.smiles);
      structure={...item,...built,id:item.id,name:item.name};
    }
    if(replay)await recording(structure);else loadMolecule(structure);
    if(scroll)byId('viewer').scrollIntoView({block:'start',behavior:'smooth'});
    if(play && shown?.result.frames.length>1 && !document.hidden && !matchMedia('(prefers-reduced-motion: reduce)').matches)byId('animate').click();
  }catch(error){flash(error.message);}
  finally{lock(false);}
}
byId('replay-peptide').onclick=()=>selectExample(molecule,{replay:true,scroll:false});
document.addEventListener('visibilitychange',()=>{if(document.hidden)haltPlayback();});
async function saveScript() {
  if (!molecule || !canCalculate(molecule) || !byId('setup').reportValidity()) return;
  const config=settings(), structure={...molecule,charge:config.charge};
  try{
    let kernel='';
    if(structure.cell){
      const response=await fetch('periodic_job.py');
      if(!response.ok)throw Error('Could not load the periodic calculation source.');
      kernel=await response.text();
    }else if(config.task==='md'){
      const response=await fetch('native_md.py');
      if(!response.ok)throw Error('Could not load the MD calculation source.');
      kernel=await response.text();
    }
    const content=makePython(structure,config.model,config.task,config,kernel);
    download(structure.id+'-'+config.task+'.py',content,'text/x-python');
    flash('Python job downloaded.');
  }catch(error){flash(error.message);}
}
function resultModel(config) {return ({'so3lr-tauri-s':'Tauri S','so3lr-tauri-m':'Tauri M','so3lr-tauri-l':'Tauri L'})[config.model]??(config.model||'');}
function metric(label,value) {
  const cell=node('div');cell.append(node('small',label),node('strong',value));return cell;
}
function changeFrame(index) {
  const data=shown.result;
  byId('scrub').value=index;
  const vector=data.dipoles_debye?.[index];
  byId('dipole-control').hidden=!vector;
  byId('dipole-label').textContent=vector?'Dipole · '+Math.hypot(...vector).toFixed(2)+' D':'Dipole';
  const coords=data.frames[index];
  const cell=data.cells?.[index]??molecule.cell;
  const stabilize=molecule.id==='ala15_folded'||data.analysis?.peptide?.residue_count===15;
  const view=stabilize?alignedView(coords,data.frames[0],molecule.elements,vector)
    :{positions:visiblePositions(molecule,coords,cell),vector};
  scene.positions(view.positions,byId('dipole-toggle').checked?view.vector:null,focusIndices?null:cell);
  analysis.frame(index);
  const step=data.frame_steps?.[index]??index, last=data.frame_steps?.at(-1)??(data.frames.length-1);
  const time=data.times_fs?.[index];
  const frameLabel=shown.config.task==='trajectory'?'Frame '+(index+1)+' / '+data.frames.length:'Step '+step+' / '+last;
  byId('frame-index').textContent=Number.isFinite(time)?formatTime(time)+' / '+formatTime(data.times_fs.at(-1)):frameLabel;
  describeAtom();
}
function addResult(entry,initialFrame) {
  journal.unshift(entry);
  if(journal.length>16)journal.pop();
  openResult(entry,initialFrame);
}
function openResult(entry,initialFrame=entry.result.frames.length-1) {
  haltPlayback();shown=entry;byId('result').hidden=false;
  const {result:r,config:c}=entry, multi=r.frames.length>1;
  const ala15Replay=entry.structure.id==='ala15_folded' && !!r.recording?.production_duration_fs;
  byId('view-status').hidden=ala15Replay;
  byId('view-kind').textContent=r.periodic?.optimizer?'Periodic optimization':'Structure view';
  const interval=r.times_fs?.[1]-r.times_fs?.[0];
  const uniform=multi && Number.isFinite(interval) && r.times_fs.every((t,i)=>Math.abs(t-r.times_fs[0]-i*interval)<1e-5);
  const sampling=multi?r.frames.length.toLocaleString()+' saved frames'+(uniform?' every '+formatTime(interval):''):'';
  const provenance=byId('provenance');
  byId('result-heading').hidden=entry.structure.id==='ala15_folded' && c.task==='md';
  provenance.textContent=[entry.provenance,entry.structure.conformer_count>1?'Geometry '+(entry.structure.source_frame??1):'',resultModel(c),r.md?.integrator,r.periodic?.optimizer,sampling].filter(Boolean).join(' · ');
  byId('result-title').textContent=entry.structure.name+' · '+taskLabel[c.task];
  const metrics=[];
  if(Number.isFinite(r.energy))metrics.push(metric(c.task==='energy'?'Potential energy':'Final potential energy',r.energy.toFixed(5)+' eV'));
  if(!ala15Replay && c.task==='md' && Number.isFinite(r.temperatures?.at(-1)))metrics.push(metric('Final temperature',r.temperatures.at(-1).toFixed(1)+' K'));
  if(['energy','relax'].includes(c.task) && r.forces?.length)metrics.push(metric('Maximum atomic force',Math.max(...r.forces.map(v=>Math.hypot(...v))).toFixed(4)+' eV/Å'));
  if(!ala15Replay && r.times_fs?.length)metrics.push(metric('Simulated time',formatTime(r.recording?.production_duration_fs??(r.times_fs.at(-1)-r.times_fs[0]))));
  if(c.task==='relax' && (typeof r.converged==='boolean'||entry.provenance==='Stopped calculation'))metrics.push(metric('Optimization',r.converged?'Converged':entry.provenance==='Stopped calculation'?'Stopped':'Step limit reached'));
  if(c.task==='trajectory')metrics.push(metric('Saved frames',r.frames.length));
  byId('measurements').replaceChildren(...metrics);
  byId('measurements').hidden=!metrics.length;
  byId('viewport-player').hidden=byId('save-trajectory').hidden=!multi;
  byId('scrub').max=r.frames.length-1;
  byId('animate').disabled=!multi;
  analysis.load(r);
  byId('view-analysis').hidden=byId('trajectory-analysis').hidden;
  changeFrame(initialFrame);
}
function lock(active) {
  working=active;
  document.querySelectorAll('#setup input,#setup button,#setup select,.sample,#runs,#starting-frame,#replay-peptide').forEach(el=>el.disabled=active);
  if(!active){configureMethod();updateAction();}
}
const sessionKey='so3lr-lab-active-job';
function remember(job){
  try{if(job)sessionStorage.setItem(sessionKey,JSON.stringify(job));else sessionStorage.removeItem(sessionKey);}catch{}
}
async function monitor(job) {
  activeJob=job;remember(job);
  const cancel=byId('cancel-job');cancel.hidden=false;cancel.disabled=false;cancel.textContent='Cancel run';
  try{
    let retries=0, lastPreviewStep=-1;
    while(true){
      let state;
      try{state=await request('api/jobs/'+encodeURIComponent(job.id));retries=0;}
      catch(error){
        if(error instanceof TypeError && ++retries<8){
          byId('engine-note').textContent='Reconnecting to local engine…';
          await new Promise(resolve=>setTimeout(resolve,2000));continue;
        }
        if(error instanceof TypeError)throw Error('Connection lost. Reload this tab with the local server running to reconnect.');
        remember(null);throw error;
      }
      if(['done','cancelled','error'].includes(state.status)){
        remember(null);
        if(state.status==='error')throw Error(state.error);
        if(state.status==='cancelled'&&!state.result){flash('Run cancelled.');return;}
        const entry={result:state.result,structure:job.structure,config:job.config,provenance:state.status==='cancelled'?'Stopped calculation':'Local calculation'};
        addResult(entry);
        byId('viewer').scrollIntoView({block:'start',behavior:'smooth'});return;
      }
      const frame=state.preview;
      if(frame && frame.step>lastPreviewStep){
        lastPreviewStep=frame.step;
        const positions=molecule.id==='ala15_folded'?alignedView(frame.positions,molecule.coordinates,molecule.elements).positions:frame.positions;
        scene.positions(positions,null,frame.cell);
        const time=job.config.task==='md'?' · '+formatTime(frame.step*job.config.timestep):'';
        byId('view-kind').textContent='Live '+(job.config.task==='md'?'MD':'optimization')+' · Step '+frame.step+time;
      }
      const p=state.progress;
      byId('engine-note').textContent=state.status==='cancelling'?'Stopping at the next step…':state.status==='queued'?'Waiting for engine':p?'Step '+p.step.toLocaleString()+' / '+p.total.toLocaleString():'Running '+resultModel(job.config);
      await new Promise(resolve=>setTimeout(resolve,state.status==='queued'?700:300));
    }
  }finally{activeJob=null;cancel.hidden=true;
    if(!shown){byId('view-kind').textContent='Structure view';displayStructure(job.structure);}
  }
}
byId('cancel-job').onclick=async()=>{
  if(!activeJob)return;
  const button=byId('cancel-job');button.disabled=true;button.textContent='Stopping…';
  try{await request('api/jobs/'+encodeURIComponent(activeJob.id)+'/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});}
  catch(error){flash(error.message);button.disabled=false;button.textContent='Cancel run';}
};
async function calculate(event) {
  event.preventDefault();if(working||!molecule||!canCalculate(molecule))return;
  const config=settings(), saved=exampleFor(config);
  if(!engine&&!saved){saveScript();return;}
  clearOutput();displayStructure(molecule);lock(true);
  try{
    if(engine){
      if(config.task==='md')config.seed=Math.floor(Math.random()*101);
      byId('action').textContent='Calculating…';
      const {id}=await request('api/calculate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...config,elements:molecule.elements,coordinates:molecule.coordinates,cell:molecule.cell,pbc:molecule.pbc})});
      await monitor({id,config,structure:molecule});
    }else{
      const entry={result:saved.result,structure:molecule,config,provenance:'Recorded example'};
      addResult(entry);
      byId('viewer').scrollIntoView({block:'start',behavior:'smooth'});
    }
  }catch(e){byId('job-error').hidden=false;byId('job-error').textContent=e.message;}
  finally{lock(false);}
}
function loadText(text,title) {
  const item=parseXYZ(text,title);loadMolecule(item);byId('importer').close();flash(item.elements.length+' atoms imported.');
}
async function handleFile(file) {
  if(!file)return;
  try{
    if(file.size>1000000)throw Error('Choose a file smaller than 1 MB.');
    const name=file.name.replace(/\.xyz$/i,'');
    loadText(await file.text(),name||'Imported molecule');
  }catch(e){byId('file-error').textContent=e.message;}
  finally{byId('input-file').value='';}
}
// The bundled ac.xyz stores positions immediately after the element column.
// Reference forces and Hirshfeld ratios in the source belong to the original dataset.
async function sourceCoordinates(entry) {
  if(sourceFrames.has(entry.xyz_file))return sourceFrames.get(entry.xyz_file);
  const response=await fetch(entry.xyz_file);
  if(!response.ok)throw Error('Could not open the source XYZ.');
  const lines=(await response.text()).trimEnd().split(/\r?\n/), frames=[];
  let cursor=0;
  while(cursor<lines.length){
    const count=Number(lines[cursor].trim());
    if(count!==entry.elements.length)throw Error('Unexpected atom count in the source XYZ.');
    cursor+=2;
    const coordinates=entry.elements.map(symbol=>{
      const [element,...fields]=(lines[cursor++]??'').trim().split(/\s+/);
      const point=fields.slice(0,3).map(Number);
      if(element!==symbol || point.length!==3 || !point.every(Number.isFinite))throw Error('Invalid coordinates in the source XYZ.');
      return point;
    });
    frames.push(coordinates);
  }
  if(frames.length!==entry.conformer_count)throw Error('Unexpected geometry count in the source XYZ.');
  sourceFrames.set(entry.xyz_file,frames);return frames;
}
byId('starting-frame').onchange=async event=>{
  const frame=Number(event.target.value),entry=molecule,config=settings();
  lock(true);
  try{
    const frames=await sourceCoordinates(entry);
    loadMolecule({...entry,source_frame:frame,coordinates:frames[frame-1]},config);
  }catch(error){event.target.value=entry.source_frame??1;flash(error.message);}
  finally{lock(false);}
};
byId('setup').onsubmit=calculate;
byId('setup').oninput=event=>{
  if(event.target.id==='precision') {
    setupDevices(hostStatus,byId('device').value);
  }
  if(!molecule)return;
  if(shown){clearOutput();displayStructure(molecule);}
  byId('job-error').hidden=true;configureMethod();updateAction();
};
byId('python').onclick=saveScript;
byId('dipole-toggle').onchange=()=>{if(shown)changeFrame(+byId('scrub').value);};
byId('mode').onchange=()=>scene.style(byId('mode').value);
byId('larger').onclick=()=>scene.zoom(1.18);byId('smaller').onclick=()=>scene.zoom(1/1.18);
byId('reset').onclick=()=>scene.reset();
function expand(open) {
  byId('viewer').classList.toggle('expanded',open);document.body.classList.toggle('focus-mode',open);
  const label=open?'Collapse viewer':'Expand viewer';byId('expand').setAttribute('aria-label',label);byId('expand').title=label;
}
byId('expand').onclick=()=>expand(!byId('viewer').classList.contains('expanded'));
byId('view-analysis').onclick=()=>expand(false);
document.addEventListener('keydown',e=>{if(e.key==='Escape')expand(false);});
byId('bench').onclick=()=>window.scrollTo({top:0,behavior:'smooth'});
byId('runs').onclick=()=>{
  const entries=byId('entries');entries.replaceChildren();
  if(!journal.length)entries.append(node('p','Open an example or run a calculation to start your history.','empty'));
  for(const entry of journal){
    const b=node('button',undefined,'entry');b.append(node('strong',entry.structure.name+(entry.structure.conformer_count>1?' · geometry '+(entry.structure.source_frame??1):'')),node('span',[taskLabel[entry.config.task],resultModel(entry.config),entry.provenance].filter(Boolean).join(' · ')));
    b.onclick=()=>{byId('journal').close();loadMolecule(entry.structure,liveConfig(entry));openResult(entry);};entries.append(b);
  }
  byId('trajectory-import').open=false;
  byId('analyze-file').disabled=!engine;
  byId('trajectory-error').textContent=engine?'':'Start the local SO3LR server to analyze a trajectory.';
  byId('journal').showModal();
};
function importMode(kind) {
  for(const tab of ['xyz','smiles']){
    byId(tab+'-pane').hidden=tab!==kind;
    byId(tab+'-tab').setAttribute('aria-pressed',String(tab===kind));
  }
}
for(const kind of ['xyz','smiles'])byId(kind+'-tab').onclick=()=>{
  smilesRequest?.abort();importMode(kind);
};
byId('importer').addEventListener('close',()=>smilesRequest?.abort());
byId('journal').addEventListener('close',()=>trajectoryRequest?.abort());
byId('trajectory-pane').onsubmit=async event=>{
  event.preventDefault();
  const file=byId('trajectory-file').files[0], button=byId('analyze-file');
  if(!engine||!file||trajectoryRequest)return;
  const controller=new AbortController();trajectoryRequest=controller;
  button.disabled=true;button.textContent='Analyzing frames…';byId('trajectory-error').textContent='';
  try{
    if(file.size>54*1024*1024)throw Error('Choose a trajectory smaller than 54 MiB.');
    const spacing=byId('frame-spacing').valueAsNumber;
    const data=await request('api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},signal:controller.signal,
      body:JSON.stringify({name:file.name.replace(/\.(ext)?xyz$/i,''),xyz:await file.text(),spacing_fs:Number.isFinite(spacing)?spacing:undefined})});
    if(controller.signal.aborted)return;
    const {structure,result}=data, config={task:result.task??'trajectory',model:result.model,charge:structure.charge,temperature:result.md?.temperature,timestep:result.md?.timestep,seed:result.md?.seed};
    loadMolecule(structure);const entry={structure,result,config,provenance:'Imported coordinates'};
    addResult(entry);byId('journal').close();
    byId('trajectory-analysis').scrollIntoView({block:'nearest',behavior:'smooth'});
  }catch(error){if(error.name!=='AbortError')byId('trajectory-error').textContent=error.message;}
  finally{trajectoryRequest=null;button.disabled=false;button.textContent='Analyze trajectory';}
};
byId('import').onclick=()=>{
  byId('file-error').textContent='';byId('coordinates').value='';byId('smiles-error').textContent='';
  importMode('xyz');byId('importer').showModal();byId('xyz-tab').focus();
};
byId('smiles-pane').onsubmit=async event=>{
  event.preventDefault();
  if(!engine||smilesRequest)return;
  const input=byId('smiles-text'),button=byId('build-smiles'),controller=new AbortController();
  smilesRequest=controller;
  const smiles=input.value.trim();byId('smiles-error').textContent='';
  input.disabled=true;button.disabled=true;button.textContent='Building geometry…';
  try {
    const built=await buildFromSmiles(smiles,controller.signal);
    if(!controller.signal.aborted && byId('importer').open && !byId('smiles-pane').hidden) {
      loadMolecule(built);byId('importer').close();flash('Built '+built.elements.length+' atoms · total charge '+built.charge+' e.');
    }
  }catch(error){if(error.name!=='AbortError')byId('smiles-error').textContent=error.message;}
  finally{smilesRequest=null;input.disabled=false;button.disabled=false;button.textContent='Build molecule';}
};
document.querySelectorAll('[data-dismiss]').forEach(btn=>btn.onclick=()=>btn.closest('dialog').close());
byId('input-file').onchange=e=>handleFile(e.target.files[0]);
byId('read-paste').onclick=()=>{try{loadText(byId('coordinates').value,'Imported molecule');}catch(e){byId('file-error').textContent=e.message;}};
const zone=byId('filezone');zone.ondragover=e=>{e.preventDefault();zone.classList.add('dragging');};zone.ondragleave=()=>zone.classList.remove('dragging');zone.ondrop=e=>{e.preventDefault();zone.classList.remove('dragging');handleFile(e.dataTransfer.files[0]);};
byId('scrub').oninput=e=>{haltPlayback();changeFrame(Number(e.target.value));};
byId('animate').onclick=()=>{
  if(playback)return haltPlayback();
  const end=shown.result.frames.length-1;
  if(+byId('scrub').value>=end)changeFrame(0);
  byId('animate').replaceChildren(glyph('pause'));byId('animate').setAttribute('aria-label','Pause trajectory');byId('animate').title='Pause trajectory';
  playback=setInterval(()=>{const n=+byId('scrub').value+1;changeFrame(n);if(n>=end)haltPlayback();},150);
};
function saveTrajectory() {
  if(!shown)return;
  const {result:r,structure:s,config:c}=shown;
  const last=r.frames.length-1, lines=[];
  for(const [index,coordinates] of r.frames.entries()) {
    const forces=index===last?r.forces:null;
    const charges=r.partial_charges?.[index], dipole=r.dipoles_debye?.[index];
    const fields='species:S:1:pos:R:3'+(charges?':partial_charges:R:1':'')+(forces?':forces:R:3':'');
    const cell=r.cells?.[index]??s.cell;
    const meta=['Properties='+fields,'charge='+(c.charge??s.charge??0),'task='+c.task,cell?'pbc="T T T"':'pbc="F F F"'];
    if(cell)meta.push('Lattice="'+cell.flat().join(' ')+'"');
    const optimizer=r.periodic?.optimizer??r.optimizer;
    if(optimizer)meta.push('optimizer='+JSON.stringify(optimizer));
    if(c.model)meta.push('model='+c.model);
    if(r.precision)meta.push('precision='+r.precision);
    if(charges)meta.push('charge_source=so3lr');
    if(dipole)meta.push('dipole_debye="'+dipole.join(' ')+'"');
    if(Number.isFinite(r.energies?.[index]))meta.push('energy='+r.energies[index]);
    if(Number.isFinite(r.kinetic_energies?.[index]))meta.push('kinetic_energy_eV='+r.kinetic_energies[index]);
    if(Number.isFinite(r.temperatures?.[index]))meta.push('temperature_K='+r.temperatures[index]);
    if(Number.isFinite(r.times_fs?.[index]))meta.push('time_fs='+r.times_fs[index]);
    if(s.conformer_count>1)meta.push('source_geometry='+(s.source_frame??1));
    if(r.md?.integrator)meta.push('integrator="'+r.md.integrator+'"');
    if(r.md?.engine)meta.push('md_engine='+r.md.engine);
    if(r.md?.force_backend)meta.push('force_backend="'+r.md.force_backend+'"');
    if(Number.isFinite(r.md?.temperature))meta.push('target_temperature_K='+r.md.temperature);
    if(c.task==='md'){
      meta.push('ensemble='+(r.md?.ensemble??'NVT'));
      if(r.md?.thermostat)meta.push('thermostat="'+r.md.thermostat+'"');
      else if(r.md?.integrator?.includes('Langevin'))meta.push('thermostat=Langevin');
      for(const [name,value] of [['friction_per_fs',r.md?.friction_per_fs],['timestep_fs',c.timestep],['seed',c.seed],
                                 ['temperature_dof',r.md?.temperature_dof],['chain_length',r.md?.chain_length],['lr_cutoff_A',r.md?.lr_cutoff_A]]){
        if(Number.isFinite(value))meta.push(name+'='+value);
      }
    }
    if(c.task!=='energy' && Number.isInteger(r.frame_steps?.[index]))meta.push('step='+r.frame_steps[index]);
    lines.push(String(s.elements.length),meta.join(' '));
    s.elements.forEach((sym,i)=>lines.push([sym,...coordinates[i],...(charges?[charges[i]]:[]),...(forces?forces[i]:[])].join(' ')));
  }
  download(s.id+'-'+c.task+'-trajectory.extxyz',lines.join('\n')+'\n','text/plain');
}
byId('save-trajectory').onclick=saveTrajectory;
function setupDevices(status,requested='auto') {
  const picker=byId('device'), precision=byId('precision').value;
  const devices=(status?.devices??[]).filter(d=>d.precisions.includes(precision));
  const preferred=devices.find(d=>d.id==='gpu')??devices.find(d=>d.id==='cpu');
  for(const row of document.querySelectorAll('.device-row'))row.hidden=!status?.devices?.some(d=>d.id==='gpu');
  const automatic=new Option(preferred?'Automatic · '+preferred.id.toUpperCase():'Automatic','auto');
  const choices=['cpu','gpu'].map(id=>{
    const found=devices.find(device=>device.id===id);
    const opt=new Option(id.toUpperCase()+(engine&&!found?' unavailable':''),id);
    opt.disabled=engine&&!found;
    if(found)opt.title=found.name;
    return opt;
  });
  picker.replaceChildren(automatic,...choices);
  picker.value=(!engine||devices.some(d=>d.id===requested))?requested:'auto';
  picker.title=engine?'Available for '+precision+' on this host: '+devices.map(d=>d.name).join(', '):'Device requested in exported Python jobs';
}
async function initialize(){
  try{
    const local=['/','/index.html'].includes(location.pathname) && ['127.0.0.1','localhost','[::1]'].includes(location.hostname);
    engineCheck=local?request('api/status').catch(()=>null):Promise.resolve(null);
    const [core,samples,paper]=await Promise.all(['molecules.json','examples.json','paper-catalog.json'].map(path=>request(path)));
    const folded=core.find(s=>s.id==='ala15_folded');
    Object.assign(folded,{recording:'ala15-paper.json',recording_label:'Saved 1 ns trajectory',default_task:'md',source_url:'https://arxiv.org/html/2611.06237v1'});
    core.forEach(s=>s.group='Molecular examples');
    const items=[...core,...paper].filter(canCalculate);
    const atomCount=item=>item.atom_count??item.elements.length;
    items.sort((a,b)=>atomCount(a)-atomCount(b));
    exampleRuns.push(...samples);
    library=createLibrary(items,item=>selectExample(item),canvas=>createScene(canvas,{interactive:false}));
    let job;try{job=JSON.parse(sessionStorage.getItem(sessionKey));}catch{}
    const query=new URLSearchParams(location.search),example=query.get('example');
    const preset=items.find(item=>item.id===(example==='ala15'?'ala15_folded':example))??folded;
    const startup=job?.id?Promise.resolve():selectExample(preset,{replay:query.get('view')!=='structure' && !!preset.recording,scroll:false,play:!example});
    const online=await engineCheck;engine=online?.engine==='so3lr';hostStatus=online;
    byId('local-guide').hidden=engine;
    if(!engine){byId('smiles-help').textContent='Start the local SO3LR server to generate 3D coordinates from SMILES.';byId('build-smiles').disabled=true;}
    await startup;
    setupDevices(online,byId('device').value);updateAction();
    if(engine && job?.id && job.config && job.structure){
      loadMolecule(job.structure,job.config);lock(true);byId('action').textContent='Calculating…';
      try{await monitor(job);}catch(e){byId('job-error').hidden=false;byId('job-error').textContent=e.message;}
      finally{lock(false);}
    }else if(!molecule){loadMolecule(folded);await selectExample(preset,{scroll:false});}

  }catch(err){byId('engine-note').textContent='Could not load workspace';flash(err.message);}
}
initialize();
