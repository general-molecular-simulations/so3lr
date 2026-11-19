// Eight-state DSSP, as returned by the analysis service. C is the original blank code.
const classes = {
  H: {name:'α-helix', color:'#c95715'}, G: {name:'3₁₀-helix', color:'#2879b3'},
  T: {name:'Turn', color:'#308451'}, C: {name:'Coil', color:'#686b70'},
  S: {name:'Bend', color:'#a77a18'}, I: {name:'π-helix', color:'#8e5ab0'},
  E: {name:'β-strand', color:'#25868c'}, B: {name:'β-bridge', color:'#ac5878'},
};
const svgNS = 'http://www.w3.org/2000/svg';
function element(tag,cls,text) {
  const el=document.createElement(tag);
  if(cls)el.className=cls;
  if(text!==undefined)el.textContent=text;
  return el;
}
function vector(tag,attributes,text) {
  const el=document.createElementNS(svgNS,tag);
  for(const [key,value] of Object.entries(attributes))el.setAttribute(key,value);
  if(text!==undefined)el.textContent=text;
  return el;
}
function numeric(value,digits=3) {return Number(value.toFixed(digits)).toString();}
export function formatTime(fs) {
  return fs>=1e6?(fs/1e6).toLocaleString(undefined,{maximumFractionDigits:4})+' ns':fs>=1000?(fs/1000).toLocaleString(undefined,{maximumFractionDigits:4})+' ps':fs.toLocaleString(undefined,{maximumFractionDigits:3})+' fs';
}
function timeAxis(result,n) {
  const valid=series=>series?.length===n && series.every((v,i)=>Number.isFinite(v)&&(!i||v>series[i-1]));
  if(valid(result.times_fs)){
    const span=result.times_fs.at(-1)-result.times_fs[0];
    const scale=span>=1e6?1e6:span>=1000?1000:1;
    return {x:result.times_fs.map(v=>v/scale),label:'Time',unit:scale===1e6?'ns':scale===1000?'ps':'fs'};
  }
  if(valid(result.frame_steps))return {x:result.frame_steps,label:'Step',unit:''};
  return {x:Array.from({length:n},(_,i)=>i+1),label:'Frame',unit:''};
}
function cellVolumes(result){
  return result.volumes_A3??result.analysis?.cell_volumes??result.cells?.map(([[a,b,c],[d,e,f],[g,h,i]])=>
    Math.abs(a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)));
}

function closest(xs,target) {
  let lo=0,hi=xs.length-1;
  while(lo<hi){const mid=(lo+hi)>>>1;if(xs[mid]<target)lo=mid+1;else hi=mid;}
  return lo>0&&target-xs[lo-1]<xs[lo]-target?lo-1:lo;
}

function chart(host, {axis,yRange,yTicks,series,name,onFrame,reference,step=false,small=false}) {
  const xs=axis.x,n=xs.length,lo=xs[0],hi=xs.at(-1);
  const svg=vector('svg',{class:'pa-chart',role:'slider',tabindex:0,'aria-label':name+' · choose a trajectory frame',
    'aria-valuemin':1,'aria-valuemax':n,'aria-keyshortcuts':'ArrowLeft ArrowRight Home End'});
  host.append(svg);
  let width=0,index=0,marker,points=[],start=39,end=0,drag=null;
  const height=small?154:190,top=12,bottom=height-38;
  const tickLabel=tick=>Number(tick.toPrecision(7)).toString();
  start=Math.max(31,Math.max(...yTicks.map(t=>tickLabel(t).length))*7.3+11);
  const px=value=>start+(value-lo)/(hi-lo||1)*(end-start);
  const py=value=>bottom-(value-yRange[0])/(yRange[1]-yRange[0])*(bottom-top);

  function move(i) {
    index=i;
    if(!marker)return;
    const x=px(xs[i]);marker.setAttribute('x1',x);marker.setAttribute('x2',x);
    for(const [point,line] of points){
      const y=line.values[i];
      point.setAttribute('visibility',Number.isFinite(y)?'visible':'hidden');
      if(Number.isFinite(y)){point.setAttribute('cx',x);point.setAttribute('cy',py(y));}
    }
    svg.setAttribute('aria-valuenow',i+1);
    svg.setAttribute('aria-valuetext',axis.label+' '+numeric(xs[i])+' '+axis.unit);
  }
  function render() {
    width=Math.max(150,Math.floor(host.clientWidth));end=width-12;
    svg.setAttribute('viewBox','0 0 '+width+' '+height);
    svg.replaceChildren();points=[];
    for(const tick of yTicks) {
      const y=py(tick);
      svg.append(vector('line',{x1:start,y1:y,x2:end,y2:y,class:'pa-grid'}),
        vector('text',{x:start-6,y:y+4,'text-anchor':'end'},tickLabel(tick)));
    }
    if(Number.isFinite(reference)){
      const guide=vector('line',{x1:start,y1:py(reference),x2:end,y2:py(reference),class:'pa-reference'});
      guide.append(vector('title',{},'Target '+numeric(reference)+' K'));svg.append(guide);
    }
    let ticks=width<330?[lo,(lo+hi)/2,hi]:[lo,lo+(hi-lo)*0.25,(lo+hi)/2,lo+(hi-lo)*0.75,hi];
    if(!axis.unit)ticks=[...new Set(ticks.map(Math.round))];
    for(const v of ticks){
      const pos=px(v),align=v===lo?'start':v===hi?'end':'middle';
      svg.append(vector('text',{x:pos,y:bottom+18,'text-anchor':align},numeric(v)));
    }
    svg.append(vector('text',{x:(start+end)/2,y:height-2,'text-anchor':'middle',class:'pa-axis-title'},
      axis.label+(axis.unit?' ('+axis.unit+')':'')));
    for(const line of series.filter(s=>s.visible!==false)){
      let path='',prev=null;
      for(let i=0;i<n;i++){
        const y=line.values[i];
        if(!Number.isFinite(y)){prev=null;continue;}
        const x=numeric(px(xs[i]),2),v=numeric(py(y),2);
        if(prev===null)path+='M'+x+','+v;
        else path+=step?'H'+x+'V'+v:'L'+x+','+v;
        prev=y;
      }
      svg.append(vector('path',{d:path,stroke:line.color,fill:'none','stroke-width':step?2.2:1.8,
        'stroke-linejoin':'round','vector-effect':'non-scaling-stroke'}));
    }
    marker=vector('line',{y1:top,y2:bottom,class:'pa-marker'});svg.append(marker);
    for(const line of series.filter(s=>s.visible!==false)){
      const point=vector('circle',{r:3.2,fill:line.color,stroke:'#fff','stroke-width':1});
      points.push([point,line]);svg.append(point);
    }
    move(index);
  }
  function seek(event) {
    const bounds=svg.getBoundingClientRect(),x=(event.clientX-bounds.left)/bounds.width*width;
    const target=lo+Math.max(0,Math.min(1,(x-start)/(end-start)))*(hi-lo);
    onFrame(closest(xs,target));
  }
  svg.onpointerdown=e=>{if(e.button!==0)return;drag=e.pointerId;svg.setPointerCapture(e.pointerId);seek(e);};
  svg.onpointermove=e=>{if(e.pointerId===drag)seek(e);};
  svg.onpointercancel=svg.onlostpointercapture=svg.onpointerup=()=>{drag=null;};
  svg.onkeydown=e=>{
    let next;
    if(e.key==='ArrowLeft'||e.key==='ArrowDown')next=Math.max(0,index-1);
    else if(e.key==='ArrowRight'||e.key==='ArrowUp')next=Math.min(n-1,index+1);
    else if(e.key==='Home')next=0;
    else if(e.key==='End')next=n-1;
    else return;
    e.preventDefault();onFrame(next);
  };
  const resize=new ResizeObserver(()=>{if(Math.floor(host.clientWidth)!==width)render();});
  resize.observe(host);render();
  return {move,render,dispose:()=>resize.disconnect()};
}

export function createAnalysis(host,{onFrame}={}) {
  host.hidden=true;
  let graphs=[],data,peptide,axis,n;
  let readouts=[];

  function clear() {
    graphs.forEach(graph=>graph.dispose());
    graphs=[];readouts=[];data=null;peptide=null;
    host.replaceChildren();host.hidden=true;
  }
  function go(i) {
    if(!data || !Number.isInteger(i) || i<0 || i>=n)return;
    frame(i);onFrame?.(i);
  }
  function frame(i) {
    if(!data || !Number.isInteger(i)||i<0||i>=n)return;
    for(const {output,values,unit,digits} of readouts)output.textContent=numeric(values[i],digits)+' '+unit;
    for(const graph of graphs)graph.move(i);
  }
  function populations(result) {
    const section=element('div','pa-group pa-population');
    const head=element('div','pa-heading');
    head.append(element('h3','',n>1?'Secondary structure over time':'Secondary structure'));
    section.append(head);host.append(section);
    if(n>1){
      const span=result.recording?.production_duration_fs??(result.times_fs?.at(-1)-result.times_fs?.[0]);
      const period=Number.isFinite(span) && span>0?formatTime(span):'the trajectory';
      section.append(element('p','pa-sub','Percentages averaged across '+period));
    }
    const fractionSeries=[];
    let populationGraph;
    const legend=element('div','pa-legend');section.append(legend);
    for(const [code,type] of Object.entries(classes)){
      const values=peptide.fractions[code];
      if(!['H','G','T','C'].includes(code)&&!values.some(v=>v>0))continue;
      const series={values,color:type.color,visible:true};fractionSeries.push(series);
      const item=element(n>1?'button':'span','pa-legend-item');
      item.style.setProperty('--ss-color',type.color);
      const mean=values.reduce((sum,value)=>sum+value,0)/values.length;
      const percent=mean*100;
      const label=percent>0 && percent<0.05?'<0.1%':numeric(percent,1)+'%';
      const value=element('span','pa-value',label);
      item.append(element('i','pa-swatch'),document.createTextNode(type.name),value);legend.append(item);
      if(n>1){
        item.type='button';item.setAttribute('aria-pressed','true');item.title='Toggle '+type.name;
        item.onclick=()=>{series.visible=!series.visible;item.setAttribute('aria-pressed',String(series.visible));populationGraph.render();};
      }
    }
    if(n>1){
      section.append(element('div','pa-axis-caption','Fraction of residues'));
      const surface=element('div','pa-surface');section.append(surface);
      populationGraph=chart(surface,{axis,yRange:[0,1],yTicks:[0,.25,.5,.75,1],series:fractionSeries,
        name:'Secondary structure population',onFrame:go,step:true});
      graphs.push(populationGraph);
      if(fractionSeries.every(s=>s.values.every(value=>value===s.values[0]))){
        section.append(element('p','pa-note','The populations stay constant across all '+n+' saved frames.'));
      }
    }
  }
  function scalarGraph(grid,{name,values,unit,color,caption,digits=3,zero=false,reference}) {
    if(n<2 || values?.length!==n || !values.every(Number.isFinite))return;
    const group=element('section','pa-group'),head=element('div','pa-heading');
    const output=element('output','pa-current');
    readouts.push({output,values,unit,digits});
    head.append(element('h3','',name),output);
    group.append(head,element('p','pa-sub',caption));
    const surface=element('div','pa-surface pa-scalar');group.append(surface);grid.append(group);
    const min=Math.min(...values,reference??Infinity),max=Math.max(...values,reference??-Infinity);
    const span=Math.max(max-(zero?0:min),Math.abs(max)*1e-6,unit==='K'?1:.001);
    // Round bounds to readable values without losing small energy changes around a large offset.
    const power=10**Math.floor(Math.log10(span/3));
    const increment=([1,2,2.5,5,10].find(m=>m*power>=span/3)??10)*power;
    const low=zero?0:Math.floor(min/increment)*increment;
    const high=Math.max(low+increment,Math.ceil(max/increment)*increment);
    const ticks=Array.from({length:Math.round((high-low)/increment)+1},(_,i)=>low+i*increment);
    graphs.push(chart(surface,{axis,yRange:[low,high],yTicks:ticks,series:[{values,color}],
      name:name+' ('+unit+')',onFrame:go,reference,small:true}));
  }
  function load(result) {
    clear();data=result?.analysis??{};n=result?.frames?.length||0;
    if(!n){data=null;return;}
    const candidate=data.peptide;
    peptide=(candidate?.residue_count??candidate?.residues?.length)===15?candidate:null;
    const plots=[
      {name:'Heavy-atom RMSD',values:!peptide && !result.cells && data.heavy_atom_count?data.rmsd:null,unit:'Å',color:'#516da1',
       caption:'Å · Aligned to frame 1 · '+data.heavy_atom_count+' heavy atoms',zero:true},
      {name:'Unit cell volume',values:cellVolumes(result),unit:'Å³',color:'#32769a',caption:'Å³ · Volume of the periodic cell'},
      {name:'Potential energy',values:result.energies,unit:'eV',color:'#795d98',caption:'eV · Potential energy of the structure'},
      {name:'Kinetic energy',values:result.kinetic_energies,unit:'eV',color:classes.H.color,caption:'eV · Sum over all atoms'},
      {name:'Temperature',values:result.temperatures,unit:'K',color:classes.G.color,digits:1,reference:Number.isFinite(result.md?.temperature)?result.md.temperature:undefined,
       caption:'K · Instantaneous kinetic temperature'+(Number.isFinite(result.md?.temperature)?' · Target '+result.md.temperature+' K':'')},
    ];
    if(!peptide && (n<2 || !plots.some(p=>p.values?.length===n && p.values.every(Number.isFinite)))){data=null;return;}
    axis=timeAxis(result,n);host.hidden=false;
    if(peptide)populations(result);
    if(n>1){
      const grid=element('div','pa-plots');host.append(grid);
      for(const plot of plots)scalarGraph(grid,plot);
    }
    frame(n-1);
  }
  return {load,frame,clear};
}
