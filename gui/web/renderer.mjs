import {cellGeometry,lattice} from './cell.mjs';

// A browser-local molecular viewport. Coordinates and display connectivity use Å.
const palette = {
  H: '#f1f1ee', C: '#a3aab5', N: '#718deb', O: '#ed726b', F: '#7fd7b3',
  S: '#e5be52', P: '#eb9859', Cl: '#87bd74', Br: '#b47d64', I: '#a282ca',
  Si: '#d3b496', B: '#ddacaa', Fe: '#c08156', Cu: '#b88966', Zn: '#a7bfc6',
  Na: '#af9acb', Mg: '#a5c57e', K: '#a999c7', Ca: '#afd789'
};
export function tint(symbol) {
  return palette[symbol] ?? '#9db6c2';
}

const symbols = 'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At'.split(' ');
const radii = [0.31,0.28,1.28,0.96,0.84,0.76,0.71,0.66,0.57,0.58,1.66,1.41,1.21,1.11,1.07,1.05,1.02,1.06,2.03,1.76,1.70,1.60,1.53,1.39,1.39,1.32,1.26,1.24,1.32,1.22,1.22,1.20,1.19,1.20,1.20,1.16,2.20,1.95,1.90,1.75,1.64,1.54,1.47,1.46,1.42,1.39,1.45,1.44,1.42,1.39,1.39,1.38,1.39,1.40,2.44,2.15,2.07,2.04,2.03,2.01,1.99,1.98,1.98,1.96,1.94,1.92,1.92,1.89,1.90,1.87,1.87,1.75,1.70,1.62,1.51,1.44,1.41,1.36,1.36,1.32,1.45,1.46,1.48,1.40,1.50];
const covalent = new Map(symbols.map((s,i) => [s,radii[i]]));
const vdw = {H:1.20,C:1.70,N:1.55,O:1.52,F:1.47,P:1.80,S:1.80,Cl:1.75,Br:1.85,I:1.98};
const bondRadius = s => covalent.get(s) ?? 0.88;
const radius = (s, mode) => mode === 'space' ? (vdw[s] ?? 1 + bondRadius(s)*0.8)
  : mode === 'wire' ? 0.17 : s === 'H' ? 0.26 : 0.35 + 0.11*bondRadius(s);

const scale = (v, a) => v.map(n => n*a);
const minus = (a,b) => a.map((v,i) => v-b[i]);
const inner = (a,b) => a.reduce((sum,n,i) => sum+n*b[i], 0);
const unit = v => scale(v, 1/(Math.hypot(...v) || 1));
const cross = ([a,b,c],[x,y,z]) => [b*z-c*y,c*x-a*z,a*y-b*x];
const identity = [1,0,0,0];

function quaternion(axis, radians) {
  return [Math.cos(radians/2), ...scale(unit(axis), Math.sin(radians/2))];
}
function compose([w,x,y,z],[s,i,j,k]) {
  return unit([w*s-x*i-y*j-z*k, w*i+x*s+y*k-z*j,
    w*j-x*k+y*s+z*i, w*k+x*j-y*i+z*s]);
}
function turn([w,x,y,z], [a,b,c]) {
  const u=2*(y*c-z*b), v=2*(z*a-x*c), t=2*(x*b-y*a);
  return [a+w*u+y*t-z*v,b+w*v+z*u-x*t,c+w*t+x*v-y*u];
}

// Use the heavy-atom span and the widest perpendicular dimension to place
// long molecules diagonally across the viewport, keeping ring faces visible.
function molecularFrame(elements, coords) {
  let anchors = coords.filter((_, i) => elements[i] !== 'H');
  if (anchors.length < 2) anchors = coords;
  if (anchors.length < 2) return [[1,0,0],[0,1,0],[0,0,1]];
  const farthest = from => anchors.reduce((best, p) =>
    inner(minus(p,from),minus(p,from)) > inner(minus(best,from),minus(best,from)) ? p : best);
  const first = farthest(anchors[0]);
  const horizontal = unit(minus(farthest(first),first));
  let vertical = [0,0,0], width = 0;
  for (const atom of anchors) {
    const v = minus(atom,first);
    const side = minus(v, scale(horizontal,inner(v,horizontal)));
    const d = inner(side,side);
    if (d > width) { vertical = side; width = d; }
  }
  if (width < 1e-12) vertical = cross(Math.abs(horizontal[1]) < 0.9 ? [0,1,0] : [0,0,1], horizontal);
  vertical = unit(vertical);
  return [horizontal,vertical,cross(horizontal,vertical)];
}

function neighbors(elements, coords) {
  const links = [];
  // The studio's import cap keeps the direct distance search small.
  for (let i=0; i<coords.length; i++) {
    for (let j=0; j<i; j++) {
      const cutoff = 1.19*(bondRadius(elements[i])+bondRadius(elements[j]));
      const v = minus(coords[i],coords[j]);
      const d2 = inner(v,v);
      if (d2 > 0.25*0.25 && d2 <= cutoff*cutoff) links.push([i,j]);
    }
  }
  return links;
}

const diffuse = [-0.46,-0.61,0.645];
const softbox = unit([diffuse[0],diffuse[1],diffuse[2]+1]);
const sphereCache = new Map();
function rgb(hex) {
  return [1,3,5].map(i => parseInt(hex.slice(i,i+2),16));
}

// Cache a softly lit, antialiased sphere for each material. Painting sprites
// keeps rotation smooth and retains a fixed light direction as the atoms turn.
function sphere(owner, key) {
  if (sphereCache.has(key)) return sphereCache.get(key);
  const size=160, half=size/2-1;
  const tile = owner.createElement('canvas');
  tile.width=tile.height=size;
  const ctx=tile.getContext('2d'), pixels=ctx.createImageData(size,size);
  const base=rgb(key);
  for (let y=0; y<size; y++) for (let x=0; x<size; x++) {
    const nx=(x+0.5-size/2)/half, ny=(y+0.5-size/2)/half;
    const d=nx*nx+ny*ny;
    if (d > 1.0125) continue;
    const nz=Math.sqrt(Math.max(0,1-d));
    const keyLight=Math.max(0,nx*diffuse[0]+ny*diffuse[1]+nz*diffuse[2]);
    const sky=Math.max(0,0.65*nx-0.15*ny+0.74*nz);
    const spec=Math.pow(Math.max(0,nx*softbox[0]+ny*softbox[1]+nz*softbox[2]),20);
    const rim=Math.max(0,-0.64*nx-0.62*ny-0.45*nz)*(1-nz);
    const light=0.34+0.58*keyLight+0.11*sky;
    const k=4*(y*size+x);
    for (let c=0;c<3;c++) pixels.data[k+c]=Math.min(255,base[c]*light+46*spec+21*rim+c*1.7*(1-keyLight));
    pixels.data[k+3]=Math.round(255*Math.max(0,Math.min(1,(1-Math.sqrt(d))*half+0.5)));
  }
  ctx.putImageData(pixels,0,0);
  sphereCache.set(key,tile);
  return tile;
}

function rod(ctx, a, b, width, color) {
  const dx=b[0]-a[0], dy=b[1]-a[1], distance=Math.hypot(dx,dy);
  if (distance < 0.1) return;
  let nx=-dy/distance*width, ny=dx/distance*width;
  if (0.6*nx+ny<0) {nx=-nx;ny=-ny;}
  const grad=ctx.createLinearGradient(a[0]-nx,a[1]-ny,a[0]+nx,a[1]+ny);
  const base=rgb(color);
  for (const [stop,lit,gloss] of [[0,.51,0],[.2,.84,6],[.36,1.01,12],[.58,.89,2],[.83,.65,0],[1,.47,0]]) {
    grad.addColorStop(stop,'rgb('+base.map(v=>Math.min(255,Math.round(v*lit+gloss))).join(',')+')');
  }
  ctx.fillStyle=grad;
  ctx.beginPath();
  ctx.moveTo(a[0]-nx,a[1]-ny); ctx.lineTo(b[0]-nx,b[1]-ny);
  ctx.lineTo(b[0]+nx,b[1]+ny); ctx.lineTo(a[0]+nx,a[1]+ny);
  ctx.closePath();ctx.fill();
}

export function createScene(canvas, {onSelect,interactive=true}={}) {
  const ctx=canvas.getContext('2d', {alpha:true});
  let elements=[], sketch=[], reference=[], links=[], basis=[], origin=[0,0,0];
  let picked=null, hover=null, mode='ball', factor=1, disposed=false, dots=[];
  const initialCamera=compose(quaternion([0,0,1],0.25),compose(quaternion([0,1,0],0.3),quaternion([1,0,0],-0.9)));
  let camera=identity, dipole=null, cell=null, corners=[], outside=[];
  const colorAt=i=>tint(elements[i]);
  const convert = p => basis.map(axis => inner(minus(p,origin),axis));
  function current(coords) {
    if(cell){
      const view=cellGeometry(cell,coords,elements,bondRadius,elements.length<=2500);
      sketch=view.points.map(convert); corners=view.corners.map(convert);links=view.links;
      outside=view.outside.map(([a,b,i])=>[convert(a),convert(b),i]);
    }else{
      // A gas-phase trajectory can translate as a whole. Center the current frame
      // for display while preserving its Cartesian coordinates for analysis/export.
      origin=[0,1,2].map(i=>coords.reduce((sum,p)=>sum+p[i],0)/coords.length);
      sketch=coords.map(convert); corners=[];outside=[];
      links=elements.length<=2500?neighbors(elements,sketch):[];
    }
  }
  function draw() {
    if (disposed || !ctx) return;
    const w=canvas.clientWidth, h=canvas.clientHeight;
    if (!w || !h) return;
    const ratio=Math.min(3,window.devicePixelRatio || 1);
    if (canvas.width!==Math.round(w*ratio)) canvas.width=Math.round(w*ratio);
    if (canvas.height!==Math.round(h*ratio)) canvas.height=Math.round(h*ratio);
    ctx.setTransform(ratio,0,0,ratio,0,0);
    ctx.clearRect(0,0,w,h);
    dots=[];
    if (!sketch.length) return;
    const bounds=cell?corners:reference;
    const extent=bounds.reduce((n,p,i) => Math.max(n,Math.hypot(...p)+(cell?0.5:radius(elements[i],mode))),1);
    const focal=Math.max(6,extent*5.5);
    let left=Infinity,right=-Infinity,bottom=Infinity,top=-Infinity;
    bounds.forEach((p,i) => {
      const [x,y,z]=turn(initialCamera,p), r=cell?0.5:radius(elements[i],mode), f=focal/(focal-z);
      left=Math.min(left,(x-r)*f); right=Math.max(right,(x+r)*f);
      bottom=Math.min(bottom,(y-r)*f); top=Math.max(top,(y+r)*f);
    });
    const size=factor*Math.min(w * (interactive ? 0.82 : 0.9) / (right-left), h * (interactive ? 0.88 : 0.84) / (top-bottom));
    const ox=w/2-(left+right)*size/2, oy=h/2+(bottom+top)*size/2;
    dots=sketch.map((p,i) => {
      const [x,y,z]=turn(camera,p), f=size*focal/Math.max(focal*0.18,focal-z), r=radius(elements[i],mode);
      return {x:ox+x*f,y:oy-y*f,z,r:r*f,f,i,depth:z+0.4*r};
    });
    const project=p=>{
      const [x,y,z]=turn(camera,p), f=size*focal/Math.max(focal*.18,focal-z);
      return [ox+x*f,oy-y*f,z,f];
    };
    if(cell){
      const vertices=corners.map(project);
      ctx.save();ctx.lineWidth=1;ctx.strokeStyle=interactive?'#8fa1b1':'#7a899a';ctx.globalAlpha=.78;
      ctx.beginPath();
      for(let i=0;i<8;i++)for(const bit of [1,2,4]){
        if(i&bit)continue;
        ctx.moveTo(vertices[i][0],vertices[i][1]);ctx.lineTo(vertices[i|bit][0],vertices[i|bit][1]);
      }
      ctx.stroke();ctx.restore();
    }
    if(mode==='points' || elements.length>2500){
      const ordered=dots.slice().sort((a,b)=>a.z-b.z);
      for(const d of ordered){
        ctx.fillStyle=colorAt(d.i);
        const r=Math.max(.7,d.r);
        ctx.fillRect(d.x-r,d.y-r,2*r,2*r);
      }
      if(picked!=null){
        ctx.strokeStyle='#f3d084';ctx.beginPath();ctx.arc(dots[picked].x,dots[picked].y,5,0,2*Math.PI);ctx.stroke();
      }
    }else{
      const items=dots.slice();
      if (mode!=='space') for (const [i,j] of links) {
        const a=dots[i], b=dots[j];
        const mx=(a.x+b.x)/2, my=(a.y+b.y)/2, mz=(a.z+b.z)/2;
        for (const end of [a,b]) {
          const dx=mx-end.x,dy=my-end.y,d=Math.hypot(dx,dy);
          if (d<0.1) continue;
          const cut=Math.min(.92,end.r*.70/d);
          items.push({a:[end.x+dx*cut,end.y+dy*cut],b:[mx+.25*dx/d,my+.25*dy/d],
            width:end.f * (mode === 'wire' ? 0.112 : 0.096),color:colorAt(end.i),depth:end.z+(mz-end.z)*(.5+.5*cut)});
        }
      }
      if(mode!=='space')for(const [p,q,i] of outside){
        const a=project(p),b=project(q);
        items.push({a,b,width:a[3]*.096,color:colorAt(i),depth:(a[2]+b[2])/2});
      }
      items.sort((a,b)=>a.depth-b.depth);
      for (const item of items) {
        if (item.i===undefined) {rod(ctx,item.a,item.b,item.width,item.color);continue;}
        const {x,y,r,i}=item;
        if (i===picked || i===hover) {
          ctx.beginPath();ctx.arc(x,y,r+3,0,2*Math.PI);
          ctx.strokeStyle=i===picked?'#f3d084':'rgba(236,238,244,.78)';ctx.lineWidth=i===picked?2:1;ctx.stroke();
        }
        const sprite=sphere(canvas.ownerDocument,colorAt(i));
        if (interactive && mode !== 'wire') {
          ctx.shadowColor='rgba(3,5,9,.24)';
          ctx.shadowBlur=r*0.14; ctx.shadowOffsetX=r*0.05; ctx.shadowOffsetY=r*0.10;
        }
        ctx.drawImage(sprite,x-r,y-r,r*2,r*2);
        ctx.shadowColor='transparent';
        ctx.shadowBlur=ctx.shadowOffsetX=ctx.shadowOffsetY=0;
      }
    }
    if (dipole && Math.hypot(...dipole)>1e-6) {
      const direction=turn(camera,basis.map(axis=>inner(unit(dipole),axis)));
      const center=sketch.reduce((sum,p)=>sum.map((v,j)=>v+p[j]/sketch.length),[0,0,0]);
      const [cx,cy,cz]=turn(camera,center), projection=size*focal/Math.max(.18*focal,focal-cz);
      const x=ox+cx*projection, y=oy-cy*projection, len=Math.min(w,h)*.23;
      const dx=direction[0]*len, dy=-direction[1]*len, a=Math.atan2(dy,dx);
      ctx.save();ctx.strokeStyle=ctx.fillStyle='#ffdb85';ctx.shadowColor='#27221c';ctx.shadowBlur=2;ctx.lineWidth=3;
      ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+dx,y+dy);ctx.stroke();
      if(Math.hypot(dx,dy)>5){
        ctx.beginPath();ctx.moveTo(x+dx,y+dy);
        for(const side of [-.5,.5])ctx.lineTo(x+dx-10*Math.cos(a+side),y+dy-10*Math.sin(a+side));
        ctx.closePath();ctx.fill();
      } else {ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.stroke();}
      ctx.font='bold 15px system-ui';ctx.fillText('μ',x+dx+7,y+dy-5);ctx.restore();
    }
  }

  function load(nextElements, coords, nextCell=null) {
    if (nextElements.length!==coords.length) throw new Error('Expected one position per atom.');
    elements=Array.from(nextElements);
    cell=nextCell;
    basis=cell || elements.length>2500?[[1,0,0],[0,1,0],[0,0,1]]:molecularFrame(elements,coords);
    origin=[0,1,2].map(i=>coords.reduce((s,p)=>s+p[i],0)/(coords.length || 1));
    if(cell)origin=lattice(cell).cart([.5,.5,.5]);
    current(coords);
    reference=sketch;
    picked=hover=null;dipole=null;
    reset();
  }
  function positions(coords, vector=null, nextCell=cell) {
    cell=nextCell;dipole=vector;
    if(cell)origin=lattice(cell).cart([.5,.5,.5]);
    if (coords.length!==elements.length) throw new Error('The atom count changed between frames.');
    current(coords);
    draw();
  }
  function style(name) {
    if (!['ball','space','wire','points'].includes(name)) return;
    mode=name;draw();
  }
  function zoom(amount) {
    if (!(amount>0) || !Number.isFinite(amount)) return;
    factor=Math.max(.22,Math.min(6,factor*amount)); draw();
  }
  function reset() {
    camera=initialCamera; factor=1; draw();
  }
  function at(x,y) {
    let index=null, nearest=-Infinity;
    for(const p of dots){
      const inside=p.r*p.r-(x-p.x)**2-(y-p.y)**2;
      if (inside<0) continue;
      const front=p.z+Math.sqrt(inside)/p.f;
      if (front>nearest) {nearest=front;index=p.i;}
    }
    return index;
  }
  const orbit=(dx,dy) => {
    const angle=Math.hypot(dx,dy);
    if (!angle) return;
    camera=compose(quaternion([dy,dx,0],angle),camera);
    draw();
  };
  const events=new AbortController();
  if (interactive) {
    const active=new Map();let dragging=false, start;
    canvas.style.touchAction='none';
    canvas.style.cursor='grab';
    if (!canvas.hasAttribute('tabindex')) canvas.tabIndex=0;
    if (!canvas.hasAttribute('aria-label')) canvas.setAttribute('aria-label','Molecule. Arrow keys rotate, plus and minus zoom, Home resets.');
    const on=(type,fn,extra={}) => canvas.addEventListener(type,fn,{signal:events.signal,...extra});
    const point=e=>{const r=canvas.getBoundingClientRect();return [e.clientX-r.left,e.clientY-r.top];};
    const spread=()=>{const [a,b]=active.values();return Math.hypot(a[0]-b[0],a[1]-b[1]);};
    on('pointerdown',e=>{
      if (e.button!==0) return;
      canvas.focus({preventScroll:true}); canvas.setPointerCapture(e.pointerId);
      start=point(e); active.set(e.pointerId,start); dragging=active.size>1;
      hover=null;canvas.style.cursor='grabbing';draw();
    });
    on('pointermove',e=>{
      const p=point(e), prev=active.get(e.pointerId);
      if (!prev) {
        if (active.size || e.pointerType==='touch') return;
        const hit=at(...p);
        if (hover!==hit) {hover=hit;canvas.style.cursor=hover===null?'grab':'pointer';draw();}
        return;
      }
      const old=active.size===2?spread():0;
      active.set(e.pointerId,p);
      if (active.size===2 && old>1) { zoom(spread()/old);return; }
      if (active.size!==1) return;
      if (Math.hypot(p[0]-start[0],p[1]-start[1])>3) dragging=true;
      if (dragging) orbit((p[0]-prev[0])*.009,(p[1]-prev[1])*.009);
    });
    const finish=e=>{
      if (!active.has(e.pointerId)) return;
      active.delete(e.pointerId);
      if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
      if (e.type==='pointerup' && !dragging) {
        picked=at(...point(e));onSelect?.(picked);draw();
      }
      if (active.size) start=active.values().next().value;
      canvas.style.cursor=active.size?'grabbing':'grab';
    };
    on('pointerup',finish);on('pointercancel',finish);on('lostpointercapture',finish);
    on('pointerleave',()=>{if(hover!==null){hover=null;draw();}});
    on('wheel',e=>{
      e.preventDefault();
      const pixels=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?canvas.clientHeight:1);
      zoom(Math.exp(-Math.max(-200,Math.min(200,pixels))*.002));
    },{passive:false});
    on('keydown',e=>{
      const rotations={ArrowLeft:[-.12,0],ArrowRight:[.12,0],ArrowUp:[0,-.12],ArrowDown:[0,.12]};
      if (rotations[e.key]) orbit(...rotations[e.key]);
      else if (e.key==='+' || e.key==='=') zoom(1.18);
      else if (e.key==='-') zoom(1/1.18);
      else if ((e.key==='[' || e.key===']') && elements.length) {
        const direction=e.key===']'?1:-1;
        picked=picked===null?(direction===1?0:elements.length-1):(picked+direction+elements.length)%elements.length;
        onSelect?.(picked);draw();
      } else if (e.key==='Home') reset();
      else return;
      e.preventDefault();
    });
  }
  const resize=new ResizeObserver(draw);
  resize.observe(canvas);
  function dispose() {disposed=true;resize.disconnect();events.abort();dots=[];}
  return {load,positions,style,zoom,reset,dispose};
}
