// ASE convention: each row is a lattice vector, in ångströms.
const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];

export function lattice(cell) {
  const [a,b,c]=cell, dual=[cross(b,c),cross(c,a),cross(a,b)];
  const volume=dot(a,dual[0]);
  if(!Number.isFinite(volume)||Math.abs(volume)<1e-8)throw Error('The unit cell is singular.');
  const reciprocal=dual.map(v=>v.map(x=>x/volume));
  const cart=f=>[0,1,2].map(i=>f[0]*a[i]+f[1]*b[i]+f[2]*c[i]);
  const fractions=p=>reciprocal.map(v=>dot(p,v));
  const corners=Array.from({length:8},(_,i)=>cart([i&1,(i>>1)&1,(i>>2)&1]));
  return {cart,fractions,corners,reciprocal};
}

// A boundary bond is displayed up to each crossed face on the two sides of the cell.
export function cellGeometry(cell,coords,elements,bondRadius,drawBonds) {
  const {cart,fractions,corners,reciprocal}=lattice(cell);
  const scaled=coords.map(p=>fractions(p).map(x=>x-Math.floor(x)));
  const points=scaled.map(cart), outside=[], links=[];
  if(!drawBonds)return {points,corners,outside,links};
  const radii=elements.map(bondRadius), distance=2.38*Math.max(...radii);
  const faceMargin=reciprocal.map(v=>distance*Math.hypot(...v));
  const buckets=new Map(), bucketOf=p=>p.map(x=>Math.floor(x/distance));
  for(let j=0;j<points.length;j++){
    const shifts=scaled[j].map((x,k)=>{
      const choices=[0];
      if(x<faceMargin[k])choices.push(1);
      if(1-x<faceMargin[k])choices.push(-1);
      return choices;
    });
    for(const x of shifts[0])for(const y of shifts[1])for(const z of shifts[2]){
      const shifted=x!==0||y!==0||z!==0;
      const frac=shifted?scaled[j].map((v,k)=>v+[x,y,z][k]):scaled[j], pos=shifted?cart(frac):points[j];
      const key=bucketOf(pos).join(',');
      if(!buckets.has(key))buckets.set(key,[]);
      buckets.get(key).push({j,pos,frac,shifted});
    }
  }
  for(let i=0;i<points.length;i++){
    const p=points[i],f=scaled[i],[x,y,z]=bucketOf(p);
    for(let u=x-1;u<=x+1;u++)for(let v=y-1;v<=y+1;v++)for(let w=z-1;w<=z+1;w++){
      for(const {j,pos,frac,shifted} of buckets.get([u,v,w].join(','))??[]){
        if(!shifted && j>=i)continue;
        const d2=(p[0]-pos[0])**2+(p[1]-pos[1])**2+(p[2]-pos[2])**2;
        if(d2<0.0625||d2>(1.19*(radii[i]+radii[j]))**2)continue;
        if(!shifted){links.push([i,j]);continue;}
        let end=1;
        for(let k=0;k<3;k++){
          const delta=frac[k]-f[k];
          if(frac[k]<0)end=Math.min(end,-f[k]/delta);
          if(frac[k]>1)end=Math.min(end,(1-f[k])/delta);
        }
        const along=t=>p.map((val,k)=>val+t*(pos[k]-val));
        outside.push([p,along(Math.min(end,.5)),i]);
        if(end>.5)outside.push([along(.5),along(end),j]);
      }
    }
  }
  return {points,corners,outside,links};
}
