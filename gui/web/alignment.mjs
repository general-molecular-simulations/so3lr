// Horn's quaternion solution to a least-squares rigid fit on the heavy atoms.
// The renderer centers each frame, so only the proper rotation is needed here.
export function alignedView(points,reference,elements,vector=null) {
  const center=[0,0,0], target=[0,0,0];
  let count=0;
  for(let i=0;i<points.length;i++)if(elements[i]!=='H'){
    count++;
    for(let k=0;k<3;k++){center[k]+=points[i][k];target[k]+=reference[i][k];}
  }
  const s=Array.from({length:3},()=>[0,0,0]);
  for(let i=0;i<points.length;i++)if(elements[i]!=='H'){
    for(let a=0;a<3;a++)for(let b=0;b<3;b++)
      s[a][b]+=(points[i][a]-center[a]/count)*(reference[i][b]-target[b]/count);
  }
  const [[xx,xy,xz],[yx,yy,yz],[zx,zy,zz]]=s;
  const h=[[xx+yy+zz,yz-zy,zx-xz,xy-yx],[yz-zy,xx-yy-zz,xy+yx,xz+zx],
           [zx-xz,xy+yx,yy-xx-zz,yz+zy],[xy-yx,xz+zx,yz+zy,zz-xx-yy]];
  const axes=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>Number(i===j)));
  // Jacobi diagonalization of the symmetric 4 × 4 matrix, including half turns.
  for(let sweep=0;sweep<12;sweep++){
    let changed=false;
    for(let p=0;p<3;p++)for(let q=p+1;q<4;q++){
      const off=h[p][q];
      if(Math.abs(off)<1e-12)continue;
      changed=true;
      const delta=(h[q][q]-h[p][p])/(2*off);
      const tangent=(delta<0?-1:1)/(Math.abs(delta)+Math.hypot(1,delta));
      const angle=Math.atan(tangent);
      const c=Math.cos(angle),s=Math.sin(angle);
      h[p][p]-=tangent*off;
      h[q][q]+=tangent*off;
      h[p][q]=h[q][p]=0;
      for(let k=0;k<4;k++){
        if(k!==p && k!==q){
          const u=h[k][p],v=h[k][q];
          h[k][p]=h[p][k]=c*u-s*v;h[k][q]=h[q][k]=s*u+c*v;
        }
        const u=axes[k][p],v=axes[k][q];
        axes[k][p]=c*u-s*v;axes[k][q]=s*u+c*v;
      }
    }
    if(!changed)break;
  }
  const values=h.map((row,i)=>row[i]),best=values.indexOf(Math.max(...values));
  const [w,x,y,z]=axes.map(row=>row[best]);
  function rotate([a,b,c]){
    const u=2*(y*c-z*b),v=2*(z*a-x*c),t=2*(x*b-y*a);
    return [a+w*u+y*t-z*v,b+w*v+z*u-x*t,c+w*t+x*v-y*u];
  }
  return {positions:points.map(rotate),vector:vector && rotate(vector)};
}
