export function createLibrary(items, choose, preview) {
  const get=id=>document.getElementById(id);
  const grid=get('molecules');
  get('library-count').textContent=items.length+' examples';
  for(const item of items){
    const btn=document.createElement('button');btn.type='button';btn.className='sample';btn.dataset.id=item.id;
    btn.title=item.description||item.source||item.name;btn.setAttribute('aria-pressed','false');
    const thumb=document.createElement('span');thumb.className='preview';
    const description=document.createElement('span'),name=document.createElement('strong'),detail=document.createElement('small');
    name.textContent=item.name;detail.textContent=item.category||item.group;
    const count=document.createElement('small');count.className='sample-formula';
    count.textContent=(item.atom_count??item.elements?.length)?.toLocaleString()+' atoms';
    description.append(name,detail,count);btn.append(thumb,description);grid.append(btn);
    if(item.structure_file){
      const im=new Image();im.alt='';im.loading='lazy';im.src='paper-data/previews/'+item.id+'.webp';thumb.append(im);
    }else{
      const canvas=document.createElement('canvas');canvas.setAttribute('aria-hidden','true');thumb.append(canvas);
      const scene=preview(canvas);scene.load(item.elements,item.coordinates,item.cell);
    }
    btn.onclick=()=>choose(item);
  }
  return {select(id){for(const b of grid.querySelectorAll('.sample'))b.setAttribute('aria-pressed',String(b.dataset.id===id));}};
}
