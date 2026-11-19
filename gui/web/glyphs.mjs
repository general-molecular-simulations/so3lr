const marks={
 desk:'<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M15 3v18m-9-3h5"/>',
 clock:'<path d="M3 11A9 9 0 1 1 6 19m-3-14v6h6m3-4v5l4 2"/>',
 book:'<path d="M3 5c3-1 6-.7 9 1.3C15 4.3 18 4 21 5v14c-3-1-6-.6-9 1.5-3-2.1-6-2.5-9-1.5zM12 6v14"/>',
 github:'<path d="M9 20c-4 1.1-4.5-1.8-6-2m12 3v-3c0-1-.3-1.6-.8-2 3-.4 5.7-1.5 5.7-6 0-1.3-.4-2.3-1.1-3.1.2-.8.4-2.2-.3-3.5-1.4-.1-2.7.7-3.5 1.3a12 12 0 0 0-6.4 0A6 6 0 0 0 5 3.4C4.4 4.8 4.6 6 4.9 6.9A4.5 4.5 0 0 0 3.7 10c0 4.5 2.7 5.6 5.7 6-.7.5-.9 1.3-.9 2.2V21"/>',
 import:'<path d="M12 16V3m-5 5 5-5 5 5M4 15v5h16v-5"/>',
 save:'<path d="M12 3v12m-4-4 4 4 4-4M4 15v5h16v-5"/>',
 crosshair:'<circle cx="12" cy="12" r="6.5"/><path d="M12 2v5m0 10v5M2 12h5m10 0h5"/>',
 cube:'<path d="m12 2 9 5v10l-9 5-9-5V7zM3 7l9 5 9-5m-9 5v10"/>',
 down:'<path d="m6 9 6 6 6-6"/>', minus:'<path d="M5 12h14"/>',plus:'<path d="M5 12h14m-7-7v14"/>',
 fit:'<path d="M8 4H4v4m12-4h4v4M4 16v4h4m8 0h4v-4"/><circle cx="12" cy="12" r="3"/>',
 fullscreen:'<path d="M3 9V3h6m6 0h6v6M3 15v6h6m12-6v6h-6"/>',
 hand:'<rect x="6" y="2.5" width="12" height="19" rx="6"/><path d="M12 6v4"/>',
 controls:'<path d="M3 7h4m5 0h9M3 17h10m5 0h3"/><circle cx="9.5" cy="7" r="2.5"/><circle cx="15.5" cy="17" r="2.5"/>',
 wave:'<path d="M2 12h5l3-8 4 16 3-8h5"/>',
 orbit:'<ellipse cx="12" cy="12" rx="10" ry="5" transform="rotate(-35 12 12)"/><circle cx="12" cy="12" r="2"/>',
 play:'<path d="m7 4 13 8-13 8z" fill="currentColor" stroke-width="1"/>',pause:'<path d="M8 4v16m8-16v16" stroke-width="4"/>',
 code:'<path d="m7 7-5 5 5 5m10-10 5 5-5 5m-4-14-2 18"/>',
 library:'<path d="M3 4h4v16H3zm6 0h4v16H9zm6 1 4-1 3 15-4 1z"/>',
 external:'<path d="M13 4h7v7m-1-6-9 9m-2-9H5v14h14v-3"/>',x:'<path d="m5 5 14 14M19 5 5 19"/>',check:'<path d="m4 12 5 5L20 6"/>'
};
export function glyph(id){const fragment=document.createElement('template');fragment.innerHTML='<svg class="glyph" aria-hidden="true" viewBox="0 0 24 24">'+marks[id]+'</svg>';return fragment.content.firstChild;}
export function mountGlyphs(){document.querySelectorAll('[data-glyph]').forEach(el=>el.replaceWith(glyph(el.dataset.glyph)));}
