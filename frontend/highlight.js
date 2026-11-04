// Exact word/punctuation diff. Text is always rendered with safe DOM text nodes.
(function (root) {
  function changedSegments(source, draft) {
    const tokenize = text => Array.from(text.matchAll(/[\p{L}\p{N}_]+(?:['’][\p{L}\p{N}_]+)*|[^\s]/gu), m => ({text:m[0],start:m.index,end:m.index+m[0].length}));
    const a=tokenize(source), b=tokenize(draft);
    let prefix=0;
    while(prefix<a.length && prefix<b.length && a[prefix].text===b[prefix].text) prefix++;
    let ae=a.length,be=b.length;
    while(ae>prefix && be>prefix && a[ae-1].text===b[be-1].text) {ae--;be--;}
    const n=ae-prefix,m=be-prefix;
    const added=new Set();
    if(n*m>4000000) {
      for(let j=prefix;j<be;j++) added.add(j);
    } else {
      const rows=Array.from({length:n+1},()=>new Uint32Array(m+1));
      for(let i=n-1;i>=0;i--) for(let j=m-1;j>=0;j--)
        rows[i][j]=a[prefix+i].text===b[prefix+j].text ? rows[i+1][j+1]+1 : Math.max(rows[i+1][j],rows[i][j+1]);
      let i=0,j=0;
      while(i<n && j<m) {
        if(a[prefix+i].text===b[prefix+j].text) {i++;j++;}
        else if(rows[i+1][j]>=rows[i][j+1]) i++;
        else {added.add(prefix+j);j++;}
      }
      while(j<m) {added.add(prefix+j);j++;}
    }
    const segments=[];
    let cursor=0;
    for(let j=0;j<b.length;j++) {
      const token=b[j];
      if(cursor<token.start) segments.push({text:draft.slice(cursor,token.start),changed:false});
      segments.push({text:token.text,changed:added.has(j)});cursor=token.end;
    }
    if(cursor<draft.length) segments.push({text:draft.slice(cursor),changed:false});
    return segments;
  }
  function render(output,source,draft) {
    const fragment=document.createDocumentFragment();
    for(const segment of changedSegments(source,draft)) {
      if(segment.changed) {
        const span=document.createElement('span');span.className='user-added-change';
        span.textContent=segment.text;fragment.append(span);
      } else fragment.append(document.createTextNode(segment.text));
    }
    output.replaceChildren(fragment);
  }
  root.proseHighlight={changedSegments,render};
  if(typeof module!=='undefined') module.exports=root.proseHighlight;
})(globalThis);
