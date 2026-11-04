"""Complete source/final-text comparisons, with factual Chinese descriptions."""
import re
from difflib import SequenceMatcher


ABBREVIATIONS={'mr.','mrs.','ms.','dr.','prof.','sr.','jr.','vs.','etc.','e.g.','i.e.','al.'}


def sentence_spans(text):
    spans=[];start=0
    for match in re.finditer(r'\n+|[.!?]+["”\']?(?=\s|$)',text):
        end=match.end()
        if not match.group().startswith('\n'):
            word=text[start:end].split()[-1].lower().strip('"”\'')
            if word in ABBREVIATIONS or re.fullmatch(r'(?:[a-z]\.)+',word):continue
        spans.append((start,end));start=end
    if start<len(text):spans.append((start,len(text)))
    return spans


def chunks(text):
    return [text[a:b].strip() for a,b in sentence_spans(text) if text[a:b].strip()]


def describe(before,after):
    if not before: return '新增以下内容。'
    if not after: return '删除以下内容。'
    a,b=before.split(),after.split()
    changes=[]
    for tag,i,j,k,l in SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
        old,new=' '.join(a[i:j]),' '.join(b[k:l])
        if tag=='replace': changes.append(f'将「{old}」改为「{new}」')
        elif tag=='insert': changes.append(f'新增「{new}」')
        elif tag=='delete': changes.append(f'删除「{old}」')
    # Do not truncate the list: every token change is accounted for.
    return '；'.join(changes)+'。'


def change_records(source,draft):
    if source==draft:return []
    token=re.compile(r"\s+|\w+(?:['’]\w+)*|[^\w\s]",re.UNICODE)
    a,b=list(token.finditer(source)),list(token.finditer(draft))
    sa,sb=sentence_spans(source),sentence_spans(draft)
    def offset(tokens,index,size):return tokens[index].start() if index<len(tokens) else size
    def context(spans,start,end):
        touched=[(a,b) for a,b in spans if a<end and b>start]
        return (touched[0][0],touched[-1][1]) if touched else (start,end)
    records=[]
    for tag,i,j,k,l in SequenceMatcher(None,[m.group() for m in a],[m.group() for m in b],autojunk=False).get_opcodes():
        if tag=='equal':continue
        x,y=offset(a,i,len(source)),offset(a,j,len(source))
        u,v=offset(b,k,len(draft)),offset(b,l,len(draft))
        change=dict(before=source[x:y],after=draft[u:v],source_start=x,source_end=y,output_start=u,output_end=v,type=tag)
        left,right=context(sa,x,y),context(sb,u,v)
        if records and (left[0]<records[-1]['source_end'] or right[0]<records[-1]['output_end']):
            r=records[-1];r['source_end']=max(r['source_end'],left[1]);r['output_end']=max(r['output_end'],right[1]);r['changes'].append(change)
        else:records.append(dict(source_start=left[0],source_end=left[1],output_start=right[0],output_end=right[1],changes=[change]))
    def visible(s):return s.replace('\r','\\r').replace('\n','\\n').replace('\t','\\t').replace(' ','␠') if s and not s.strip() else s
    for r in records:
        r.update(before=source[r['source_start']:r['source_end']],after=draft[r['output_start']:r['output_end']],change_count=len(r['changes']),type='edit')
        descriptions=[]
        for c in r['changes']:
            before,after=visible(c['before']),visible(c['after'])
            kind='空白／格式' if (c['before'].isspace() or c['after'].isspace()) else ('标点' if not re.search(r'\w',c['before']+c['after']) else '文字')
            descriptions.append(f'将{kind}「{before}」改为「{after}」' if before and after else (f'新增{kind}「{after}」' if after else f'删除{kind}「{before}」'))
        r['action_taken']='；'.join(descriptions)+'。'
        old_sentences=len([1 for x,y in sentence_spans(r['before']) if r['before'][x:y].strip()])
        new_sentences=len([1 for x,y in sentence_spans(r['after']) if r['after'][x:y].strip()])
        if old_sentences and new_sentences and old_sentences!=new_sentences:
            r['action_taken']=f'句组调整：{old_sentences} 个文本单元改为 {new_sentences} 个；'+r['action_taken']
    # Exact moved groups are represented once rather than as unrelated delete/add.
    for r in records:
        if r.get('_removed') or not r['before'].strip() or r['after'].strip():continue
        moved=next((t for t in records if t is not r and not t.get('_removed') and not t['before'].strip() and t['after'].strip()==r['before'].strip()),None)
        if moved:
            r.update(after=moved['after'],output_start=moved['output_start'],output_end=moved['output_end'],type='move',
                     action_taken=f'句组移位：原文字符位置 {r["source_start"]+1} → 成稿字符位置 {moved["output_start"]+1}。',
                     changes=r['changes']+moved['changes'],change_count=len(r['changes'])+len(moved['changes']))
            moved['_removed']=True
    records=[r for r in records if not r.get('_removed')]
    return records
