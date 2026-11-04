from backend.change_records import change_records


def test_complete_changes_and_full_context():
    a='Title\nThe old wording is clear. An unchanged sentence. Another old sentence.'
    b='Title\nThe new wording is clear. An unchanged sentence. Another better sentence.'
    rows=change_records(a,b)
    assert len(rows)==2
    assert rows[0]['before']=='The old wording is clear.'
    assert '「old」改为「new」' in rows[0]['action_taken']
    assert rows[1]['after'].strip()=='Another better sentence.'


def test_insert_delete_and_split_not_lost():
    assert change_records('Keep. Remove.','Keep.')[0]['after']==''
    assert change_records('Keep.','Keep. Add.')[0]['before']==''
    rows=change_records('First part and second part.','First part. Second part.')
    assert 'First part. Second part.'==''.join(r['after'] for r in rows)
    assert change_records('Same\ntext.','Same text.')


def assert_lossless(a,b):
    records=change_records(a,b)
    atoms=sorted([c for r in records for c in r['changes']],key=lambda c:c['source_start'])
    cursor=0;out=[]
    for c in atoms:
        assert a[c['source_start']:c['source_end']]==c['before']
        out.extend([a[cursor:c['source_start']],c['after']]);cursor=c['source_end']
    out.append(a[cursor:])
    assert ''.join(out)==b
    assert records==sorted(records,key=lambda r:r['source_start'])


def test_all_differences_can_reconstruct_final_exactly():
    pairs=[('','New text.'),('Remove all.',''),('A. B. C.','C. A. B.'),
           ('Dr. Smith measured 3.14 units. Next sentence.','Dr. Smith measured 3.15 units. Next sentence!'),
           ('Title\n\nOne sentence. Two sentences.','Title\nOne revised sentence; two joined sentences.'),
           (' a  b\n\tc ','a b\n\n c'),('Repeat. Repeat.','Repeat.'),
           ('同一句，原文。','同一句！改文。')]
    for a,b in pairs:assert_lossless(a,b)


def test_abbreviation_does_not_break_sentence():
    rows=change_records('Dr. Smith agrees. Next sentence.','Dr. Smith disagrees. Next sentence.')
    assert rows[0]['before']=='Dr. Smith agrees.'
