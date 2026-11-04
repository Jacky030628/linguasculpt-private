from backend.workflow import build_workflow
from backend.tools import analyze_paper_features
from test_workflow import FakeModel, initial, SOURCE


def test_scored_candidate_is_submitted_without_extra_editor_call(monkeypatch):
    candidate=SOURCE+' Additional wording.'
    score={'available':True,'ai_class_probability':.2,'locked_threshold':.34}
    monkeypatch.setattr(analyze_paper_features,'func',lambda text:score)
    calls=[]
    def source_score(text):
        calls.append(text)
        return dict(score,ai_class_probability=.7)
    model=FakeModel([{'action':'analyze_paper_features','arguments':{'text':candidate}}, {'passed':True,'issues':[]}])
    result=build_workflow(model,paper_scorer=source_score,eager_candidates=True).invoke(initial())
    assert result['status']=='completed'
    assert result['draft']==candidate and result['rounds']==1 and result['steps']==1
    assert calls==[SOURCE]  # candidate measurement is reused, not recomputed
    assert [u['stage'] for u in result['usage_log']]==['editor','semantic_review']


def test_source_tool_does_not_trigger_fake_candidate(monkeypatch):
    monkeypatch.setattr(analyze_paper_features,'func',lambda text:{'available':False})
    model=FakeModel([{'action':'analyze_paper_features','arguments':{'text':SOURCE}},
        {'action':'finish','draft':SOURCE},{'passed':True,'issues':[]}])
    result=build_workflow(model,eager_candidates=True).invoke(initial())
    assert result['steps']==2
