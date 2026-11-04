from backend.workflow import build_workflow
from test_workflow import FakeModel,initial,finish,SOURCE


def scorer(values):
    seq=iter(values)
    def score(text):
        return {'available':True,'ai_class_probability':next(seq),'locked_threshold':0.34}
    return score


def replies(n):
    return sum(([finish(SOURCE+f' Draft {chr(65+i)}.'),{'passed':True,'issues':[]}] for i in range(n)),[])


def test_regression_rejected_and_target_reached():
    result=build_workflow(FakeModel(replies(2)),paper_scorer=scorer([.70,.84,.30])).invoke(initial())
    assert result['status']=='completed' and result['rounds']==2
    assert result['current_paper']['ai_class_probability']==.30
    assert result['stop_reason']=='target_reached'


def test_no_progress_retains_original():
    result=build_workflow(FakeModel(replies(2)),paper_scorer=scorer([.70,.84,.80])).invoke(initial())
    assert result['stop_reason']=='no_progress'
    assert result['status']=='needs_review'
    assert result['draft']==SOURCE and result['edits']==[]
    assert result['current_paper']['ai_class_probability']==.70


def test_best_candidate_survives_later_regression():
    result=build_workflow(FakeModel(replies(3)),paper_scorer=scorer([.70,.50,.60,.65])).invoke(initial())
    assert result['stop_reason']=='no_progress'
    assert result['current_paper']['ai_class_probability']==.50
    assert result['draft'].endswith('Draft A.')
    assert len(result['round_records'])==3


def test_round_limit():
    result=build_workflow(FakeModel(replies(5)),paper_scorer=scorer([.90,.80,.70,.60,.50,.40])).invoke(initial(),{'recursion_limit':60})
    assert result['stop_reason']=='max_rounds' and result['status']=='needs_review'


def test_low_score_with_fidelity_failure_not_accepted():
    model=FakeModel([finish(),{'passed':False,'issues':['Meaning changed']},finish(),{'passed':False,'issues':['Meaning changed']}])
    result=build_workflow(model,paper_scorer=scorer([.7,.2,.1])).invoke(initial())
    assert result['status']=='needs_review' and not result['best_record']
