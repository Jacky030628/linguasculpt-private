import pytest
from backend.planning import build_plan,evaluate_plan
from backend.paper import resources,score


def row(name,coef,z):
    return dict(feature=name,coefficient=coef,standardized_value=z,contribution=coef*z,
                raw_value=z,training_mean=0,training_scale=1,block_weight=1,clipped=False)


def test_direction_uses_sign_and_local_contribution():
    p={'available':True,'linguistic_contributions':[row('1_1_mtld',2,1),row('3_1_avg_sentence_length',-3,-1),row('ignored',100,-1)]}
    plan=build_plan('Example sentence.',p)
    assert [x['feature'] for x in plan['targets']]==['3_1_avg_sentence_length','1_1_mtld']
    assert plan['targets'][0]['direction']=='increase'
    assert plan['targets'][1]['direction']=='decrease'
    assert all(x['feature']!='ignored' for x in plan['targets'])


def test_failed_feature_feedback_and_plan_evaluation():
    before={'available':True,'linguistic_contributions':[row('1_1_mtld',2,1)]}
    after={'linguistic_contributions':[row('1_1_mtld',2,1.5)]}
    plan=build_plan('Some words.',before,{'largest_linguistic_increases':[{'feature':'1_1_mtld','logit_change':1}]})
    assert plan['targets'][0]['avoid_repeating_failed_approach']
    evaluated=evaluate_plan(plan,before,after)
    assert evaluated['targets'][0]['logit_change']==1
    assert not evaluated['targets'][0]['improved']


def test_fitted_weights_reconstruct_local_contribution():
    result=score(' '.join(['The study may improve recall but does not establish causality.']*9))
    for item in result['linguistic_contributions']:
        assert item['contribution']==pytest.approx(item['coefficient']*item['standardized_value']*item['block_weight'])
    assert result['logit']==pytest.approx(result['intercept']+sum(result['block_contributions'].values()))
    plan=build_plan('A source sentence.',result)
    assert len(plan['targets'])<=5


def test_five_distinct_groups_merge_related_features():
    features=['1_1_mtld','1_3_root_ttr','3_1_avg_sentence_length','4_11_comma_ratio',
              '2_7_pronoun_1st_ratio','3_6_passive_voice_ratio','4_1_flesch_reading_ease']
    paper={'available':True,'linguistic_contributions':[row(f,1,10-i) for i,f in enumerate(features)]}
    text='We consider the proposal, which was carefully reviewed by our team. Another sentence follows.'
    plan=build_plan(text,paper)
    assert len(plan['targets'])==5
    assert len({t['family'] for t in plan['targets']})==5
    assert len(plan['targets'][0]['supporting_features'])==2
    for target in plan['targets']:
        assert target['optional']
        for e in target['evidence']: assert text[e['start']:e['end']]==e['quote']


def test_missing_text_evidence_not_padded_to_five():
    paper={'available':True,'linguistic_contributions':[row('2_7_pronoun_1st_ratio',1,1)]}
    assert build_plan('Quiet mornings bring relief.',paper)['targets']==[]
