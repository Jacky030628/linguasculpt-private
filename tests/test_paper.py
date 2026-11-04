import numpy as np
from backend.paper import score, resources


def test_paper_inference_and_attribution():
    text = ' '.join(['The study may improve recall but does not establish causality.'] * 9)
    result = score(text)
    assert result['available']
    assert len(result['raw_features']) == 65
    assert result['retained_features'] == 43
    assert abs(result['locked_threshold'] - 0.34) < 1e-12
    assert abs(1/(1+np.exp(-result['logit']))-result['ai_class_probability']) < 1e-10
    assert len(result['linguistic_contributions']) == 43


def test_out_of_scope_not_scored():
    assert not score('A short text.')['available']
    assert not score('word ' * 1501)['available']
