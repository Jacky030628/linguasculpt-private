"""Inference-only adapter for the supplied ACL experiment artifacts."""
from functools import lru_cache
from pathlib import Path
import re
import threading
import joblib
import numpy as np
import spacy
import nltk
from scipy.sparse import hstack, csr_matrix
from .paper_features import extract_features_from_doc

LOCK = threading.RLock()
PATH = Path(__file__).resolve().parents[1] / 'models' / 'paper_bundle.joblib'
nltk.data.path.insert(0, str(PATH.parent / 'nltk_data'))


@lru_cache(maxsize=1)
def resources():
    bundle = joblib.load(PATH)
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    return bundle, nlp


def score(text: str) -> dict:
    # Current notebook robust_text_cleaner, without clipping user documents.
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    count = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
    if not 80 <= count <= 1500:
        return {'available': False, 'reason': '论文评分适用长度为 80–1500 英文词；未截断原文或外推概率。', 'words': count}
    with LOCK:
        b, nlp = resources()
        raw = extract_features_from_doc(nlp(text))
        values = np.asarray([[raw.get(name, 0) for name in b['feature_names']]], dtype=np.float64)
        values = np.where(np.isfinite(values), values, 0).astype(np.float32)
        unbounded_scaled = b['scaler'].transform(values)
        scaled = np.clip(unbounded_scaled, -b['z_clip'], b['z_clip'])
        selected = scaled[:,b['kept_indices']]
        word = b['word_vectorizer'].transform([text]) * b['weights'][0]
        char = b['char_vectorizer'].transform([text]) * b['weights'][1]
        x = hstack([word,char,csr_matrix(selected*b['weights'][2])]).tocsr()
        model = b['fusion']
        probability = float(model.predict_proba(x)[0,list(model.classes_).index(1)])
        names = ['word:'+n for n in b['word_vectorizer'].get_feature_names_out()] + ['char:'+n for n in b['char_vectorizer'].get_feature_names_out()] + [b['feature_names'][i] for i in b['kept_indices']]
        contributions = x.multiply(model.coef_[0]).toarray()[0]
        top = np.argsort(contributions)[::-1][:12]
        ling_start = word.shape[1]+char.shape[1]
        ling = []
        for offset, index in enumerate(b['kept_indices']):
            i = ling_start + offset
            ling.append({'feature':names[i], 'contribution':float(contributions[i]),
                'raw_value':float(raw[names[i]]), 'coefficient':float(model.coef_[0,i]),
                'standardized_value':float(scaled[0,index]), 'block_weight':float(b['weights'][2]),
                'training_mean':float(b['scaler'].mean_[index]),'training_scale':float(b['scaler'].scale_[index]),
                'clipped':bool(abs(float(unbounded_scaled[0,index]))>=b['z_clip'])})
        ling.sort(key=lambda v: v['contribution'],reverse=True)
        logit = float(model.intercept_[0]+contributions.sum())
        return {'available':True,'model':'VIF-pruned fusion logistic regression',
                'ai_class_probability':probability,'locked_threshold':b['fusion_threshold'],
                'predicted_class':'AI' if probability >= b['fusion_threshold'] else 'Human',
                'words':count,'raw_features':{k:float(v) for k,v in raw.items()},
                'retained_features':len(b['kept_indices']), 'linguistic_contributions':ling,
                'top_positive_contributions':[{'feature':names[i],'contribution':float(contributions[i])} for i in top if contributions[i]>0],
                'intercept':float(model.intercept_[0]),'logit':logit,
                'block_contributions':{'word':float(contributions[:word.shape[1]].sum()),
                    'character':float(contributions[word.shape[1]:ling_start].sum()),
                    'linguistic':float(contributions[ling_start:].sum())},
                'note':'论文分类器输出，不代表作者身份事实或改写质量；局部贡献基于标准化、截断和加权后的输入。'}
