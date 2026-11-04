"""Local coefficient-aware planning. No extra model call or fabricated causal effect."""
import math
import re
from collections import Counter


def strategy(feature, direction):
    name=feature.split('_',2)[-1]
    if any(k in name for k in ['mtld','ttr','hdd','yule','simpson','hapax']):
        return ('Review unnecessary synonym churn and vocabulary consistency. Preserve natural repeated references; '
                'avoid decorative synonyms. Do not inject repetition or delete content to game diversity. '
                'These diversity metrics interact, so judge only the recomputed combined score.')
    if 'sentence_length' in name:
        return ('Consider clause joining/splitting at natural boundaries, preserving emphasis and all relations. '
                'Sentence-length standard deviation is a document statistic: do not assume longer sentences always help.')
    if any(k in name for k in ['comma','colon','dash','parenthesis','punctuation']):
        return ('Adjust punctuation only where an equivalent clause structure reads naturally. '
                'Do not remove definition boundaries or add punctuation solely to alter counts.')
    if 'pronoun' in name:
        return ('Review repeated pronoun constructions while retaining referents and narrative voice. '
                'Never replace we/you with people throughout the text automatically.')
    if any(k in name for k in ['clause','passive','dependency','tense','aspect']):
        return ('Consider equivalent clause structure or voice where agents, time, modality and causal force remain explicit. '
                'Never alter tense or certainty merely to meet a feature target.')
    if any(k in name for k in ['reading','grade','fog','smog','readability','liau']):
        return ('Review word and sentence complexity together. Keep the requested register and technical precision; '
                'do not inflate nominalizations or make prose awkward to move readability scores.')
    return ('Find a natural local construction affecting this feature; follow the indicated conditional numeric direction '
            'only when meaning and register permit. Skip if no defensible edit exists.')


def feature_family(feature):
    if any(k in feature for k in ['mtld','ttr','hdd','yule','simpson','hapax']): return 'lexical_diversity'
    if 'sentence_length' in feature: return 'sentence_rhythm'
    if any(k in feature for k in ['comma','colon','dash','parenthesis','punctuation','contraction']): return 'punctuation'
    if 'pronoun' in feature: return 'reference_and_voice'
    if any(k in feature for k in ['clause','passive','dependency','tense','aspect']): return 'syntax'
    if any(k in feature for k in ['reading','grade','fog','smog','readability','liau']): return 'readability'
    if any(k in feature for k in ['word_length','syllable','zipf','freq','difficult_word']): return 'word_complexity'
    return 'grammar_and_word_classes'


def evidence_for(text, family):
    """Heuristic inspection sites, with exact offsets; never causal attribution."""
    spans=[]
    for match in re.finditer(r'[^.!?]+(?:[.!?]+|$)',text):
        quote=match.group().strip()
        if not quote: continue
        start=match.start()+len(match.group())-len(match.group().lstrip())
        if family=='punctuation' and not re.search(r'[,;:—()\-\'’]',quote): continue
        if family=='reference_and_voice' and not re.search(r'\b(I|we|you|he|she|they|it|our|their|this|that|these|those)\b',quote,re.I): continue
        if family=='syntax' and not re.search(r'\b(that|which|who|when|while|because|if|to|was|were|is|are|have|has|had)\b',quote,re.I): continue
        spans.append({'quote':quote,'start':start,'end':start+len(quote)})
    # Long clauses give a useful inspection site; do not call it feature localization.
    spans.sort(key=lambda x:len(x['quote'].split()),reverse=True)
    return spans[:1]


def build_plan(text, paper, previous_feedback=None, limit=5):
    if not paper.get('available'):
        return {'targets':[], 'reason':'Paper score unavailable'}
    increases={x['feature']:x['logit_change'] for x in (previous_feedback or {}).get('largest_linguistic_increases',[])}
    ranked=sorted((x for x in paper.get('linguistic_contributions',[])
                   if x.get('contribution',0)>0.02 and abs(x.get('coefficient',0))>1e-9),
                  key=lambda x:x['contribution'],reverse=True)
    groups={}
    for x in ranked:
        groups.setdefault(feature_family(x['feature']),[]).append(x)
    # Rank by largest actual local contribution, preserving the fitted model basis.
    targets=[]
    for family, members in groups.items():
        if len(targets)>=min(5,max(0,limit)): break
        evidence=evidence_for(text,family)
        if not evidence: continue
        x=members[0]
        direction='decrease' if x['coefficient']>0 else 'increase'
        targets.append({k:x[k] for k in ['feature','coefficient','raw_value','standardized_value','contribution','training_mean','training_scale','block_weight','clipped'] if k in x})
        targets[-1].update(direction=direction,
            family=family,
            supporting_features=[{k:m[k] for k in ['feature','coefficient','standardized_value','contribution'] if k in m} for m in members],
            evidence=evidence,
            fidelity_constraint='Preserve proposition, polarity, modality, referents, conditions, examples and requested register. Leave already effective sentences unchanged.',
            optional=True,
            evidence_scope='Heuristic inspection location, not sentence-level attribution or a proven error.',
            rationale='For this fitted LR, holding other transformed inputs fixed, coefficient × change in standardized input determines the logit change.',
            operation=strategy(x['feature'],direction),
            previous_candidate_increase=increases.get(x['feature'],0),
            avoid_repeating_failed_approach=increases.get(x['feature'],0)>0.02)
    words=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",text.lower())
    sents=[s.strip() for s in re.split(r'(?<=[.!?])\s+',text) if s.strip()]
    examples=sorted(sents,key=lambda s:len(s.split()),reverse=True)[:3]
    threshold=paper.get('locked_threshold',.34)
    needed=paper.get('logit',0)-math.log(threshold/(1-threshold))
    return {'targets':targets,'logit_reduction_to_threshold':max(0,needed),
            'block_contributions':paper.get('block_contributions',{}),
            'document_evidence':{'long_sentence_examples':examples,'recurring_words':Counter(w for w in words if len(w)>4).most_common(8)},
            'evidence_scope':'Examples are text evidence, NOT sentence-level feature attributions.',
            'instructions':'Consider at most five distinct feature families; these are optional priorities, not a checklist. Check sign AND current value; do not rank by absolute coefficient alone. '
            'Related features are grouped; they may have conflicting directions. Skip a group if no natural, faithful edit is defensible. '
            'A clipped feature may require a large raw change before its score moves: skip unrealistic edits. '
            'Do not undo helpful negative contributions. Choose concrete edits within the supplied evidence; submit once for whole-model scoring. '
            'If a previous candidate increased a target contribution, avoid the same editing pattern. No universal human target values are assumed.'}


def evaluate_plan(plan, before, after):
    old={x['feature']:x for x in before.get('linguistic_contributions',[])}
    new={x['feature']:x for x in after.get('linguistic_contributions',[])}
    results=[]
    feature_targets={}
    for target in plan.get('targets',[]):
        for item in target.get('supporting_features',[target]): feature_targets[item['feature']]=item
    for target in feature_targets.values():
        name=target['feature']
        if name in old and name in new:
            delta=new[name]['contribution']-old[name]['contribution']
            results.append({'feature':name,'logit_change':delta,'improved':delta < -1e-9,
                            'raw_change':new[name]['raw_value']-old[name]['raw_value']})
    return {'targets':results,'block_changes':{k:after.get('block_contributions',{}).get(k,0)-v
                for k,v in before.get('block_contributions',{}).items()},
            'total_logit_change':after.get('logit',0)-before.get('logit',0),
            'note':'Measured changes, not proof that one editing operation caused the change.'}
