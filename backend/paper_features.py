"""Verbatim feature functions from the current ACL notebook; no training cells executed."""

import re, math

from collections import Counter

import numpy as np

import textstat

from wordfreq import zipf_frequency

LONG_WORD_LEN = 7

LOW_FREQ_ZIPF_THRESHOLD = 3.0

# [Cell 7] 语言学特征词表常量定义
WH_WORDS = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how"}
DEMONSTRATIVES = {"this", "that", "these", "those"}
INDEFINITE_PRONOUNS = {
    "someone", "somebody", "something", "anyone", "anybody", "anything",
    "everyone", "everybody", "everything", "noone", "nobody", "nothing",
    "each", "either", "neither", "one", "ones", "none"
}
FIRST_PRON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
SECOND_PRON = {"you", "your", "yours"}
THIRD_PRON = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"}
NEGATIONS = {"not", "n't", "never", "no", "none", "nobody", "nothing", "neither", "nowhere"}
WH_TAGS = {"WDT", "WP", "WP$", "WRB"}

# [Cell 8] 词汇多样性与信息熵计算函数
def safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def syllable_count_word(w: str) -> int:
    w = re.sub(r"[^a-zA-Z]", "", w)
    return textstat.syllable_count(w) if w else 0

def pos_entropy_from_counts(pos_counts: Counter) -> float:
    total = sum(pos_counts.values())
    if total == 0:
        return 0.0
    ps = [c / total for c in pos_counts.values() if c > 0]
    return -sum(p * math.log(p + 1e-12) for p in ps)

def content_function_ratio_from_pos(pos_counts: Counter) -> float:
    content = (
        pos_counts.get("NOUN", 0) + pos_counts.get("VERB", 0) +
        pos_counts.get("ADJ", 0) + pos_counts.get("ADV", 0)
    )
    function = (
        pos_counts.get("DET", 0) + pos_counts.get("PRON", 0) + pos_counts.get("ADP", 0) +
        pos_counts.get("CCONJ", 0) + pos_counts.get("SCONJ", 0) +
        pos_counts.get("AUX", 0) + pos_counts.get("PART", 0)
    )
    return safe_div(content, function)

def root_ttr(tokens):
    if not tokens:
        return 0.0
    return len(set(tokens)) / math.sqrt(len(tokens))

def hapax_ratio(tokens):
    if not tokens:
        return 0.0
    c = Counter(tokens)
    hapax = sum(1 for _, n in c.items() if n == 1)
    return hapax / len(tokens)

def yule_k(tokens):
    N = len(tokens)
    if N == 0:
        return 0.0
    freqs = Counter(tokens)
    ff = Counter(freqs.values())
    sum_i2_vi = sum((i**2) * vi for i, vi in ff.items())
    return 1e4 * (sum_i2_vi - N) / (N * N)

def simpson_index(tokens):
    if not tokens:
        return 0.0
    freqs = Counter(tokens)
    N = len(tokens)
    psq = sum((n / N) ** 2 for n in freqs.values())
    return 1.0 - psq

def maas_ttr(tokens):
    if not tokens:
        return 0.0
    N = len(tokens)
    V = len(set(tokens))
    if N <= 1 or V <= 1:
        return 0.0
    lnN = math.log(N)
    return (lnN - math.log(V)) / (lnN * lnN) if lnN else 0.0

def mtld(tokens, ttr_threshold=0.72):
    if len(tokens) < 10:
        return 0.0
    factors = 0
    types = set()
    token_count = 0
    for w in tokens:
        token_count += 1
        types.add(w)
        ttr = len(types) / token_count
        if ttr <= ttr_threshold:
            factors += 1
            types = set()
            token_count = 0
    if token_count > 0:
        ttr = len(types) / token_count
        if ttr != 1:
            factors += (1 - ttr) / (1 - ttr_threshold)
    return len(tokens) / factors if factors else 0.0

def hdd(tokens, sample_size=42):
    if len(tokens) < 10:
        return 0.0
    freqs = Counter(tokens)
    N = len(tokens)
    s = min(sample_size, N)
    from math import comb
    denom = comb(N, s)
    if denom == 0:
        return 0.0
    hdd_sum = 0.0
    for f in freqs.values():
        if N - f >= s:
            p0 = comb(N - f, s) / denom
        else:
            p0 = 0.0
        hdd_sum += (1.0 - p0)
    return hdd_sum / s if s else 0.0

# [Cell 9] 单篇文档 65 维宏观语言学特征提取器
def extract_features_from_doc(doc) -> dict:
    toks = [t for t in doc if t.is_alpha]
    tokens = [t.text.lower() for t in toks]
    total_tokens = len(tokens)

    sents = list(doc.sents)
    sent_token_lens = []
    for sent in sents:
        sent_tokens = [t for t in sent if t.is_alpha]
        sent_token_lens.append(len(sent_tokens))
    avg_sentence_length = float(np.mean(sent_token_lens)) if sent_token_lens else 0.0
    std_sentence_length = float(np.std(sent_token_lens)) if sent_token_lens else 0.0
    total_sentences = len(sent_token_lens) if sent_token_lens else 0

    word_lens = [len(w) for w in tokens]
    avg_word_length = float(np.mean(word_lens)) if word_lens else 0.0
    std_word_length = float(np.std(word_lens)) if word_lens else 0.0
    long_word_ratio = safe_div(sum(1 for w in tokens if len(w) >= LONG_WORD_LEN), total_tokens)
    avg_syllables_per_word = safe_div(sum(syllable_count_word(w) for w in tokens), total_tokens)

    zipfs = [zipf_frequency(w, "en") for w in tokens] if tokens else []
    zipf_mean = float(np.mean(zipfs)) if zipfs else 0.0
    zipf_std = float(np.std(zipfs)) if zipfs else 0.0
    low_freq_word_ratio = safe_div(sum(1 for z in zipfs if z < LOW_FREQ_ZIPF_THRESHOLD), len(zipfs))

    difficult_words_count = textstat.difficult_words(doc.text)
    difficult_word_ratio = safe_div(difficult_words_count, total_tokens)

    pos_counts = Counter([t.pos_ for t in toks])
    noun_ratio = safe_div(pos_counts.get("NOUN", 0), total_tokens)
    verb_ratio = safe_div(pos_counts.get("VERB", 0), total_tokens)
    adj_ratio = safe_div(pos_counts.get("ADJ", 0), total_tokens)
    adv_ratio = safe_div(pos_counts.get("ADV", 0), total_tokens)
    proper_noun_ratio = safe_div(pos_counts.get("PROPN", 0), total_tokens)
    pos_entropy = pos_entropy_from_counts(pos_counts)

    article_ratio = safe_div(sum(1 for t in toks if t.lower_ in {"a", "an", "the"}), total_tokens)
    preposition_ratio = safe_div(pos_counts.get("ADP", 0), total_tokens)
    auxiliary_ratio = safe_div(pos_counts.get("AUX", 0), total_tokens)
    determiner_ratio = safe_div(pos_counts.get("DET", 0), total_tokens)
    conjunction_ratio = safe_div(pos_counts.get("CCONJ", 0) + pos_counts.get("SCONJ", 0), total_tokens)
    negation_ratio = safe_div(sum(1 for t in toks if t.lower_ in NEGATIONS), total_tokens)

    pronoun_1st_ratio = safe_div(sum(1 for t in toks if t.lower_ in FIRST_PRON), total_tokens)
    pronoun_2nd_ratio = safe_div(sum(1 for t in toks if t.lower_ in SECOND_PRON), total_tokens)
    pronoun_3rd_ratio = safe_div(sum(1 for t in toks if t.lower_ in THIRD_PRON), total_tokens)
    pronoun_it_ratio = safe_div(sum(1 for t in toks if t.lower_ == "it"), total_tokens)
    demonstrative_pronoun_ratio = safe_div(sum(1 for t in toks if t.lower_ in DEMONSTRATIVES), total_tokens)
    indefinite_pronoun_ratio = safe_div(sum(1 for t in toks if t.lower_ in INDEFINITE_PRONOUNS), total_tokens)

    content_function_ratio = content_function_ratio_from_pos(pos_counts)

    text = doc.text
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    gunning_fog_index = textstat.gunning_fog(text)
    smog_index = textstat.smog_index(text)
    automated_readability_index = textstat.automated_readability_index(text)
    coleman_liau_index = textstat.coleman_liau_index(text)
    dale_chall_readability_score = textstat.dale_chall_readability_score(text)

    punct_chars = set(list(""".,;:—-()[]{}!?'\""""))
    punct_count = sum(1 for ch in text if ch in punct_chars)
    comma = text.count(",")
    semicolon = text.count(";")
    colon = text.count(":")
    dash = text.count("-") + text.count("—")
    parenthesis = text.count("(") + text.count(")")
    contraction = len(re.findall(r"\b\w+'\w+\b", text))
    punctuation_ratio = safe_div(punct_count, total_tokens)

    sent_initial_conj = 0
    for sent in sents:
        first_alpha = next((t for t in sent if t.is_alpha), None)
        if first_alpha and first_alpha.pos_ in {"CCONJ", "SCONJ"}:
            sent_initial_conj += 1
    sentence_initial_conjunction_ratio = safe_div(sent_initial_conj, total_sentences)

    verb_aux_tokens = [t for t in toks if t.pos_ in {"VERB", "AUX"}]
    total_verbs = len(verb_aux_tokens)

    past = sum(1 for t in verb_aux_tokens if t.tag_ == "VBD")
    pres = sum(1 for t in verb_aux_tokens if t.tag_ in {"VBP", "VBZ"})
    past_tense_ratio = safe_div(past, total_verbs)
    present_tense_ratio = safe_div(pres, total_verbs)

    perfect = 0
    for t in doc:
        if t.lemma_ == "have" and t.pos_ == "AUX":
            head = t.head
            if head.pos_ in {"VERB", "AUX"} and head.tag_ == "VBN":
                perfect += 1
                continue
            been = [ch for ch in t.children if ch.lemma_ == "be" and ch.pos_ == "AUX" and ch.tag_ == "VBN"]
            if been:
                for bc in been:
                    h2 = bc.head
                    if h2.pos_ in {"VERB", "AUX"} and h2.tag_ in {"VBN", "VBG"}:
                        perfect += 1
                        break
    perfect_aspect_ratio = safe_div(perfect, total_verbs)

    def is_passive_predicate(v):
        if v.pos_ not in {"VERB", "AUX"}:
            return False
        if any(ch.dep_ == "nsubjpass" for ch in v.children):
            return True
        if any(ch.dep_ == "auxpass" for ch in v.children):
            return True
        return False

    def has_by_agent(v):
        if any(ch.dep_ == "agent" for ch in v.children):
            return True
        for ch in v.children:
            if ch.dep_ == "prep" and ch.lower_ == "by":
                return True
        return False

    passive_sent = 0
    by_passive_sent = 0
    agentless_passive_sent = 0
    for sent in sents:
        sent_tokens = list(sent)
        passive_preds = [t for t in sent_tokens if is_passive_predicate(t)]
        if passive_preds:
            passive_sent += 1
            if any(has_by_agent(v) for v in passive_preds):
                by_passive_sent += 1
            else:
                agentless_passive_sent += 1

    passive_voice_ratio = safe_div(passive_sent, total_sentences)
    by_passive_ratio = safe_div(by_passive_sent, total_sentences)
    agentless_passive_ratio = safe_div(agentless_passive_sent, total_sentences)

    that_clause_sent = 0
    wh_clause_sent = 0
    infinitive_clause_sent = 0
    participial_clause_sent = 0

    for sent in sents:
        sent_tokens = list(sent)

        that_marks = [
            t for t in sent_tokens
            if t.lower_ == "that" and t.dep_ == "mark"
            and t.head.dep_ in {"ccomp", "advcl", "acl", "relcl"}
        ]
        if that_marks:
            that_clause_sent += 1

        wh_tokens = [t for t in sent_tokens if t.tag_ in WH_TAGS]
        clause_heads = [t for t in sent_tokens if t.dep_ in {"ccomp", "advcl", "relcl"}]
        found_wh = False
        if wh_tokens and clause_heads:
            wh_set = set(wh_tokens)
            for h in clause_heads:
                if any(w in set(h.subtree) for w in wh_set):
                    found_wh = True
                    break
        if found_wh:
            wh_clause_sent += 1

        to_tokens = [t for t in sent_tokens if t.tag_ == "TO"]
        found_inf = False
        for to in to_tokens:
            h = to.head
            if h.pos_ == "VERB" and h.dep_ in {"xcomp", "acl", "advcl", "ccomp"}:
                found_inf = True
                break
        if found_inf:
            infinitive_clause_sent += 1

        part_heads = [
            t for t in sent_tokens
            if t.tag_ in {"VBG", "VBN"} and t.dep_ in {"advcl", "acl", "relcl", "xcomp"}
        ]
        if part_heads:
            participial_clause_sent += 1

    that_clause_ratio = safe_div(that_clause_sent, total_sentences)
    wh_clause_ratio = safe_div(wh_clause_sent, total_sentences)
    infinitive_clause_ratio = safe_div(infinitive_clause_sent, total_sentences)
    participial_clause_ratio = safe_div(participial_clause_sent, total_sentences)

    def token_depth(tok):
        d = 0
        cur = tok
        while cur.head is not None and cur != cur.head:
            d += 1
            cur = cur.head
            if d > 200:
                break
        return d

    depths = []
    dep_dists = []
    for sent in sents:
        for t in sent:
            if t.is_space:
                continue
            depths.append(token_depth(t))
            if sent.start <= t.head.i < sent.end:
                dep_dists.append(abs(t.i - t.head.i))

    avg_dependency_tree_depth = float(np.mean(depths)) if depths else 0.0
    avg_dependency_distance = float(np.mean(dep_dists)) if dep_dists else 0.0

    nominalizations = sum(1 for w in tokens if w.endswith(("tion", "sion", "ment", "ness", "ity", "ism")))
    gerunds = sum(1 for t in toks if t.tag_ == "VBG")
    nominalization_ratio = safe_div(nominalizations, total_tokens)
    gerund_ratio = safe_div(gerunds, total_tokens)

    return {
        "1_1_mtld": mtld(tokens),
        "1_2_hdd": hdd(tokens),
        "1_3_root_ttr": root_ttr(tokens),
        "1_4_maas_ttr": maas_ttr(tokens),
        "1_5_yule_k": yule_k(tokens),
        "1_6_simpson_index": simpson_index(tokens),
        "1_7_hapax_ratio": hapax_ratio(tokens),
        "1_8_avg_word_length": avg_word_length,
        "1_9_std_word_length": std_word_length,
        "1_10_long_word_ratio": long_word_ratio,
        "1_11_avg_syllables_per_word": avg_syllables_per_word,
        "1_12_zipf_mean": zipf_mean,
        "1_13_zipf_std": zipf_std,
        "1_14_low_freq_word_ratio": low_freq_word_ratio,
        "1_15_difficult_word_ratio": difficult_word_ratio,
        "1_16_content_function_ratio": content_function_ratio,

        "2_1_article_ratio": article_ratio,
        "2_2_preposition_ratio": preposition_ratio,
        "2_3_auxiliary_ratio": auxiliary_ratio,
        "2_4_conjunction_ratio": conjunction_ratio,
        "2_5_negation_ratio": negation_ratio,
        "2_6_determiner_ratio": determiner_ratio,
        "2_7_pronoun_1st_ratio": pronoun_1st_ratio,
        "2_8_pronoun_2nd_ratio": pronoun_2nd_ratio,
        "2_9_pronoun_3rd_ratio": pronoun_3rd_ratio,
        "2_10_pronoun_it_ratio": pronoun_it_ratio,
        "2_11_demonstrative_pronoun_ratio": demonstrative_pronoun_ratio,
        "2_12_indefinite_pronoun_ratio": indefinite_pronoun_ratio,
        "2_13_noun_ratio": noun_ratio,
        "2_14_verb_ratio": verb_ratio,
        "2_15_adj_ratio": adj_ratio,
        "2_16_adv_ratio": adv_ratio,
        "2_17_proper_noun_ratio": proper_noun_ratio,
        "2_18_pos_entropy": pos_entropy,

        "3_1_avg_sentence_length": avg_sentence_length,
        "3_2_std_sentence_length": std_sentence_length,
        "3_3_past_tense_ratio": past_tense_ratio,
        "3_4_present_tense_ratio": present_tense_ratio,
        "3_5_perfect_aspect_ratio": perfect_aspect_ratio,
        "3_6_passive_voice_ratio": passive_voice_ratio,
        "3_7_agentless_passive_ratio": agentless_passive_ratio,
        "3_8_by_passive_ratio": by_passive_ratio,
        "3_9_that_clause_ratio": that_clause_ratio,
        "3_10_wh_clause_ratio": wh_clause_ratio,
        "3_11_infinitive_clause_ratio": infinitive_clause_ratio,
        "3_12_participial_clause_ratio": participial_clause_ratio,
        "3_13_avg_dependency_tree_depth": avg_dependency_tree_depth,
        "3_14_avg_dependency_distance": avg_dependency_distance,

        "4_1_flesch_reading_ease": flesch_reading_ease,
        "4_2_flesch_kincaid_grade": flesch_kincaid_grade,
        "4_3_gunning_fog_index": gunning_fog_index,
        "4_4_smog_index": smog_index,
        "4_5_automated_readability_index": automated_readability_index,
        "4_6_coleman_liau_index": coleman_liau_index,
        "4_7_dale_chall_readability_score": dale_chall_readability_score,
        "4_8_nominalization_ratio": nominalization_ratio,
        "4_9_gerund_ratio": gerund_ratio,
        "4_10_punctuation_ratio": punctuation_ratio,
        "4_11_comma_ratio": safe_div(comma, total_tokens),
        "4_12_semicolon_ratio": safe_div(semicolon, total_tokens),
        "4_13_colon_ratio": safe_div(colon, total_tokens),
        "4_14_dash_ratio": safe_div(dash, total_tokens),
        "4_15_parenthesis_ratio": safe_div(parenthesis, total_tokens),
        "4_16_contraction_ratio": safe_div(contraction, total_tokens),
        "4_17_sentence_initial_conjunction_ratio": sentence_initial_conjunction_ratio,
    }
