import json
import os
import pandas as pd
import numpy as np
from src.constants import CACHE_DIR, WORD_FILE
from tqdm import tqdm
from pprint import pprint

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

from src.cefr import CEFR_LEVELS
from collections import Counter
import spacy
import numpy as np
from datasets import Dataset

from datasets.utils.logging import enable_progress_bar

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

WORD_FILE_CACHE = os.path.join(CACHE_DIR, 'word_ranks.json')


POS_TAGS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CONJ",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
    # "SPACE",
]


HIGH_FREQ_DEPS = [
    "acomp",
    "advcl",
    "advmod",
    "amod",
    "attr",
    "aux",
    "auxpass",
    "cc",
    "ccomp",
    "compound",
    "conj",
    "det",
    "dobj",
    "expl",
    "mark",
    "neg",
    "npadvmod",
    "nsubj",
    "nsubjpass",
    "nummod",
    "pcomp",
    "pobj",
    "poss",
    "prep",
    "prt",
    "punct",
    "relcl",
    "xcomp",
]

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation and numbers
    text = ''.join([ch for ch in text if ch not in string.punctuation and not ch.isdigit()])

    # Tokenize
    words_orig = word_tokenize(text)

    # Remove stopwords and lemmatize
    # words = [lemmatizer.lemmatize(word) for word in words_orig if word not in stopwords.words('english')]
    words = [
        dict(orig=dict(word=word, index=i),
             proc=lemmatizer.lemmatize(word)) for i,word in enumerate(words_orig) if word not in stopwords.words('english')]

    return words #' '.join(words)


def load_word_data():
    ## --- UPDATE --- ##
    # Use a word frequency file to get word ranks and cefr levels
    boundaries = [250, 500, 1000, 2000, 5000, 10000, 20000]

    if os.path.exists(WORD_FILE_CACHE):
        print("Loading cached word data")
        words = json.load(open(WORD_FILE_CACHE, 'r'))
        # words = pd.read_csv(WORD_FILE_CACHE)
        return words, boundaries

    os.makedirs(CACHE_DIR, exist_ok=True)

    words = pd.read_csv(WORD_FILE)

    # Preprocessing `word` column in words dataframe, similar to what we did with text
    words = words[~ words['word'].isna()]
    words['wordRaw'] = words['word']
    preprocess_word = lambda x: ' '.join(a['proc'] for a in preprocess_text(x))

    words['word'] = words['wordRaw'].apply(preprocess_word)
    words = words[words['word'] != ""] # filter out stopwords and stuff
    # Assuming df is a DataFrame with 'rank' and 'frequency' columns
    words = words.sort_values('ofr_freq_rank')  # make sure DataFrame is sorted based on ranks

    # group by word and take the lowe rrank and lowest cef score
    # words = words.groupby('word').agg({
    #     'freq_rank': 'min',
    #     'cefr': lambda z: CEFR_LEVELS.index(cefr_predicted_label)
    # }).reset_index()

    from collections import defaultdict
    word_ranks = defaultdict(dict)
    for i, d in tqdm(words.iterrows(), total=len(words)):
        word = d['word']
        freq = d['ofr_freq_rank']
        cefr = d['cefr_predicted_label'] if d['cefr_predicted_label'] != 'C' else 'C1'

        word_ranks[word]['freq_rank'] = freq
        word_ranks[word]['cefr'] = CEFR_LEVELS.index(cefr)
        word_ranks[word]['rank_bin'] = 0
        for b in boundaries:
            word_ranks[word]['rank_bin'] += int(freq > b)

    json.dump(word_ranks, open(WORD_FILE_CACHE, 'w'), indent=2)
    return word_ranks, boundaries

all_words, rank_bins = load_word_data()

def featurise(data, sent_scorer=None, trace=False):
    try:
        text = data['text']
        word_fts, word_ft_traces = wordlevel_features(text)

        sent_fts, sent_ft_traces = sentlevel_features(text, sent_scorer=sent_scorer)

        assert word_fts.keys().isdisjoint(sent_fts.keys())

        ret = dict(**word_fts, **sent_fts)
        ret_trace = dict(**word_ft_traces, **sent_ft_traces)
        if trace:
            return ret, ret_trace
        else:
            return ret
    except ValueError as e:
        print('Error in featurise')
        print(str(e))
        print(text)
        return dict()


def wordlevel_features(text):
    # words: list of words
    words_in_text = preprocess_text(text)
    # words_in_text = text_clean.split()

    n_bins = len(rank_bins) + 1
    bin_names = [f'rank_{0}_{rank_bins[0]}'] + \
                [f'rank_{rank_bins[i]}_{rank_bins[i+1]}' for i in range(n_bins - 2)] + \
                [f'rank_{rank_bins[-1]}p'] + ['rank_NA']
    unsup_feats, rank_traces = fts_rank_bin(words_in_text, n_bins, all_words)
    unsup_feats = unsup_feats / unsup_feats.sum() if unsup_feats.sum() > 0 else unsup_feats

    rank_d = {
        k:v for k,v in zip(bin_names, unsup_feats)
    }
    rank_traces_d = {
        k:v for k,v in zip(bin_names, rank_traces)
    }

    cefr_bins = [f'cefr_{l}' for l in CEFR_LEVELS[:-1]] + ['cefr_NA']
    cefr_cnts = np.zeros(len(cefr_bins))
    cefr_traces = [[] for _ in range(len(cefr_bins))]
    for word_dict in words_in_text:
        word = word_dict['proc']
        orig = word_dict['orig']
        word_info = all_words.get(word, {})
        cefr_bin = word_info.get('cefr', len(cefr_bins) - 1)
        cefr_cnts[cefr_bin] += 1
        cefr_traces[cefr_bin].append(orig)


    cefr_cnts = cefr_cnts / cefr_cnts.sum() if cefr_cnts.sum() > 0 else cefr_cnts
    cefr_d = {
        k: v for k,v in zip(cefr_bins, cefr_cnts)
    }

    cefr_traces_d = {
        k:v for k,v in zip(cefr_bins, cefr_traces)
    }

    ret_fts = dict(
        **rank_d,
        **cefr_d,
    )
    ret_traces = dict(
        **rank_traces_d,
        **cefr_traces_d
    )

    return ret_fts, ret_traces



def predict(model, scaler, feats, texts):
    data = Dataset.from_pandas(pd.DataFrame({'text': texts}))
    enable_progress_bar()
    data = data.map(featurise, batched=False)

    def data_collator(dataset):
        return np.stack([dataset[f] for f in feats], axis=1)
    Xx = data_collator(data)
    Xx = scaler.transform(Xx)
    return model.predict(Xx), data


def score_sentences(sentences, sent_scorer=None):
    if sent_scorer is None:
        return {}, {}

    preds, _ = predict(sent_scorer['model'], sent_scorer['scaler'], sent_scorer['feats'], [str(sent) for sent in sentences])
    # create binned count of preds
    # round preds to +- 0.5
    preds = np.round(preds, 0).astype(int)
    preds = np.clip(preds, -1, 6)
    bins = ['a1-', 'a1', 'a2', 'b1', 'b2', 'c1', 'c2', 'c2+']
    bin_cnts = np.zeros(len(bins))
    score_trace = [[] for _ in range(len(bins))]
    for p, sent in zip(preds, sentences):
        bin_cnts[p] += 1
        score_trace[p].append(str(sent))

    bin_cnts = bin_cnts / bin_cnts.sum() if bin_cnts.sum() > 0 else bin_cnts
    bin_d = {
        f'sent_pred_{k}': v for k,v in zip(bins, bin_cnts)
    }

    score_trace_d = {
        f'sent_pred_{k}': v for k,v in zip(bins, score_trace)
    }

    return bin_d, score_trace_d


def sentlevel_features(text, sent_scorer=None):
    doc = nlp(text.strip())
    sentences = [sent for sent in doc.sents if str(sent).strip() != ""]
    nlp_sent = sentences
    # sentences: list of senteces
    # nlp_sent = [nlp(sent) for sent in sentences]
    all_stats = [sentence_stats(sent) for sent in nlp_sent]
    all_stats = [sent for sent in all_stats if sent['max_depth'] is not None]

    sent_score_bins, sent_score_trace = score_sentences(sentences, sent_scorer=sent_scorer)

    def sent_data_collator(data, key):
        stats = [d[key] for d in data]
        trace = [dict(stat=d[key], index=i, text=sentences[i]) for i,d in enumerate(data)]
        # trace = sorted(trace, key=lambda x: x['stat'])
        return stats, trace

    def counter_to_arr(ctr, cols):
        return [ctr[v] for v in cols]

    pos_counts, _ = sent_data_collator(all_stats, 'pos_counts')
    pos_counts_arr = np.stack(list(map(lambda x: counter_to_arr(x, POS_TAGS),
                                       pos_counts)))

    pos_count_denom = pos_counts_arr.sum(axis=1, keepdims=True)
    pos_count_denom[pos_count_denom == 0] = 1
    pos_counts_arr = pos_counts_arr / pos_count_denom # TODO: proportion
    pos_avgs = np.mean(pos_counts_arr, axis=0)
    pos_avgs_d = {f'avg_pos_{k.lower()}': v for k,v in zip(POS_TAGS, pos_avgs)}

    # dep_counts = data_collator(all_stats, 'dep_cnts')
    # dep_counts_arr = np.stack(list(map(lambda x: counter_to_arr(x, HIGH_FREQ_DEPS),
    #                                     dep_counts)))
    # dep_counts_arr = dep_counts_arr / dep_counts_arr.sum(axis=1, keepdims=True) # TODO: proportion
    # dep_avgs = np.mean(dep_counts_arr, axis=0)
    # dep_avgs_d = {f'avg_dep_{k.lower()}': v for k,v in zip(HIGH_FREQ_DEPS, dep_avgs)}
    dep_avgs_d = {}
    slen_stats, slen_trace = sent_data_collator(all_stats, 'sent_len')
    avg_sent_len = np.mean(slen_stats)

    depth_stats, depth_trace = sent_data_collator(all_stats, 'max_depth')
    avg_max_depth = np.mean(depth_stats)

    max_children_stats, max_children_trace = sent_data_collator(all_stats, 'max_children')
    avg_max_children = np.mean(max_children_stats)

    n_unique_deps_stats, n_unique_deps_trace = sent_data_collator(all_stats, 'n_unique_deps')
    avg_n_unique_deps = np.mean(n_unique_deps_stats)

    ret = dict(
        avg_sent_len=avg_sent_len,
        avg_max_depth=avg_max_depth,
        avg_max_children=avg_max_children,
        avg_n_unique_deps=avg_n_unique_deps,
        **pos_avgs_d,
        **dep_avgs_d,
        **sent_score_bins
    )

    ret_trace = dict(
        avg_sent_len=slen_trace,
        avg_max_depth=depth_trace,
        avg_max_children=max_children_trace,
        avg_n_unique_deps=n_unique_deps_trace,
        **sent_score_trace
    )

    return ret, ret_trace


# words, rank_bins = load_word_data(word_file)
def fts_rank_bin(words, n_bins, all_words):
    cnts = np.zeros(n_bins + 1) # last for unkown words
    traces = [[] for _ in range(n_bins + 1)]

    for word_dict in words:
        word = word_dict['proc']
        orig = word_dict['orig']
        word_info = all_words.get(word, {})
        rank_bin = word_info.get('rank_bin', n_bins)
        cnts[rank_bin] += 1
        traces[rank_bin].append(orig)

    return cnts, traces



def trace_depth(token, depth=0):
    if depth > 100:
        print('Depth exceeeded')
        print('Token: ', token)
        raise ValueError
    if token.dep_ == "ROOT":
        return 0
    return 1 + trace_depth(token.head, depth + 1)

# def trace_depth(token, depth=0):
#     depths = [trace_depth(child, depth + 1) for child in token.children]
#     return max(depths) if len(depths) > 0 else depth

def sentence_stats(sentDoc):
    pos = [token.pos_ for token in sentDoc]
    pos_counts = Counter(pos)
    sent_len = len(sentDoc)

    # dependency tree stats
    try:
        depths = [trace_depth(token) for token in sentDoc]
        max_depth = max(depths) / sent_len
    except ValueError as e:
        print('Max depth exceeeded')
        print(str(e))
        print(sentDoc)
        # max_depth = None
        raise e

    child_counts = [len(list(token.children)) for token in sentDoc]
    max_children = max(child_counts) / sent_len

    deps = [token.dep_ for token in sentDoc]
    n_unique_deps = len(set(deps))

    dep_cnts = {k: deps.count(k) for k in HIGH_FREQ_DEPS}

    return dict(
        pos_counts=pos_counts,
        sent_len=sent_len,
        max_depth=max_depth,
        max_children=max_children,
        n_unique_deps=n_unique_deps,
        dep_cnts=dep_cnts
    )


