from typing import List, Tuple

import spacy
from spacy.tokens import Doc

class __LazySpacyDict(dict):
    def __missing__(self, lang):
        if lang == 'en':
            self[lang] = spacy.load('en_core_web_sm')
        elif lang == 'de':
            self[lang] = spacy.load('de_core_news_sm')
        elif lang == 'it':
            self[lang] = spacy.load('it_core_news_sm')
        elif lang == 'fr':
            self[lang] = spacy.load('fr_core_news_sm')
        else:
            raise ValueError(f"Unsupported language {lang}")
        return self[lang]

SPACY_PIPES = __LazySpacyDict()

# This function is called a lot. More efficient probably to not re-make the set every time
__tags = {"NOUN", "PNOUN", "ADJ", "ADV"}
def extract_lemmas(pipeline, sentence: str):
    global __tags
    return [t.lemma_.lower() for t in pipeline(sentence) if t.pos_ in __tags]

__cut_relations = {
    "advcl",
    # "csubj",
    # "ccomp",
}
def get_spans(spacy_doc: Doc) -> List[Tuple[int, int]]:
    global __cut_relations
    spans = []
    for sent in spacy_doc.sents:
        root = sent.root
        root_i = root.i

        sent_start = sent[0].i
        sent_end = sent[-1].i + 1

        span_a = None
        span_b = None
        for child in filter(lambda c: c.dep_ in __cut_relations, root.children):
            child_subtree = list(child.subtree)
            continuous_span = all(child_subtree[j].i == (child_subtree[j-1].i + 1) for j in range(1, len(child_subtree)))
            if not continuous_span:
                continue
            span_start = child_subtree[0].i
            span_end = child_subtree[-1].i + 1
            if root_i < span_start:
                span_a = (sent_start, span_start)
                span_b = (span_start, sent_end)
            else:
                assert root_i >= span_end
                span_a = (sent_start, span_end)
                span_b = (span_end, sent_end)
        if span_a is not None:
            spans.append(span_a)
            spans.append(span_b)
        else:
            spans.append((sent_start, sent_end))
    return spans