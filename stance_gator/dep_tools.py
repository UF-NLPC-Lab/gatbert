# STL
from typing import Tuple, List
# 3rd Party
from spacy.tokens import Doc

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