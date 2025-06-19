import spacy

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
 