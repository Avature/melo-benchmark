import spacy


class Lemmatizer:

    def __init__(self, language: str):
        self.spacy_model = self._load_spacy_model(language)

    def preprocess(self, text: str) -> str:
        text = self.spacy_model(text.lower())
        tokens = [
            token.lemma_
            for token in text
            if not token.is_stop and not token.is_punct
        ]
        text = ' '.join(tokens)
        return text

    def _load_spacy_model(self, language: str):
        if language == "en":
            return spacy.load('en_core_web_sm')
        elif language == "da":
            return spacy.load('da_core_news_sm')
        elif language == "de":
            return spacy.load('de_core_news_sm')
        elif language == "es":
            return spacy.load('es_core_news_sm')
        elif language == "fr":
            return spacy.load('fr_core_news_sm')
        elif language == "hr":
            return spacy.load('hr_core_news_sm')
        elif language == "it":
            return spacy.load('it_core_news_sm')
        elif language == "lt":
            return spacy.load('lt_core_news_sm')
        elif language == "nl":
            return spacy.load('nl_core_news_sm')
        elif language == "no":
            return spacy.load('nb_core_news_sm')
        elif language == "pt":
            return spacy.load('pt_core_news_sm')
        elif language == "pl":
            return spacy.load('pl_core_news_sm')
        elif language == "ro":
            return spacy.load('ro_core_news_sm')
        elif language == "sl":
            return spacy.load('sl_core_news_sm')
        elif language == "sv":
            return spacy.load('sv_core_news_sm')
        elif language == "zh":
            return spacy.load('zh_core_web_sm')
        else:
            raise NotImplementedError(f"No model for language `{language}`")
