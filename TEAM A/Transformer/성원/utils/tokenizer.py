""" 
역 모델 학습을 위한 토크나이저(Tokenizer) 클래스를 정의한 거야.
주요 역할은 영어와 독일어 문장을 토큰(단어) 리스트로 변환
"""

import spacy


class Tokenizer:

    def __init__(self):
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(self, text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]


