"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
import spacy


class Tokenizer:
    # 독일어 및 영어 문장을 토큰화하는 클래스

    def __init__(self):
        # SpaCy의 사전 학습된 토크나이저 로드 (독일어, 영어)
        self.spacy_de = spacy.load('de_core_news_sm')  # 독일어 토크나이저
        self.spacy_en = spacy.load('en_core_web_sm')   # 영어 토크나이저

    def tokenize_de(self, text):
        """
        Tokenizes German text from a string into a list of strings
        :param text: 독일어 문장 (str)
        :return: 토큰화된 단어 리스트 (List[str])
        """
        return [tok.text for tok in self.spacy_de.tokenizer(text)]
        # SpaCy 토크나이저가 반환한 토큰들을 문자열 리스트로 반환

    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings
        :param text: 영어 문장 (str)
        :return: 토큰화된 단어 리스트 (List[str])
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]
        # SpaCy 영어 토크나이저 사용
