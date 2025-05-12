"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""

import spacy  # spacy: 고성능 자연어처리 라이브러리


class Tokenizer:
    """
    독일어와 영어 문장을 spacy를 이용해 토큰화

    - 'de_core_news_sm': 독일어 소형 모델
    - 'en_core_web_sm' : 영어 소형 모델
    - 각 언어에 대해 별도 토크나이저를 구성하여 언어별 처리를 가능하게 함
    """

    def __init__(self):
        # 독일어 토크나이저
        # 예: "Ich bin ein Berliner" -> ['Ich', 'bin', 'ein', 'Berliner']
        self.spacy_de = spacy.load('de_core_news_sm')

        # 영어 토크나이저
        # 예: "I am a student" -> ['I', 'am', 'a', 'student']
        self.spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(self, text):
        """
        주어진 독일어 문장을 토큰 리스트로 분할

        입력: text: 문자열 (예: "Ich bin ein Berliner")
        출력: 문자열 리스트 (예: ['Ich', 'bin', 'ein', 'Berliner'])
        """
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        """
        주어진 영어 문장을 토큰 리스트로 분할

        입력: text: 문자열 (예: "I am a student")
        출력: 문자열 리스트 (예: ['I', 'am', 'a', 'student'])
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]
