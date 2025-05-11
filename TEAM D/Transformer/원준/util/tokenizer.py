import spacy  # 자연어처리용 라이브러리 (언어별 tokenizer 포함)

class Tokenizer:

    def __init__(self):
        # spaCy에서 제공하는 독일어 및 영어 tokenizer 모델 불러오기
        self.spacy_de = spacy.load('de_core_news_sm')  # 독일어 tokenizer
        self.spacy_en = spacy.load('en_core_web_sm')   # 영어 tokenizer

    def tokenize_de(self, text):
        """
        독일어 문장을 토큰 단위 리스트로 변환하는 함수
        예: "Ich bin ein Student." → ['Ich', 'bin', 'ein', 'Student', '.']
        """
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        """
        영어 문장을 토큰 단위 리스트로 변환하는 함수
        예: "I am a student." → ['I', 'am', 'a', 'student', '.']
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]
