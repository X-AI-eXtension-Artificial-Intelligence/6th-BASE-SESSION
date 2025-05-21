"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
'''
from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer

tokenizer = Tokenizer()
loader = DataLoader(ext=('.en', '.de'),
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>')

train, valid, test = loader.make_dataset()
loader.build_vocab(train_data=train, min_freq=2)
train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                     batch_size=batch_size,
                                                     device=device)

src_pad_idx = loader.source.vocab.stoi['<pad>']
trg_pad_idx = loader.target.vocab.stoi['<pad>']
trg_sos_idx = loader.target.vocab.stoi['<sos>']

enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)
'''
# data.py

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

# 🔹 학습용 데이터 (확장된 문장쌍 리스트)
train_data = [
    ("I love you", "Ich liebe dich"),
    ("How are you", "Wie geht es dir"),
    ("Good morning", "Guten Morgen"),
    ("Thank you", "Danke"),
    ("Yes", "Ja"),
    ("No", "Nein"),
    ("I am happy", "Ich bin glücklich"),
    ("I am sad", "Ich bin traurig"),
    ("What is your name", "Wie heißt du"),
    ("My name is John", "Ich heiße John"),
    ("Nice to meet you", "Schön dich zu treffen"),
    ("I don't understand", "Ich verstehe nicht"),
    ("Do you speak English", "Sprichst du Englisch"),
    ("Help me please", "Hilf mir bitte"),
    ("Where is the bathroom", "Wo ist die Toilette"),
    ("I am hungry", "Ich habe Hunger"),
    ("I am tired", "Ich bin müde"),
    ("Can you help me", "Kannst du mir helfen"),
    ("See you later", "Bis später"),
    ("Good night", "Gute Nacht"),
    ("I like this", "Ich mag das"),
    ("This is good", "Das ist gut"),
    ("This is bad", "Das ist schlecht"),
    ("What time is it", "Wie spät ist es"),
    ("I am from Korea", "Ich komme aus Korea"),
    ("I live in Seoul", "Ich wohne in Seoul"),
    ("Please wait", "Bitte warte"),
    ("Let's go", "Lass uns gehen"),
    ("Be careful", "Sei vorsichtig"),
    ("Congratulations", "Herzlichen Glückwunsch"),
    ("I need water", "Ich brauche Wasser"),
    ("I am lost", "Ich habe mich verirrt"),
    ("Where is the station", "Wo ist der Bahnhof"),
    ("Turn left", "Biegen Sie links ab"),
    ("Turn right", "Biegen Sie rechts ab"),
    ("Go straight", "Gehen Sie geradeaus"),
    ("Stop here", "Halten Sie hier an"),
    ("How much is it", "Wie viel kostet es"),
    ("I want this", "Ich will das"),
    ("Do you have this", "Haben Sie das"),
    ("I am learning German", "Ich lerne Deutsch"),
    ("Do you like music", "Magst du Musik"),
    ("I like to travel", "Ich reise gern"),
    ("I am a student", "Ich bin Student"),
    ("What do you do", "Was machst du beruflich"),
    ("I work in IT", "Ich arbeite in der IT"),
    ("It's cold today", "Es ist kalt heute"),
    ("It's hot today", "Es ist heiß heute"),
    ("It's raining", "Es regnet"),
    ("It's snowing", "Es schneit"),
    ("See you tomorrow", "Bis morgen"),
    ("Good afternoon", "Guten Tag"),
    ("Excuse me", "Entschuldigung"),
    ("I'm sorry", "Es tut mir leid"),
    ("Welcome", "Willkommen"),
    ("Happy birthday", "Alles Gute zum Geburtstag"),
    ("Enjoy your meal", "Guten Appetit"),
    ("Bless you", "Gesundheit"),
    ("I agree", "Ich stimme zu"),
    ("I disagree", "Ich stimme nicht zu"),
    ("I’m bored", "Mir ist langweilig"),
    ("I'm busy", "Ich bin beschäftigt"),
    ("Call me", "Ruf mich an"),
    ("Text me", "Schreib mir"),
    ("What’s the problem", "Was ist das Problem"),
    ("No problem", "Kein Problem"),
    ("Good job", "Gut gemacht"),
    ("I’m ready", "Ich bin bereit"),
    ("Wait a moment", "Warte einen Moment"),
    ("Take your time", "Lass dir Zeit"),
    ("Don’t worry", "Mach dir keine Sorgen"),
    ("I'm okay", "Mir geht’s gut"),
    ("That’s fine", "Das ist in Ordnung"),
    ("Are you okay", "Geht es dir gut"),
    ("I hope so", "Ich hoffe es"),
    ("I’m not sure", "Ich bin nicht sicher"),
    ("Can you repeat that", "Kannst du das wiederholen"),
    ("Speak slowly", "Sprich langsam"),
    ("What does that mean", "Was bedeutet das"),
    ("Can I help you", "Kann ich dir helfen"),
    ("Where are you", "Wo bist du"),
    ("Where are we", "Wo sind wir"),
    ("It’s far", "Es ist weit"),
    ("It’s near", "Es ist in der Nähe"),
    ("Come here", "Komm her"),
    ("Go there", "Geh dorthin"),
    ("I forgot", "Ich habe es vergessen"),
    ("I remember", "Ich erinnere mich"),
    ("One moment", "Einen Moment"),
    ("I'm thirsty", "Ich habe Durst"),
    ("I'm full", "Ich bin satt"),
    ("Do you understand", "Verstehst du"),
    ("I understand", "Ich verstehe"),
    ("Can I come in", "Darf ich reinkommen"),
    ("What should I do", "Was soll ich tun"),
    ("How do you say this", "Wie sagt man das"),
    ("What’s your favorite food", "Was ist dein Lieblingsessen"),
    ("I love coffee", "Ich liebe Kaffee"),
    ("I hate waiting", "Ich hasse es zu warten")
]

valid_data = [
    ("Where is the airport", "Wo ist der Flughafen"),
    ("What is your favorite color", "Was ist deine Lieblingsfarbe"),
    ("Do you like books", "Magst du Bücher"),
    ("I like reading", "Ich lese gern"),
    ("Can you swim", "Kannst du schwimmen")
]    
# 토크나이저 및 vocab 생성
def tokenizer(text):
    return text.lower().strip().split()

def build_vocab(sentences):
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    idx = 4
    for sent in sentences:
        for token in tokenizer(sent):
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

src_vocab = build_vocab([src for src, _ in train_data + valid_data])
trg_vocab = build_vocab([trg for _, trg in train_data + valid_data])

# 텐서화
def numericalize(sentence, vocab):
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokenizer(sentence)] + [vocab["<eos>"]]

# Dataset 클래스
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, data, src_vocab, trg_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, trg = self.data[idx]
        src_ids = numericalize(src, self.src_vocab)
        trg_ids = numericalize(trg, self.trg_vocab)
        return torch.tensor(src_ids), torch.tensor(trg_ids)

# collate_fn
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=src_vocab["<pad>"])
    trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=trg_vocab["<pad>"])
    return src_batch, trg_batch

# DataLoader 정의
batch_size = 2
train_loader = DataLoader(TranslationDataset(train_data, src_vocab, trg_vocab), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(TranslationDataset(valid_data, src_vocab, trg_vocab), batch_size=1, shuffle=False, collate_fn=collate_fn)