import torch
from torch.utils.data import Dataset, DataLoader
import os
import urllib.request
import re
from typing import List, Tuple

class WikiTextDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
        
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x), torch.tensor(y)

def download_wikitext2():
    """WikiText2 데이터셋을 다운로드합니다."""
    base_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/"
    files = ['train.txt', 'valid.txt', 'test.txt']
    
    if not os.path.exists('data'):
        os.makedirs('data')
    
    for file in files:
        url = base_url + file
        save_path = os.path.join('data', file)
        if not os.path.exists(save_path):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(url, save_path)
    
    return [os.path.join('data', file) for file in files]

def build_vocab(text):
    """텍스트에서 어휘 사전을 구축합니다."""
    words = re.findall(r'\w+', text.lower())
    vocab = {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab

def load_wikitext2(batch_size=32, seq_length=20):
    # 데이터셋 다운로드
    train_path, valid_path, test_path = download_wikitext2()
    
    # 데이터 로드 및 전처리
    def load_and_process(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    
    train_text = load_and_process(train_path)
    valid_text = load_and_process(valid_path)
    test_text = load_and_process(test_path)
    
    # 어휘 사전 구축
    vocab = build_vocab(train_text)
    
    # 텍스트를 토큰 ID로 변환
    def text_to_ids(text, vocab):
        words = re.findall(r'\w+', text.lower())
        return [vocab.get(word, vocab['<unk>']) for word in words]
    
    train_data = text_to_ids(train_text, vocab)
    valid_data = text_to_ids(valid_text, vocab)
    test_data = text_to_ids(test_text, vocab)
    
    # 데이터셋 생성
    train_dataset = WikiTextDataset(train_data, seq_length)
    valid_dataset = WikiTextDataset(valid_data, seq_length)
    test_dataset = WikiTextDataset(test_data, seq_length)
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader, vocab

if __name__ == "__main__":
    # 테스트
    train_loader, val_loader, test_loader, vocab = load_wikitext2()
    print(f"Vocabulary size: {len(vocab)}")
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"Input shape: {data.shape}")
        print(f"Target shape: {target.shape}")
        if batch_idx == 2:  # 처음 3개 배치만 출력
            break 