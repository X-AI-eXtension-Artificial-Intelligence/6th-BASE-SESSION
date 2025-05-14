import torch
import torch.nn as nn
import torch.optim as optim
import math
import time

from dataset import load_dataset, get_device
from model import Encoder, Decoder, Transformer

# 디바이스 설정
device = get_device()

# 데이터셋 로드
SRC, TRG, train_iterator, valid_iterator, test_iterator = load_dataset(device=device)

'''
하이퍼파라미터 설정 및 모델 초기화
'''
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HIDDEN_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

# 모델 정의
encoder = Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
decoder = Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

model = Transformer(encoder, decoder, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

# 모델 파라미터 개수 출력
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Model has {count_parameters(model):,} trainable parameters')

# 모델 가중치 초기화
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights)

'''
학습 및 평가 함수 정의
'''
# Optimizer 및 Loss 함수 설정
LEARNING_RATE = 0.0005
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX).to(device)

# 모델 학습 함수
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src.to(device)
        trg = batch.trg.to(device)

        optimizer.zero_grad()

        # 출력의 마지막 인덱스(<eos>)는 제외, 입력은 <sos>부터
        output, _ = model(src, trg[:, :-1])

        # output: [batch_size, trg_len - 1, output_dim]
        # trg: [batch_size, trg_len]

        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        # Loss 계산 및 역전파
        loss = criterion(output, trg)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# 모델 평가 함수
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in iterator:
            src = batch.src.to(device)
            trg = batch.trg.to(device)

            output, _ = model(src, trg[:, :-1])

            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# 학습 시간 계산 함수
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

'''
모델 학습 실행
'''
N_EPOCHS = 100
CLIP = 1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer_german_to_english.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
    print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')
