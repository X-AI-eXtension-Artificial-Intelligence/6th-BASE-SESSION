"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time

# 모델 내 학습 가능한 파라미터 수 계산 (분석용)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 가중치 초기화 함수 (Kaiming 방식: ReLU 기반 네트워크에 적합)
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

# Transformer 모델 인스턴스 생성
model = Transformer(
    src_pad_idx=src_pad_idx,         # source 문장에서 pad 위치를 가릴 때 사용
    trg_pad_idx=trg_pad_idx,         # target 문장에서 pad 위치 무시용
    trg_sos_idx=trg_sos_idx,         # 디코더 입력 시작 <sos> 토큰 인덱스
    d_model=d_model,                 # embedding + hidden dimension 크기
    enc_voc_size=enc_voc_size,       # source 단어 수
    dec_voc_size=dec_voc_size,       # target 단어 수
    max_len=max_len,                 # positional encoding 최대 길이
    ffn_hidden=ffn_hidden,           # FeedForward hidden layer 크기
    n_head=n_heads,                  # multi-head attention 헤드 수
    n_layers=n_layers,               # encoder/decoder 층 수
    drop_prob=drop_prob,             # dropout 비율
    device=device                    # 연산 장치 (cuda or cpu)
).to(device)                         # 모델을 device에 올림

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)      # weight 초기화 적용

# Adam optimizer 설정
optimizer = Adam(
    params=model.parameters(),
    lr=init_lr,
    weight_decay=weight_decay,  # L2 정규화
    eps=adam_eps                # 수치 안정성 개선
)

# 성능 향상 없을 경우 learning rate 감소 (ReduceLROnPlateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    verbose=True,
    factor=factor,              # 감소 비율
    patience=patience           # 몇 epoch 기다릴지
)

# 손실 함수: pad 토큰 위치는 무시
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)



# 학습 함수 (1 epoch)
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src              # source 문장
        trg = batch.trg              # target 문장

        optimizer.zero_grad()        # gradient 초기화

        # 디코더 입력: <sos>부터 마지막-1까지
        output = model(src, trg[:, :-1])  # shape: [batch, trg_len - 1, vocab]

        # CrossEntropyLoss 계산 위해 reshape
        output_reshape = output.contiguous().view(-1, output.shape[-1])  # [batch*trg_len, vocab]
        trg = trg[:, 1:].contiguous().view(-1)  # 정답 시퀀스에 <sos> 포함 X, 그 이후 토큰들만 손실 계산에 사용

        loss = criterion(output_reshape, trg)  # 손실 계산
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # gradient clipping
        optimizer.step()  # 파라미터 업데이트

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)  # 평균 손실 반환



# 평가 함수 (loss + BLEU 점수 포함)
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            # 디코더 입력: <sos>부터 마지막-1까지
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            # BLEU 계산
            total_bleu = []
            for j in range(batch_size):
                try:
                    # 정답 문장
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)

                    # 예측 문장 (가장 확률 높은 토큰 선택)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)

                    # BLEU 점수 계산
                    bleu = get_bleu(
                        hypotheses=output_words.split(),
                        reference=trg_words.split()
                    )
                    total_bleu.append(bleu)
                except:
                    pass

            # 해당 배치의 평균 BLEU
            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu



# 전체 학습 루프 (여러 epoch 동안)
def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []

    for step in range(total_epoch):
        start_time = time.time()

        # 학습 및 검증
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # warmup 이후 scheduler 적용
        if step > warmup:
            scheduler.step(valid_loss)

        # 로그 기록
        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)

        # 모델 저장 (성능 향상 시)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f'saved/model-{valid_loss}.pt')

        # 결과 파일 저장
        with open('result/train_loss.txt', 'w') as f: f.write(str(train_losses))
        with open('result/test_loss.txt', 'w') as f: f.write(str(test_losses))
        with open('result/bleu.txt', 'w') as f: f.write(str(bleus))

        # 로그 출력
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')



if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)

