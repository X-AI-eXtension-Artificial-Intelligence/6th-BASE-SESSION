"""
@author : Hyunwoong                      # 코드 작성자 정보
@when : 2019-10-22                      # 코드 작성 날짜
@homepage : https://github.com/gusdnd852  # 작성자 깃허브
"""
import math                               # 수학 연산을 위한 모듈
import time                               # 시간 측정용 모듈

from torch import nn, optim               # PyTorch의 신경망과 최적화 모듈 import
from torch.optim import Adam              # Adam Optimizer 직접 import

# 전처리 및 설정 파일 import
from data import *                        # 데이터 관련 설정 및 로더
from models.model.transformer import Transformer  # Transformer 모델 정의
from util.bleu import idx_to_word, get_bleu       # BLEU 점수 계산 관련 함수
from util.epoch_timer import epoch_time           # 에폭 시간 측정 함수

def count_parameters(model):                       # 모델의 학습 가능한 파라미터 수 계산
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):                         # 가중치 초기화 함수 정의
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)    # Kaiming uniform 초기화 적용

# 모델 초기화
model = Transformer(
    src_pad_idx=src_pad_idx,                       # 소스 패딩 인덱스
    trg_pad_idx=trg_pad_idx,                       # 타겟 패딩 인덱스
    trg_sos_idx=trg_sos_idx,                       # 타겟 시작 토큰 인덱스
    d_model=d_model,                               # 모델 차원 수
    enc_voc_size=enc_voc_size,                     # 인코더 어휘 크기
    dec_voc_size=dec_voc_size,                     # 디코더 어휘 크기
    max_len=max_len,                               # 최대 문장 길이
    ffn_hidden=ffn_hidden,                         # FFN의 은닉층 차원 수
    n_head=n_heads,                                # 멀티헤드 수
    n_layers=n_layers,                             # 인코더/디코더 레이어 수
    drop_prob=drop_prob,                           # 드롭아웃 확률
    device=device                                   # 연산 디바이스 (cpu/gpu)
).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')  # 파라미터 수 출력
model.apply(initialize_weights)                     # 모델의 모든 레이어에 weight 초기화 적용

# 옵티마이저 정의
optimizer = Adam(
    params=model.parameters(),                     # 최적화 대상 파라미터
    lr=init_lr,                                     # 초기 학습률
    weight_decay=weight_decay,                     # 가중치 감소 (L2 정규화)
    eps=adam_eps                                    # 작은 값으로 0으로 나누는 것 방지
)

# ReduceLROnPlateau 스케줄러 정의
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    verbose=True,
    factor=factor,                                 # 성능 향상 없을 시 학습률 감소 비율
    patience=patience                              # 몇 에폭 동안 개선이 없을 때 감소
)

# 손실 함수 정의 (패딩 인덱스 무시)
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

def train(model, iterator, optimizer, criterion, clip):
    model.train()                                   # 학습 모드 전환
    epoch_loss = 0

    for i, batch in enumerate(iterator):            # 배치 순회
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()                       # 기울기 초기화
        output = model(src, trg[:, :-1])            # 디코더 입력은 <sos> ~ 마지막 전 토큰
        output_reshape = output.contiguous().view(-1, output.shape[-1])  # [B*T, V] 형태로 reshape
        trg = trg[:, 1:].contiguous().view(-1)      # 정답도 <sos> 이후로 reshape

        loss = criterion(output_reshape, trg)       # 손실 계산
        loss.backward()                             # 역전파

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # gradient clipping
        optimizer.step()                            # 파라미터 업데이트

        epoch_loss += loss.item()                   # 손실 누적
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())  # 진행률 출력

    return epoch_loss / len(iterator)               # 평균 손실 반환

def evaluate(model, iterator, criterion):
    model.eval()                                    # 평가 모드 전환
    epoch_loss = 0
    batch_bleu = []

    with torch.no_grad():                           # 기울기 추적 비활성화
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            # BLEU 계산
            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)  # 정답 문장
                    output_words = output[j].max(dim=1)[1]                      # 예측 결과 인덱스 추출
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu) if total_bleu else 0
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu) if batch_bleu else 0
    return epoch_loss / len(iterator), batch_bleu    # 평균 손실, BLEU 반환

def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []

    for step in range(total_epoch):
        start_time = time.time()                    # 시간 측정 시작

        # 학습 및 검증 수행
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)

        end_time = time.time()                      # 시간 측정 종료

        if step > warmup:                           # warm-up 에폭 이후 스케줄러 작동
            scheduler.step(valid_loss)

        # 로그 저장
        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # best loss 갱신 시 모델 저장
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f'saved/model-{valid_loss:.4f}.pt')

        # 결과 로그 파일로 저장
        with open('result/train_loss.txt', 'w') as f:
            f.write(str(train_losses))
        with open('result/bleu.txt', 'w') as f:
            f.write(str(bleus))
        with open('result/test_loss.txt', 'w') as f:
            f.write(str(test_losses))

        # 로그 출력
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)           # 메인 함수 실행 (훈련 시작)
