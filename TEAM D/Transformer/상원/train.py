"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time

from torch import nn, optim     # 신경망 계층(nn)과 최적화 함수(optim) 불러오기
from torch.optim import Adam    # Adam 옵티마이저 불러오기

from data import *
from models.model.transformer import Transformer    # Transformer 모델 클래스 불러오기기
from util.bleu import idx_to_word, get_bleu         # idx_to_word: 인덱스를 실제 단어로 변환하는 함수 , get_bleu: 모델의 번역 성능 평가를 위한 BLEU 점수 계산 함수
from util.epoch_timer import epoch_time         

# 모델의 학습 가능한(gradient를 계산하는) 전체 파라미터 수를 반환
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)    # p.numel(): 각 파라미터 텐서의 요소 수 , requires_grad: 학습 가능한 파라미터만 선택

# 주어진 모듈 m에 대해 He(kaiming) 초기화를 적용
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:     # weight의 차원이 2 이상이면 초기화
        nn.init.kaiming_uniform(m.weight.data)


model = Transformer(src_pad_idx=src_pad_idx,        # 소스 문장에서의 패딩 토큰 인덱스
                    trg_pad_idx=trg_pad_idx,        # 타겟 문장에서의 패딩 토큰 인덱스
                    trg_sos_idx=trg_sos_idx,        # 타겟 문장에서의 시작 토큰 인덱스
                    d_model=d_model,                # 임베딩 차원 (예: 512) → 전체 모델의 feature 크기
                    enc_voc_size=enc_voc_size,      # 인코더의 전체 단어 집합 크기
                    dec_voc_size=dec_voc_size,      # 디코더의 전체 단어 집합 크기
                    max_len=max_len,                # 최대 시퀀스 길이
                    ffn_hidden=ffn_hidden,          # 피드포워드 신경망(hidden layer)의 크기
                    n_head=n_heads,                 # 멀티헤드 어텐션의 head 수
                    n_layers=n_layers,              # 인코더/디코더 블록의 레이어 수
                    drop_prob=drop_prob,            # 드롭아웃 
                    device=device).to(device)       # 모델을 지정한 디바이스로 이동

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)                                         # 모델 가중치 초기화
# 옵티마이저 설정 (Adam)
optimizer = Adam(params=model.parameters(),         # 학습할 파라미터 (모델의 모든 가중치)
                 lr=init_lr,                        # 초기 학습률 
                 weight_decay=weight_decay,         # 가중치 감쇠(L2 정규화) 계수
                 eps=adam_eps)                      # 작은 값으로 분모가 0 되는 걸 방지
# 스케줄러 설정 (ReduceLROnPlateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,          # 학습률이 줄어들 때마다 로그 출력
                                                 factor=factor,         # 감소 계수
                                                 patience=patience)
# 손실 함수 정의 (CrossEntropyLoss)
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)               # 패딩 토큰 인덱스는 손실 계산에서 무시

# 학습
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src     # 소스 시퀀스 (입력)
        trg = batch.trg     # 타겟 시퀀스 (정답)

        
        optimizer.zero_grad()                   # 이전 스텝의 gradient 초기화
        output = model(src, trg[:, :-1])        # 모델 forward
        output_reshape = output.contiguous().view(-1, output.shape[-1]) # 크로스엔트로피는 2D 입력 필요 → (batch_size * trg_len-1, output_dim)로 reshape
        trg = trg[:, 1:].contiguous().view(-1)                          # 타겟도 (batch_size * trg_len-1)로 평탄화

        # 손실 계산
        loss = criterion(output_reshape, trg)
        # 역전파
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)        # gradient clipping: gradient 폭주 방지
        # 파라미터 업데이트
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())
    # 전체 에폭 동안의 평균 손실 반환
    return epoch_loss / len(iterator)

# 평가
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():           # 평가 중에는 gradient를 계산하지 않음
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()
            
            # BLEU 점수 계산
            total_bleu = []
            for j in range(batch_size):
                # 타겟 문장을 인덱스 → 단어로 변환
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    # 출력: 가장 높은 확률의 단어 인덱스 추출
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    # BLEU 점수 계산
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass
            # 배치 BLEU 점수 평균
            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)
    # 전체 배치의 평균 BLEU 점수
    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu

# 학습 및 평가 루프를 실행
def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []       # 손실 및 BLEU 기록용 리스트
    for step in range(total_epoch):
        # 학습 루프 실행
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        # 검증 루프 실행
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        #  모델 저장 (최고 성능 모델)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
