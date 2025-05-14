# train.py : 전체 학습 파이프라인 담당

import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time


def count_parameters(model): # 학습 가능한 파라미터의 총 개수 계산
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m): # 레이어에 weight 손성이 존재하고, 2차원 이상이면 Kaiming 균등 분포로 가중치 초기화
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

# 설정값에 따라 Transformer 모델 객체를 생성하고 지정한 디바이스로 이동
model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters') # 학습 가능한 파라미터 개수 출력
model.apply(initialize_weights) # 모든 레이어에 initialize_weights 함수 적용해 가중치 초기화
optimizer = Adam(params=model.parameters(), # 모델 파라미터 대상으로 Adam optimizer 생성
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, # 검증 손실이 개선되지 않으면 학습률을 줄이는 스케줄러
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx) # 패딩 토큰 무시하고, 교차 엔트로피 오차를 손실 함수로 사용

# 학습 모드로 모델 설정 -> 각 배치마다 순전파, 손실 계산, 역전파, 그래디언트 클리핑, 최적화, 손실 누적, 진행률 출력 등 수행 -> 평균 손실 반환
def train(model, iterator, optimizer, criterion, clip):
    model.train() # 학습 모드 활성화
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg 

        optimizer.zero_grad() # 이전 그래디언트 초기화
        output = model(src, trg[:, :-1]) # 디코더 입력으로 [SOS] 토큰 전달
        output_reshape = output.contiguous().view(-1, output.shape[-1]) # 출력 형태 재구성 : (배치 * 시퀀스 길이, 어휘 크기)
        trg = trg[:, 1:].contiguous().view(-1) # [EOS] 토큰 제거 및 형태 재구성

        loss = criterion(output_reshape, trg) # 손실 계산
        loss.backward() # 역전파
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # 그래디언트 클리핑
        optimizer.step() # 매개변수 업데이트

        epoch_loss += loss.item() # 에폭 손실 누적
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator) # 평균 손실 반환

# 평가 모드로 모델 설정 -> 각 배치마다 손실 BLEU 점수 계산 (그래디언트 계산 X) -> 평균 손실과 BLEU 반환
def evaluate(model, iterator, criterion):
    model.eval() # 
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad(): # 그래디언트 계산 비활성화[
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = [] # BLEU 점수 계산
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1] # 최대 확률 인덱스
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except: # 변환 실패 시 예외 처리
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu

# 전체 학습 루프 관리 : 각 에폭마다 학습, 평가, 스케줄러 업데이트, 손실 및 BLEU 저장, 모델 저장, 학습 정보 출력 등
def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()

        # 학습 및 평가
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        # 학습률 스케줄링
        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # 모델 상태 저장
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        # 결과 기록 (JSON 형식 저장)
        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        # 진행 상황 출력
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
