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

def count_parameters(model):  # 함수:파라미터 개수
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # 반환:학습가능 파라미터 수


def initialize_weights(m):  # 함수:가중치 초기화
    if hasattr(m, 'weight') and m.weight.dim() > 1:  # 조건:weight 속성 및 차원
        nn.init.kaiming_uniform(m.weight.data)  # 초기화:Kaiming 균등


model = Transformer(src_pad_idx=src_pad_idx,  # 인스턴스:Transformer
                    trg_pad_idx=trg_pad_idx,  # 파라미터:trg_pad_idx
                    trg_sos_idx=trg_sos_idx,  # 파라미터:trg_sos_idx
                    d_model=d_model,  # 차원:d_model
                    enc_voc_size=enc_voc_size,  # 어휘크기:인코더
                    dec_voc_size=dec_voc_size,  # 어휘크기:디코더
                    max_len=max_len,  # 최대길이:max_len
                    ffn_hidden=ffn_hidden,  # FFN은닉:ffn_hidden
                    n_head=n_heads,  # 헤드수:n_heads
                    n_layers=n_layers,  # 레이어수:n_layers
                    drop_prob=drop_prob,  # 드롭아웃:drop_prob
                    device=device).to(device)  # 디바이스:device

print(f'The model has {count_parameters(model):,} trainable parameters')  # 출력:파라미터 수
model.apply(initialize_weights)  # 적용:가중치 초기화

optimizer = Adam(params=model.parameters(),  # 옵티마이저:Adam
                 lr=init_lr,  # 학습률:init_lr
                 weight_decay=weight_decay,  # 가중치감쇠:weight_decay
                 eps=adam_eps)  # epsilon:adam_eps

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,  # 스케줄러:ReduceLROnPlateau
                                                 verbose=True,  # 출력:True
                                                 factor=factor,  # 감소비율:factor
                                                 patience=patience)  # 대기:patience

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)  # 손실:CrossEntropy(ignore<pad>)


def train(model, iterator, optimizer, criterion, clip):  # 함수:train
    model.train()  # 모드:train
    epoch_loss = 0  # 초기화:epoch_loss
    for i, batch in enumerate(iterator):  # 반복:배치
        src = batch.src  # 데이터:src
        trg = batch.trg  # 데이터:trg

        optimizer.zero_grad()  # 초기화:grad
        output = model(src, trg[:, :-1])  # 순전파
        output_reshape = output.contiguous().view(-1, output.shape[-1])  # 변환:output
        trg = trg[:, 1:].contiguous().view(-1)  # 변환:trg

        loss = criterion(output_reshape, trg)  # 손실 계산
        loss.backward()  # 역전파
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 클리핑:clip
        optimizer.step()  # 업데이트

        epoch_loss += loss.item()  # 누적:loss
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())  # 출력:진행률

    return epoch_loss / len(iterator)  # 반환:평균 loss


def evaluate(model, iterator, criterion):  # 함수:evaluate
    model.eval()  # 모드:eval
    epoch_loss = 0  # 초기화:epoch_loss
    batch_bleu = []  # 초기화:batch_bleu
    with torch.no_grad():  # no_grad
        for i, batch in enumerate(iterator):  # 반복:배치
            src = batch.src  # 데이터:src
            trg = batch.trg  # 데이터:trg
            output = model(src, trg[:, :-1])  # 순전파
            output_reshape = output.contiguous().view(-1, output.shape[-1])  # 변환:output
            trg = trg[:, 1:].contiguous().view(-1)  # 변환:trg

            loss = criterion(output_reshape, trg)  # 손실 계산
            epoch_loss += loss.item()  # 누적:loss

            total_bleu = []  # 초기화:total_bleu
            for j in range(batch_size):  # 반복:batch_size
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)  # 변환:ref 문장
                    output_words = output[j].max(dim=1)[1]  # 예측 인덱스
                    output_words = idx_to_word(output_words, loader.target.vocab)  # 변환:hyp 문장
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())  # BLEU 계산
                    total_bleu.append(bleu)  # 추가:bleu
                except:
                    pass  # 예외 무시

            total_bleu = sum(total_bleu) / len(total_bleu)  # 평균:batch BLEU
            batch_bleu.append(total_bleu)  # 추가:batch_bleu

    batch_bleu = sum(batch_bleu) / len(batch_bleu)  # 평균:전체 BLEU
    return epoch_loss / len(iterator), batch_bleu  # 반환:loss, BLEU


def run(total_epoch, best_loss):  # 함수:run
    train_losses, test_losses, bleus = [], [], []  # 초기화:리스트
    for step in range(total_epoch):  # 반복:에포크
        start_time = time.time()  # 측정:시작 시간
        train_loss = train(model, train_iter, optimizer, criterion, clip)  # 학습 호출
        valid_loss, bleu = evaluate(model, valid_iter, criterion)  # 평가 호출
        end_time = time.time()  # 측정:종료 시간

        if step > warmup:  # 조건:워밍업 이후
            scheduler.step(valid_loss)  # 스케줄:업데이트

        train_losses.append(train_loss)  # 추가:train_losses
        test_losses.append(valid_loss)  # 추가:test_losses
        bleus.append(bleu)  # 추가:bleus
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)  # 시간계산

        if valid_loss < best_loss:  # 조건:최저 loss
            best_loss = valid_loss  # 갱신:best_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))  # 저장:모델

        f = open('result/train_loss.txt', 'w')  # 파일열기:train_loss
        f.write(str(train_losses))  # 기록:train_losses
        f.close()  # 파일닫기

        f = open('result/bleu.txt', 'w')  # 파일열기:bleu
        f.write(str(bleus))  # 기록:bleus
        f.close()  # 파일닫기

        f = open('result/test_loss.txt', 'w')  # 파일열기:test_loss
        f.write(str(test_losses))  # 기록:test_losses
        f.close()  # 파일닫기

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')  # 출력:에포크, 시간
        print(f'	Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')  # 출력:Train PPL
        print(f'	Val Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')  # 출력:Val PPLL
        print(f'	BLEU Score: {bleu:.3f}')  # 출력:BLEU

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)  # 실행:main