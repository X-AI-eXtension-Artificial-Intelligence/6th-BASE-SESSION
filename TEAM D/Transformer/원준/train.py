
import math
import time

from torch import nn, optim
from torch.optim import Adam

# 데이터셋 및 사전 처리된 구성 요소 (vocab, pad index 등)
from data import *
# Transformer 모델 정의
from models.model.transformer import Transformer
# BLEU 점수 계산 함수 및 예측 결과 변환 함수
from util.bleu import idx_to_word, get_bleu
# 에폭별 시간 측정 함수
from util.epoch_timer import epoch_time

#  학습 가능한 파라미터 수 계산 함수
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Kaiming 초기화 적용 함수
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)

# Transformer 모델 초기화
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

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)

# 옵티마이저 및 스케줄러 정의
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

#  손실 함수 정의 (PAD 토큰은 무시)
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

# 학습 함수 정의
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src              # 인코더 입력
        trg = batch.trg              # 디코더 입력 및 정답

        optimizer.zero_grad()  #  누적 방지하기 위해서 
        output = model(src, trg[:, :-1])  # 디코더 입력은 <sos> ~ 마지막 이전

        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)  # 정답은 <sos> 이후 ~ <eos>   루프 돌면서 계속 바뀐다 
        #  3D 텐서인 output을 2D 텐서로 reshape해서 loss 계산에 적합하게 변환
        # .contiguous()는 메모리 재배열을 해주는 함수로, .view()를 안정적으로 사용할 수 있게 보장

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 그래디언트 클리핑
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)


# 검증 함수 정의 (BLEU 점수 포함)
def evaluate(model, iterator, criterion):
    model.eval()  # 모델을 평가 모드로 전환 (Dropout 등 비활성화)
    epoch_loss = 0  # 에폭 손실 초기화
    batch_bleu = []  # BLEU 점수 누적 리스트

    with torch.no_grad():  # 평가 중에는 gradient 계산 비활성화
        for i, batch in enumerate(iterator):
            src = batch.src  # 인코더 입력 시퀀스
            trg = batch.trg  # 디코더 입력 및 정답 시퀀스

            # 디코더 입력은 <sos>부터 마지막 이전까지
            output = model(src, trg[:, :-1])
            # 모델 출력 텐서를 [배치 × 시퀀스 길이, 어휘 수] 형태로 reshape
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            # 정답도 [배치 × 시퀀스 길이] 형태로 reshape (정답은 <sos> 이후)
            trg = trg[:, 1:].contiguous().view(-1)

            # CrossEntropyLoss 계산
            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            # BLEU 점수 계산
            total_bleu = []
            for j in range(batch_size):  # 배치 내 각 문장별로 BLEU 측정
                try:
                    # 정답 시퀀스를 단어로 변환
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    # 예측 확률에서 가장 높은 인덱스를 선택 (argmax)
                    output_words = output[j].max(dim=1)[1]
                    # 예측 시퀀스도 단어로 변환
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    # BLEU 점수 계산
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass  # 예외 발생 시 해당 샘플 BLEU는 건너뜀

            # 배치 내 평균 BLEU 점수 계산
            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    # 전체 배치 평균 BLEU, 평균 손실 반환
    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


# 전체 학습 루프 정의
def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []  # 결과 저장 리스트

    for step in range(total_epoch):
        start_time = time.time()  # 시간 측정 시작

        # 1 에폭 학습
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        # 검증 수행
        valid_loss, bleu = evaluate(model, valid_iter, criterion)

        end_time = time.time()  # 시간 측정 종료

        # warmup 이후부터 learning rate scheduler 적용
        if step > warmup:
            scheduler.step(valid_loss)

        # 결과 저장
        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)

        # 에폭 시간 계산
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # 성능이 개선된 경우 모델 저장
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f'saved/model-{valid_loss:.4f}.pt')

        # 손실 및 BLEU 점수 결과 파일로 저장
        with open('result/train_loss.txt', 'w') as f:
            f.write(str(train_losses))
        with open('result/bleu.txt', 'w') as f:
            f.write(str(bleus))
        with open('result/test_loss.txt', 'w') as f:
            f.write(str(test_losses))

        # 콘솔 출력 (PPL = Perplexity = exp(loss))
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


# 진입점: 직접 실행 시 학습 시작
if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
