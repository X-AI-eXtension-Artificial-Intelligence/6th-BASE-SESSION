import torch
import torch.nn as nn
import math
from pathlib import Path

# 모델, 데이터셋, 설정 관련 모듈 임포트
from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

# 외부 라이브러리 임포트
import torchtext.datasets as datasets  # 사용하지 않으면 제거 가능
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import warnings
from tqdm import tqdm  # 진행률 표시
import os

# Hugging Face datasets 및 토크나이저
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics  # 평가 지표 (CER, WER, BLEU 등)
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 로깅

# -------------------------- 디코딩 함수 --------------------------
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    그리디 디코딩 구현
    :param model: Transformer 모델
    :param source: 인코더 입력 텐서 (1, seq_len)
    :param source_mask: 인코더 마스크
    :param tokenizer_src: 소스 토크나이저
    :param tokenizer_tgt: 타겟 토크나이저
    :param max_len: 최대 시퀀스 길이
    :param device: 'cpu' 또는 'cuda'
    :return: 디코딩된 토큰 ID 시퀀스 (seq_len)
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')  # 시작 토큰 ID
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')  # 종료 토큰 ID

    # 인코더 출력 계산 (재사용)
    encoder_output = model.encode(source, source_mask)
    # 디코더 입력에 [SOS] 토큰 삽입
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        # 최대 길이 도달 시 중단
        if decoder_input.size(1) == max_len:
            break

        # 인과적 마스크 생성
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # 디코더 단계 실행
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        # 마지막 시점 단어 분포 예측
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        # 예측 토큰 ID를 디코더 입력에 추가
        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
        ], dim=1)

        # [EOS] 예측 시 디코딩 종료
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

# -------------------------- 검증 함수 --------------------------
def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    """
    검증 데이터셋에서 예시 문장들을 디코딩하고, CER/WER/BLEU 계산 후 TensorBoard에 기록
    :param model: Transformer 모델
    :param validation_ds: 검증용 DataLoader
    :param tokenizer_src: 소스 토크나이저
    :param tokenizer_tgt: 타겟 토크나이저
    :param max_len: 최대 시퀀스 길이
    :param device: 'cpu' 또는 'cuda'
    :param print_msg: tqdm 출력 함수
    :param global_step: 현재 학습 스텝
    :param writer: TensorBoard SummaryWriter
    :param num_examples: 출력할 예시 개수
    """
    model.eval()
    count = 0
    source_texts, expected, predicted = [], [], []

    # 콘솔 너비 얻기 (출력 레이아웃용)
    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            src_text = batch['src_text'][0]
            tgt_text = batch['tgt_text'][0]
            pred_text = tokenizer_tgt.decode(model_out.cpu().numpy())

            # 예시 출력
            print_msg('-' * console_width)
            print_msg(f"{'SOURCE:':>12} {src_text}")
            print_msg(f"{'TARGET:':>12} {tgt_text}")
            print_msg(f"{'PREDICTED:':>12} {pred_text}")

            source_texts.append(src_text)
            expected.append(tgt_text)
            predicted.append(pred_text)

            if count == num_examples:
                print_msg('-' * console_width)
                break

    if writer:
        # CER 계산 및 기록
        cer = torchmetrics.CharErrorRate()(predicted, expected)
        writer.add_scalar('validation/cer', cer, global_step)
        # WER 계산 및 기록
        wer = torchmetrics.WordErrorRate()(predicted, expected)
        writer.add_scalar('validation/wer', wer, global_step)
        # BLEU 계산 및 기록
        bleu = torchmetrics.BLEUScore()(predicted, expected)
        writer.add_scalar('validation/bleu', bleu, global_step)
        writer.flush()

# -------------------------- 토크나이저 생성/로드 --------------------------
def get_all_sentences(ds, lang):
    # 데이터셋의 모든 문장 스트림으로 반환
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    """
    토크나이저 파일이 없으면 학습 후 저장, 있으면 로드
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# -------------------------- 데이터로더 생성 --------------------------
def get_ds(config):
    """
    데이터셋 로드 & 토크나이저 빌드 & 학습/검증 분할 & DataLoader 반환
    """
    # Hugging Face 데이터셋 로드 (train split만 제공됨)
    ds_raw = load_dataset(config['datasource'], f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # 토크나이저
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # 90% 학습, 10% 검증 분할
    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size
    train_raw, val_raw = random_split(ds_raw, [train_size, val_size])

    train_ds = BilingualDataset(train_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds   = BilingualDataset(val_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=True)

    # 문장 길이 통계 출력 (디버깅용)
    max_src = max(len(tokenizer_src.encode(item['translation'][config['lang_src']]).ids) for item in ds_raw)
    max_tgt = max(len(tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids) for item in ds_raw)
    print(f"Max length - source: {max_src}, target: {max_tgt}")

    return train_loader, val_loader, tokenizer_src, tokenizer_tgt

# -------------------------- 모델 및 학습 루프 --------------------------
def get_model(config, vocab_src_len, vocab_tgt_len):
    # Transformer 모델 생성
    return build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], d_model=config['d_model'])


def train_model(config):
    """
    전체 학습 파이프라인: 디바이스 설정, 데이터로더/모델 준비, 옵티마이저, 체크포인트 로드, 학습/검증, 모델 저장
    """
    # 디바이스 선택 (cuda > mps > cpu)
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.has_mps else 'cpu')
    print("Using device:", device)
    device = torch.device(device)

    # 가중치 저장 폴더 생성
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # 데이터로더 및 토크나이저 준비
    train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_ds(config)
    # 모델 초기화 및 장치 할당
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # TensorBoard SummaryWriter
    writer = SummaryWriter(config['experiment_name'])

    # 옵티마이저 및 손실 함수
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # 체크포인트 로드 (latest 혹은 지정 epoch)
    initial_epoch, global_step = 0, 0
    if config['preload']:
        ckpt = latest_weights_file_path(config) if config['preload']=='latest' else get_weights_file_path(config, config['preload'])
        if ckpt and Path(ckpt).exists():
            print(f"Preloading checkpoint {ckpt}")
            state = torch.load(ckpt)
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            initial_epoch = state['epoch'] + 1
            global_step = state['global_step']

    # 에폭 반복
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        tqdm_iter = tqdm(train_loader, desc=f"Epoch {epoch:02d}")

        for batch in tqdm_iter:
            # 배치 텐서 준비
            enc_in = batch['encoder_input'].to(device)
            dec_in = batch['decoder_input'].to(device)
            enc_mask = batch['encoder_mask'].to(device)
            dec_mask = batch['decoder_mask'].to(device)
            label =    batch['label'].to(device)

            # 순전파
            enc_out = model.encode(enc_in, enc_mask)
            dec_out = model.decode(enc_out, enc_mask, dec_in, dec_mask)
            logits  = model.project(dec_out)

            # 손실 계산
            loss = loss_fn(logits.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            tqdm_iter.set_postfix({'loss': f"{loss.item():.3f}"})
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.flush()

            # 역전파 및 옵티마이저 스텝
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        # 검증 실행 및 지표 기록
        run_validation(model, val_loader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                       lambda msg: tqdm_iter.write(msg), global_step, writer)

        # 체크포인트 저장
        ckpt_path = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'global_step': global_step}, ckpt_path)

# -------------------------- 스크립트 실행 --------------------------
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()  # 설정 불러오기
    train_model(config)  # 학습 시작
