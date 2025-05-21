from transformer import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import warnings
import os
from tqdm import tqdm

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

# greedy decoding을 수행하는 함수
# 디코더가 한 번에 하나씩 다음 토큰을 예측
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]') # 시작 토큰 ID를 의미
    eos_idx = tokenizer_tgt.token_to_id('[EOS]') # 종료 토큰 ID를 의미

    encoder_output = model.encode(source, source_mask) # 인코더 출력 계산
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device) # 시작 토큰으로 디코더 입력 초기화

    while True:
        if decoder_input.size(1) == max_len:
            break # 최대 길이 도달 시 종료

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device) # 디코더 마스크 생성
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask) # 디코딩 수행
        prob = model.project(out[:, -1]) # 마지막 토큰 위치에 대한 확률 분포 계산
        _, next_word = torch.max(prob, dim=1) # 확률이 가장 높은 토큰 선택

        # 디코더 입력에 새 토큰 추가
        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
        ], dim=1)

        if next_word == eos_idx:
            break # 종료 토큰 예측 시 중단

    return decoder_input.squeeze(0)

# 모델 검증을 수행하는 함수
def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    source_texts, expected, predicted = [], [], []

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "validation 시 배치 크기는 1이어야 함"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg('-' * console_width)
            print_msg(f"{'SOURCE:':>12} {source_text}")
            print_msg(f"{'TARGET:':>12} {target_text}")
            print_msg(f"{'PREDICTED:':>12} {model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break

    if writer:
        cer = torchmetrics.CharErrorRate()(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)

        wer = torchmetrics.WordErrorRate()(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)

        bleu = torchmetrics.BLEUScore()(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)

        writer.flush()

# 데이터셋에서 문장들을 순차적으로 반환
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

# 토크나이저를 새로 만들거나 기존 파일을 로드
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# 데이터셋 및 토크나이저를 준비하고 DataLoader 반환
def get_ds(config):
    ds_raw = load_dataset(config['datasource'], f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_size, val_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # 최대 문장 길이 확인
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_len = len(tokenizer_src.encode(item['translation'][config['lang_src']]).ids)
        tgt_len = len(tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids)
        max_len_src = max(max_len_src, src_len)
        max_len_tgt = max(max_len_tgt, tgt_len)

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_loader, val_loader, tokenizer_src, tokenizer_tgt

# Transformer 모델 객체 생성
def get_model(config, vocab_src_len, vocab_tgt_len):
    return build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], d_model=config['d_model'])

# 전체 학습 루프 정의
def train_model(config):
    # 사용할 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.has_mps or torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    # 가중치 저장 폴더 생성
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # 데이터 및 모델 준비
    train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # 이전 모델 로드 여부 확인
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_path = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None

    if model_path:
        print(f"Preloading model {model_path}")
        state = torch.load(model_path)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    else:
        print("No model to preload, starting from scratch")

    # 학습 루프 실행
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        run_validation(model, val_loader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        model_file = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_file)

# 메인 실행 진입점
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
