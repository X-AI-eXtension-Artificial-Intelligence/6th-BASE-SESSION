import torch
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from models.build_model import build_model
import pickle

# ----- 설정 -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
CHECKPOINT_PATH = "./checkpoints/final_model.pt"
VOCAB_SRC_PATH = "./checkpoints/vocab_src.pkl"
VOCAB_TGT_PATH = "./checkpoints/vocab_tgt.pkl"
SOS_IDX, EOS_IDX, PAD_IDX = 2, 3, 1

# ----- 토크나이저 -----
tokenizer_input = get_tokenizer("basic_english")
tokenizer_target = get_tokenizer("basic_english")  # 출력 텍스트는 토크나이즈 후 디코딩용

# ----- 사전 로드 -----
def load_vocab(path):
    with open(path, "rb") as f:
        return pickle.load(f)

vocab_src = load_vocab(VOCAB_SRC_PATH)
vocab_tgt = load_vocab(VOCAB_TGT_PATH)

# ----- 문장 수치화 -----
def numericalize(tokens, vocab):
    return [SOS_IDX] + [vocab[token] for token in tokens] + [EOS_IDX]

# ----- 요약 함수 (Greedy Decode) -----
def summarize_text(text, model):
    model.eval()
    tokens = tokenizer_input(text)
    src_tensor = torch.tensor(numericalize(tokens, vocab_src), dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    ys = torch.tensor([[SOS_IDX]], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        for i in range(MAX_LEN):
            out, _ = model(src_tensor, ys)
            next_token = out[:, -1, :].argmax(dim=-1).unsqueeze(1)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == EOS_IDX:
                break
    output_tokens = ys.squeeze(0).tolist()[1:]  # remove SOS
    output_text = vocab_tgt.lookup_tokens(output_tokens)
    if "<eos>" in output_text:
        output_text = output_text[:output_text.index("<eos>")]
    return " ".join(output_text)

# ----- 모델 로딩 -----
model = build_model(len(vocab_src), len(vocab_tgt), device=DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# ----- 실행 -----
if __name__ == "__main__":
    while True:
        text = input("📰 Enter news article (or 'exit'): ")
        if text.strip().lower() == "exit":
            break
        result = summarize_text(text, model)
        print("🟢 Summary:", result)
        