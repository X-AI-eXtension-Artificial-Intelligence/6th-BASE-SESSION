from pathlib import Path

# 파라미터 설정
def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 75,
        "lr": 10**-4,
        "seq_len": 300, # 전체 100만개 Pair 중 300 시퀀스 길이 넘는 것 1000개도 안되기 때문에 300으로 지정
        "d_model": 512, # 임베딩 차원 지정
        "datasource": "Helsinki-NLP/opus-100", # 영어->한국어 번역 데이터셋
        "lang_src": "en", # 소스 언어 -> 영어
        "lang_tgt": "ko", # 타겟 언어 -> 한국어
        "model_folder": "weights", # 가중치 폴더
        "model_basename": "tmodel_", 
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json", # 토크나이저 파일
        "experiment_name": "runs/tmodel" # tensorboard 로그 폴더 지정
    }

# 가중치 파일 경로 가져오기
def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# 가중치 폴더에서 가장 최근 가중치 파일 불러오기
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
