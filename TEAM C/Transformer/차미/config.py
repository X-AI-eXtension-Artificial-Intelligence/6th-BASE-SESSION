from pathlib import Path

# 학습 설정 값을 반환
def get_config():
    return {
        "batch_size": 8, # 배치 크기 설정
        "num_epochs": 20, # 학습 에폭 수
        "lr": 1e-4, # 학습률 설정
        "seq_len": 350, # 시퀀스 길이
        "d_model": 512, # 모델 차원 수
        "datasource": "opus100", 
        "lang_src": "en", # 소스 언어
        "lang_tgt": "ko", # 타겟 언어
        "model_folder": "weights", # 모델 가중치 저장 폴더
        "model_basename": "tmodel_", # 모델 파일 이름 접두사
        "preload": "latest", # 사전 로드할 체크포인트 이름
        "tokenizer_file": "tokenizer_{0}.json", # 토크나이저 파일 이름 형식
        "experiment_name": "runs/tmodel" # 텐서보드 로그 저장 경로
    }

# 특정 에폭에 해당하는 가중치 파일 경로 반환
def get_weights_file_path(config, epoch: str):
    folder = f"{config['datasource']}_{config['model_folder']}"
    filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / folder / filename)

# 가장 최근의 가중치 파일 경로 반환
def latest_weights_file_path(config):
    folder = f"{config['datasource']}_{config['model_folder']}"
    pattern = f"{config['model_basename']}*"
    files = list(Path(folder).glob(pattern))
    if not files:
        return None
    files.sort()
    return str(files[-1])
