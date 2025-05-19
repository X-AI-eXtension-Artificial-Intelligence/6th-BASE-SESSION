import torch
import torch.nn as nn
from transformer import Transformer
from data_loader import load_dummy_data
import time

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data, target)
        
        # Reshape output for loss calculation
        output = output.view(-1, output.size(-1))
        target = target.view(-1)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, target)
            
            output = output.view(-1, output.size(-1))
            target = target.view(-1)
            
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    # 하이퍼파라미터 설정
    batch_size = 32
    seq_length = 20
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 2048
    dropout = 0.1
    num_epochs = 10
    learning_rate = 0.0001
    vocab_size = 10000
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 데이터 로드
    print('Loading dummy dataset...')
    train_loader, val_loader, test_loader, vocab_size = load_dummy_data(batch_size, seq_length, vocab_size)
    print(f'Vocabulary size: {vocab_size}')
    
    # 모델 생성
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)
    
    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    # 학습 루프
    print('Starting training...')
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 학습
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 검증
        val_loss = evaluate(model, val_loader, criterion, device)
        
        # 에폭 시간 계산
        epoch_mins, epoch_secs = divmod(int(time.time() - start_time), 60)
        
        # 결과 출력
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}')
        
        # 최고 성능 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(f'Best model saved with validation loss: {val_loss:.3f}')

if __name__ == "__main__":
    main() 