import torch
import torch.nn as nn
from torch.optim import Adam
from data_loader import load_wikitext2
from transformer import Transformer
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data, target)
        
        # Reshape output and target for loss calculation
        output = output.view(-1, output.size(-1))
        target = target.view(-1)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data, target)
            
            # Reshape output and target for loss calculation
            output = output.view(-1, output.size(-1))
            target = target.view(-1)
            
            loss = criterion(output, target)
            total_loss += loss.item()
            
    return total_loss / len(val_loader)

def main():
    # 하이퍼파라미터 설정
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001
    
    # 데이터 로드
    train_loader, val_loader, test_loader, vocab = load_wikitext2(batch_size=batch_size)
    vocab_size = len(vocab)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 초기화
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=dim_feedforward,
        dropout=dropout
    ).to(device)
    
    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # 학습 및 검증 손실 기록
    train_losses = []
    val_losses = []
    
    # 학습 루프
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 학습
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # 검증
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
    
    # 학습 곡선 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.close()
    
    # 모델 저장
    torch.save(model.state_dict(), 'transformer_model.pth')
    print("Model saved as 'transformer_model.pth'")

if __name__ == "__main__":
    main() 