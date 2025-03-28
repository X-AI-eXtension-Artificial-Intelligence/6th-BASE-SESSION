import torch
import torchvision.transforms as transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from PIL import Image

# 1. Hugging Face 포켓몬 데이터셋 로드
pokemon_dataset = load_dataset("keremberke/pokemon-classification", name="full")

# 2. 전체 포켓몬 이름 라벨 (150개)
class_names = [
    'Porygon', 'Goldeen', 'Hitmonlee', 'Hitmonchan', 'Gloom', 'Aerodactyl', 'Mankey', 'Seadra', 'Gengar', 'Venonat', 
    'Articuno', 'Seaking', 'Dugtrio', 'Machop', 'Jynx', 'Oddish', 'Dodrio', 'Dragonair', 'Weedle', 'Golduck', 
    'Flareon', 'Krabby', 'Parasect', 'Ninetales', 'Nidoqueen', 'Kabutops', 'Drowzee', 'Caterpie', 'Jigglypuff', 'Machamp', 
    'Clefairy', 'Kangaskhan', 'Dragonite', 'Weepinbell', 'Fearow', 'Bellsprout', 'Grimer', 'Nidorina', 'Staryu', 'Horsea', 
    'Electabuzz', 'Dratini', 'Machoke', 'Magnemite', 'Squirtle', 'Gyarados', 'Pidgeot', 'Bulbasaur', 'Nidoking', 'Golem', 
    'Dewgong', 'Moltres', 'Zapdos', 'Poliwrath', 'Vulpix', 'Beedrill', 'Charmander', 'Abra', 'Zubat', 'Golbat', 
    'Wigglytuff', 'Charizard', 'Slowpoke', 'Poliwag', 'Tentacruel', 'Rhyhorn', 'Onix', 'Butterfree', 'Exeggcute', 'Sandslash', 
    'Pinsir', 'Rattata', 'Growlithe', 'Haunter', 'Pidgey', 'Ditto', 'Farfetchd', 'Pikachu', 'Raticate', 'Wartortle', 
    'Vaporeon', 'Cloyster', 'Hypno', 'Arbok', 'Metapod', 'Tangela', 'Kingler', 'Exeggutor', 'Kadabra', 'Seel', 
    'Voltorb', 'Chansey', 'Venomoth', 'Ponyta', 'Vileplume', 'Koffing', 'Blastoise', 'Tentacool', 'Lickitung', 'Paras', 
    'Clefable', 'Cubone', 'Marowak', 'Nidorino', 'Jolteon', 'Muk', 'Magikarp', 'Slowbro', 'Tauros', 'Kabuto', 
    'Spearow', 'Sandshrew', 'Eevee', 'Kakuna', 'Omastar', 'Ekans', 'Geodude', 'Magmar', 'Snorlax', 'Meowth', 
    'Pidgeotto', 'Venusaur', 'Persian', 'Rhydon', 'Starmie', 'Charmeleon', 'Lapras', 'Alakazam', 'Graveler', 'Psyduck', 
    'Rapidash', 'Doduo', 'Magneton', 'Arcanine', 'Electrode', 'Omanyte', 'Poliwhirl', 'Mew', 'Alolan Sandslash', 'Mewtwo', 
    'Weezing', 'Gastly', 'Victreebel', 'Ivysaur', 'MrMime', 'Shellder', 'Scyther', 'Diglett', 'Primeape', 'Raichu'
]

# 2-1. 전체 데이터셋(모든 split)에서 실제 사용되는 고유 레이블 추출  
all_labels = set()
for split in pokemon_dataset.keys():
    all_labels.update({int(label) for label in pokemon_dataset[split]["labels"]})
unique_labels = sorted(all_labels)
print("Unique dataset labels:", unique_labels)

# 2-2. 전체 데이터를 대상으로 원래 레이블(문자열) -> 연속적인 레이블(0부터 시작)로 재매핑  
# (예: unique_labels가 [0, 1, 2, …, 149]라면 매핑은 '0'->0, '1'->1, ...)
orig_label_to_new_index = {str(orig): new for new, orig in enumerate(unique_labels)}

# 2-3. 실제 포함된 레이블에 해당하는 class_names만 필터링  
# 데이터셋의 레이블이 0부터 시작하는 경우, 인덱스로 바로 접근
filtered_class_names = [class_names[label] for label in unique_labels]

# 3. 전처리: 이미지 -> 텐서 변환 및 정규화
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 4. 각 샘플 전처리 함수
def process_example(example):
    # 이미지 필드가 파일 경로 문자열인 경우, PIL 이미지로 열어 RGB로 변환
    if isinstance(example["image"], str):
        example["image"] = Image.open(example["image"]).convert("RGB")
    example["image"] = transform(example["image"])
    
    # 원래 레이블 저장 (시각화를 위해)
    orig = int(example["labels"])
    example["orig_label"] = orig
    # 전체 데이터셋의 mapping을 사용하여 연속적인 레이블로 재매핑
    example["labels"] = orig_label_to_new_index[str(orig)]
    
    return example

# 5. 모든 split에 대해 전처리 적용
pokemon_dataset = pokemon_dataset.map(process_example, batched=False)

# 6. Hugging Face Dataset을 PyTorch 텐서 형식으로 설정 (시각화를 위해 orig_label 포함)
pokemon_dataset.set_format("torch", columns=["image", "labels", "orig_label"])

# 7. DataLoader 생성 함수
def dataset_loader(split="train", batch_size=32, shuffle=True):
    return DataLoader(pokemon_dataset[split], batch_size=batch_size, shuffle=shuffle)

train_loader = dataset_loader("train", 32, True)

# 8. 시각화 함수
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('pokemon_Dataset.png')
    plt.show()

def random_visualize(data_loader, class_names_filtered):
    batch = next(iter(data_loader))
    images = batch["image"]
    new_labels = batch["labels"]

    print("images shape:", images.shape)
    grid_img = torchvision.utils.make_grid(images, nrow=8, padding=2, normalize=True)
    imshow(grid_img)

    names = [class_names_filtered[label.item()] for label in new_labels]
    print(" ".join(names))

# 시각화 실행 (train split 기준)
random_visualize(train_loader, filtered_class_names)
