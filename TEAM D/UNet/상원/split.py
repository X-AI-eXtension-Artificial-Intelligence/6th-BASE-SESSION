import os
import shutil
import random

## 다운받은 데이터가 나눠져 있지 않아서 비율대로 나눠서 학습을 진행##

# 경로 설정 (현재 구조에 맞춤)
base_dir = './datasets/Kvasir-SEG'
images_dir = os.path.join(base_dir, 'images')
masks_dir = os.path.join(base_dir, 'masks')

# 새로 나눌 경로
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# 비율 설정
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 폴더 생성
for split in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(split, 'masks'), exist_ok=True)

# 이미지 파일 리스트
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()  # 정렬 (선택)

# 셔플
random.shuffle(image_files)

# 나누기
total = len(image_files)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

train_files = image_files[:train_end]
val_files = image_files[train_end:val_end]
test_files = image_files[val_end:]

# 복사 함수
def copy_files(file_list, split_dir):
    for img_file in file_list:
        # 이미지 복사
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(split_dir, 'images', img_file)
        shutil.copy(src_img, dst_img)
        
        # 마스크 복사 (파일명 규칙: img1.jpg -> img1_mask.jpg)
        mask_name = img_file
        src_mask = os.path.join(masks_dir, mask_name)
        dst_mask = os.path.join(split_dir, 'masks', mask_name)
        
        if os.path.exists(src_mask):
            shutil.copy(src_mask, dst_mask)
        else:
            print(f"⚠️ 경고: {mask_name} 마스크가 존재하지 않음!")

# 복사 실행
print(f'총 {total}개 중 {len(train_files)}개 train, {len(val_files)}개 val, {len(test_files)}개 test로 분할.')

copy_files(train_files, train_dir)
copy_files(val_files, val_dir)
copy_files(test_files, test_dir)

print('✅ 데이터셋 분할 완료!')
