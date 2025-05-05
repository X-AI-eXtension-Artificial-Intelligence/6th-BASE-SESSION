import os
import glob
import shutil
from sklearn.model_selection import train_test_split

# 💾 원본 경로 설정 (예: TCGA 폴더 경로)
source_dir = '/home/work/xai/unet코드변경/datasets/lgg-mri-segmentation/kaggle_3m/TCGA_DU_7294_19890104'

# 📦 대상 디렉토리
target_root = './datasets'
os.makedirs(os.path.join(target_root, 'train'), exist_ok=True)
os.makedirs(os.path.join(target_root, 'val'), exist_ok=True)
os.makedirs(os.path.join(target_root, 'test'), exist_ok=True)

# 🔍 모든 마스크 파일 경로
mask_paths = sorted(glob.glob(os.path.join(source_dir, '*_mask.tif')))
image_paths = [p.replace('_mask.tif', '.tif') for p in mask_paths]

# 📊 데이터셋 분할 (8:1:1 비율)
train_imgs, temp_imgs = train_test_split(image_paths, test_size=0.2, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

# 📂 복사 함수 정의
def copy_pairs(image_list, target_dir):
    for img_path in image_list:
        mask_path = img_path.replace('.tif', '_mask.tif')
        shutil.copy(img_path, os.path.join(target_dir, os.path.basename(img_path)))
        shutil.copy(mask_path, os.path.join(target_dir, os.path.basename(mask_path)))

# 🛠 분할 실행
copy_pairs(train_imgs, os.path.join(target_root, 'train'))
copy_pairs(val_imgs, os.path.join(target_root, 'val'))
copy_pairs(test_imgs, os.path.join(target_root, 'test'))

print(f"✅ 분할 완료: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
