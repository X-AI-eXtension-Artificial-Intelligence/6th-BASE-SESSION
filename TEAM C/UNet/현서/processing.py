from PIL import Image
import numpy as np
import os
import cv2  # 꼭 import 해주세요!


def preprocess_brain_images(src_dir, dst_dir, size=(512, 512)):
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(src_dir) if f.endswith('.tif') and 'mask' not in f])
    for i, file in enumerate(files[:10]):
        img = Image.open(os.path.join(src_dir, file)).convert('L').resize(size)
        mask_file = file.replace('.tif', '') + '_mask.tif'
        mask = Image.open(os.path.join(src_dir, mask_file)).convert('L').resize(size)

        img_np = np.array(img)
        mask_np = np.array(mask)

        np.save(os.path.join(dst_dir, f'input_{i:03d}.npy'), img_np)
        np.save(os.path.join(dst_dir, f'input_canny_{i:03d}.npy'), cv2.Canny(img_np, 100, 200))
        np.save(os.path.join(dst_dir, f'label_{i:03d}.npy'), mask_np)
        print(f'save input_{i:03d}.npy done.')



if __name__ == '__main__':
    preprocess_brain_images('./data_brain', './data_brain/test')

# from PIL import Image
# import numpy as np
# import os
# import cv2

# def preprocess_brain_images(src_dir, dst_dir, size=(512, 512)):
#     print(f'✅ preprocess_brain_images 실행 시작\n   src_dir: {src_dir}\n   dst_dir: {dst_dir}')
    
#     os.makedirs(dst_dir, exist_ok=True)
#     files = sorted([f for f in os.listdir(src_dir) if f.endswith('.tif') and 'mask' not in f])

#     if not files:
#         print('⚠️ 경고: src_dir에 .tif 파일이 없습니다.')
#         return

#     for i, file in enumerate(files[:10]):  # 앞에서 10개만 사용
#         img_path = os.path.join(src_dir, file)
#         mask_path = os.path.join(src_dir, file.replace('.tif', '_mask.tif'))

#         print(f'🔹 처리 중: {img_path}')
#         print(f'🔹 마스크 찾는 중: {mask_path}')
        
#         if not os.path.exists(mask_path):
#             print(f'❌ 마스크 파일 없음: {mask_path}, 건너뜀')
#             continue

#         img = Image.open(img_path).convert('L').resize(size)
#         mask = Image.open(mask_path).convert('L').resize(size)

#         img_np = np.array(img)
#         mask_np = np.array(mask)

#         np.save(os.path.join(dst_dir, f'input_{i:03d}.npy'), img_np)
#         np.save(os.path.join(dst_dir, f'input_canny_{i:03d}.npy'), cv2.Canny(img_np, 100, 200))
#         np.save(os.path.join(dst_dir, f'label_{i:03d}.npy'), mask_np)

#     print('✅ 모든 처리 완료')

# if __name__ == '__main__':
#     print('✅ 실행: processing.py 시작')
#     preprocess_brain_images('./data_brain', './data_brain/test')
