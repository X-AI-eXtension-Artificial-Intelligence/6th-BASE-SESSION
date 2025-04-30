import os
import numpy as np
import tifffile as tiff

def save_slices(volume_path, label_path, out_dir, prefix="train"):
    os.makedirs(os.path.join(out_dir, prefix), exist_ok=True)

    # tif 불러오기
    volume = tiff.imread(volume_path)  # shape: (Z, H, W)
    if label_path:
        labels = tiff.imread(label_path)
        assert volume.shape == labels.shape, "입력 이미지와 라벨 크기 불일치"

    # 슬라이싱 & 저장
    for i in range(volume.shape[0]):
        input_slice = volume[i]
        input_path = os.path.join(out_dir, prefix, f"input_{i:03d}.npy")
        np.save(input_path, input_slice)

        if label_path:
            label_slice = labels[i]
            label_path_out = os.path.join(out_dir, prefix, f"label_{i:03d}.npy")
            np.save(label_path_out, label_slice)

        print(f"{prefix.upper()} saved slice {i:03d}")

# 실행
if __name__ == "__main__":
    os.makedirs("datasets", exist_ok=True)

    save_slices(
    volume_path="./datasets/train-volume.tif",
    label_path="./datasets/train-labels.tif",
    out_dir="./datasets",
    prefix="train"
)

    save_slices(
    volume_path="./datasets/test-volume.tif",
    label_path=None,
    out_dir="./datasets",
    prefix="test"
)

