import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET
import cv2

class MoNuSegDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform

        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.tif')))
        self.annotation_paths = sorted(glob.glob(os.path.join(self.annotation_dir, '*.xml')))

        assert len(self.image_paths) == len(self.annotation_paths), "Image and Annotation count mismatch."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        ann_path = self.annotation_paths[idx]

        image = Image.open(img_path).convert('RGB')
        mask = self._load_mask_from_xml(ann_path, image.size)

        if self.transform:
            image = self.transform(image)

            # mask도 같이 transform
            mask = Image.fromarray(mask)  # numpy → PIL
            mask = transforms.Resize((256, 256), interpolation=Image.NEAREST)(mask)
            mask = transforms.ToTensor()(mask)  # (1, 256, 256)
        else:
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


    def _load_mask_from_xml(self, xml_path, image_size):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

        for region in root.iter('Region'):
            vertices = []
            for vertex in region.iter('Vertex'):
                x = float(vertex.attrib['X'])
                y = float(vertex.attrib['Y'])
                vertices.append((int(x), int(y)))

            if vertices:
                cv2.fillPoly(mask, [np.array(vertices, dtype=np.int32)], 1)

        return mask
