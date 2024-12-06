import os
import torch
import pandas as pd
from PIL import Image

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, images_dir, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.data['Image_ID'].unique())

    def __getitem__(self, idx):
        # Get the unique image
        image_id = self.data['Image_ID'].unique()[idx]
        image_path = os.path.join(self.images_dir, image_id)
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Get bounding boxes and labels for this image
        records = self.data[self.data['Image_ID'] == image_id]
        boxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values
        labels = records['class'].astype('category').cat.codes.values + 1  # Labels start at 1
        
        # Convert to PyTorch tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        # Prepare target
        target = {
            "boxes": boxes,
            "labels": labels,
        }
        
        if self.transforms:
            # Albumentations or torchvision transforms
            transformed = self.transforms(image=np.array(image), bboxes=boxes, labels=labels)
            image = transformed["image"]
            target["boxes"] = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        
        return image, target