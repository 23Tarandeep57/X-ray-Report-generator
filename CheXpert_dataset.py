import numpy as np
import pandas as pd
from torchvision import transforms

class CheXpertDataset(Dataset):
    """
    Expects a CSV with columns: Path, and 14 binary label columns.
    """
    def __init__(self, img_dir, csv_file, label_cols, transform=None):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)
        # Replace uncertain labels (-1) with 0
        self.df[label_cols] = self.df[label_cols].fillna(0).replace(-1, 0)
        self.label_cols = label_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Path'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        labels = torch.tensor(row[self.label_cols].values.astype(float), dtype=torch.float32)
        return image, labels
