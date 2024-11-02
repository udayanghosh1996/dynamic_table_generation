import json
import os
from PIL import Image
from torch.utils.data import Dataset, random_split


class PubTabNet_Dataset(Dataset):
    def __init__(self, json_file, images_folder, tokenizer, transform=None, split=None, train_ratio=0.8):
        with open(json_file, 'r') as file:
            self.labels = json.load(file)
        self.images_folder = images_folder
        self.transform = transform
        self.tokenizer = tokenizer

        if split is not None:
            assert split in ['train', 'val'], "split must be 'train', 'val', or None"
            train_size = int(train_ratio * len(self.labels))
            val_size = len(self.labels) - train_size
            train_dataset, val_dataset = random_split(self.labels, [train_size, val_size])

            if split == 'train':
                self.labels = train_dataset
            else:
                self.labels = val_dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_folder, self.labels[idx]['filename'])
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]['html']

        if self.transform:
            image = self.transform(image)

        encoding = self.tokenizer(label, return_tensors="pt", padding="max_length", truncation=True, max_length=2048)
        return {"pixel_values": image, "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0)}
