from torchvision import transforms
from utils.Dataset_Creator import *
from torch.utils.data import DataLoader
import os
from transformers import CLIPTokenizer


main_dataset_path = os.path.join(os.path.join(os.getcwd(), "Dataset"), "train")
test_dataset_path = os.path.join(os.path.join(os.getcwd(), "Dataset"), "val")


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

main_json_file = os.path.join(main_dataset_path, 'labels.json')
main_images_folder = os.path.join(main_dataset_path, 'images')

test_json_file = os.path.join(test_dataset_path, 'labels.json')
test_images_folder = os.path.join(test_dataset_path, 'images')

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", force_download=True)


def get_train_val_dataset(train_batch=15, val_batch=15, train_shuffle=True, val_shuffle=False):
    
    train_dataset = PubTabNet_Dataset(json_file=main_json_file, images_folder=main_images_folder, tokenizer=tokenizer,
                                      transform=transform, split="train", train_ratio=0.8)
    val_dataset = PubTabNet_Dataset(json_file=main_json_file, images_folder=main_json_file, tokenizer=tokenizer,
                                    transform=transform, split='val', train_ratio=0.8)

    train = DataLoader(train_dataset, batch_size=train_batch, shuffle=train_shuffle)

    val = DataLoader(val_dataset, batch_size=val_batch, shuffle=val_shuffle)

    return train, val


def get_test_dataset(batch=15, shuffle=True):
    test_dataset = PubTabNet_Dataset(json_file=test_json_file, images_folder=test_images_folder, tokenizer=tokenizer,
                                     transform=transform, split=None, train_ratio=0.0)

    test = DataLoader(test_dataset, batch_size=batch, shuffle=shuffle)

    return test
