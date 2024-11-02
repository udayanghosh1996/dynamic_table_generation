from utils.Dataset_Split import *
from train.train_function import train_function

if __name__ == "__main__":
    print("inside main")
    train_loader, val_loader = get_train_val_dataset(train_batch=1, val_batch=1, train_shuffle=True,
                                                     val_shuffle=False)

    print("After dataloader")
    train_function(train_loader, val_loader, 70, 1e-07, 50)
