from library import *
from image_transform import ImageTransform
from config import *
from utils import make_datapath_list, train_model, load_model
from dataset import MyDataset
from network import CNN

def main():
    train_list = make_datapath_list("train")
    val_list = make_datapath_list("val")

    # dataset
    train_dataset = MyDataset(train_list, transform=ImageTransform(resize, mean, std), phase="train")
    val_dataset = MyDataset(val_list, transform=ImageTransform(resize, mean, std), phase="val")

    # dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle = False)
    dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

    # network
    model = CNN()

    #loss
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # training
    train_model(model, dataloader_dict, criterion, exp_lr_scheduler, optimizer, num_epochs=2)

if __name__ == "__main__":
    # main()
    model = CNN()
    load_model(model, save_path)
