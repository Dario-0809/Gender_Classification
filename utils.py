from library import *
from config import *

def make_datapath_list(phase="train"):
    rootpath = "./data/gender-classification-dataset/"
    target_path = osp.join(rootpath+phase+"/**/*.jpg")
    path_list = []
    for path in glob.glob(target_path):
        path_list.append(path)
    
    return path_list


def train_model(model, dataloader_dict, criterion, scheduler, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.to(device)
        torch.backends.cudnn.benchmark = True

        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloader_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    torch.save(model.state_dict(), save_path)


def load_model(model, model_path):
    load_weights = torch.load(model_path)
    model.load_state_dict(load_weights)
    return model