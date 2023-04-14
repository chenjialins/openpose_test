from Dataset import Keypoints_image_dataset
from lrcn_model import ConvLstm
from torch.utils.data import DataLoader
import torch, csv

# GPU device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# train Function
def train(epochs, model, criterion, optimizer, trainData):
    model = model.to(device)
    loss_epoch = 0
    loss_train = []
    
    # train model and display loss data
    for epoch in range(epochs):
        print("[==========epoch:{}/{}==========]".format(epoch+1, epochs))
        for batch_idx, (x, y) in enumerate(trainData):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_batch = criterion(out, y) 

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            loss_epoch += loss_batch
            print("batch_idx:{}, loss_batch:{:0.8f}".format(batch_idx, loss_batch))

        loss_epoch = loss_epoch/(len(trainData))
        loss_train.append(loss_epoch)
        print("epoch:{}, loss_epoch:{}".format(epoch+1, loss_epoch))
    print("model train is done ")

    # save loss data and checkpoint pth
    with open("loss_epoch.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for loss in loss_train:
            writer.writerow([loss])
    print("write loss into file is done")

    torch.save(model, "checkpoint.pth")
    print("save checkpoint is done")


if __name__ == "__main__":
    # Init dataset ==> trainData/testData
    dataset_path = "dataset_list.csv"
    train_dataset = Keypoints_image_dataset(dataset_path, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4)

    # Init params ==> train(epochs, model, criterion, optimizer, trainData):
    epochs = 100
    net = ConvLstm(latent_dim=512, hidden_size=256, 
                   lstm_layers=2, bidirectional=True, n_class=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.002, betas=(0.9, 0.99))
    
    # start train
    train(epochs, net, criterion, optimizer, train_loader)