import csv, os, torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms 
from PIL import Image

class Keypoints_image_dataset(Dataset):
    def __init__(self, dataset, mode):
        super(Keypoints_image_dataset, self).__init__()
        self.dataset = dataset
        self.names, self.labels = self.read_data()
        if mode == "train":
            self.names = self.names[:int(0.8*(len(self.names)))]
            self.labels = self.labels[:int(0.8*(len(self.labels)))]
        else:
            self.names = self.names[int(0.8*(len(self.names))):]
            self.labels = self.labels[int(0.8*(len(self.labels))):]
        pass
    
    # read name/label 
    def read_data(self):
        name_list, label_list = [], []
        with open(self.dataset, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                name, label = row
                name_list.append(name)
                label_list.append(int(label))
        assert len(name_list) == len(label_list)
        return name_list, label_list
    
    def transfroms(self, x):
        tf = transforms.Compose([
            lambda x:Image.open(x),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
        out = tf(x)
        return out

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name, label = self.names[index], self.labels[index]
        images_list = []

        for path in (os.listdir(name)):
            dir = name+"/"+path
            img = self.transfroms(dir)
            images_list.append(img)
        
        name = torch.stack([image for image in images_list], dim=0)
        label = torch.tensor(int(label))
        return name, label
            

if __name__ == "__main__":
    dataset_path = "dataset_list.csv"
    train_dataset = Keypoints_image_dataset(dataset_path, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=2)

    test_dataset = Keypoints_image_dataset(dataset_path, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True, num_workers=2)
    image, label = next(iter(train_loader))
    print(image.shape, label.shape)
    # print(len(train_loader.dataset))
    # print(len(test_loader.dataset))