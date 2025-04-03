import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class StarsGoalHornDataset(Dataset):
    def __init__(self, root_dir, input_dim):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.image_paths = []
        self.image_labels = []
        self.transform = transforms.Compose([
            transforms.Resize((input_dim[0], input_dim[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                self.image_paths.append(image_path)
                self.image_labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.image_labels[idx]
        image = self.transform(image)

        return image, label