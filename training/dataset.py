import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FER2013Dataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir example:
        data/FER2013/images
        """
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.class_names = sorted(os.listdir(root_dir))

        for label, emotion in enumerate(self.class_names):
            emotion_dir = os.path.join(root_dir, emotion)
            for img_name in os.listdir(emotion_dir):
                self.image_paths.append(os.path.join(emotion_dir, img_name))
                self.labels.append(label)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
