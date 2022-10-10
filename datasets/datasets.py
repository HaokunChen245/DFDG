import os
import random

import torch
import torchvision.transforms as tfs
from PIL import Image, ImageFile
from torch.utils import data
from torch.utils.data import ConcatDataset, random_split
from torchvision.datasets import MNIST, SVHN, USPS

ImageFile.LOAD_TRUNCATED_IMAGES = True
class BaseDataset(data.Dataset):
    def __init__(self, dataset_root_dir, mode, domain, img_size):
        self.root_dir = dataset_root_dir
        self.imgs = []
        self.domain = domain
        self.mode = mode
        if mode == "test" or mode == "val":
            self.transforms = tfs.Compose(
                [
                    tfs.Resize((img_size, img_size)),
                    tfs.ToTensor(),
                    tfs.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif mode == "train":
            self.transforms = tfs.Compose(
                [
                    tfs.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                    tfs.RandomHorizontalFlip(),
                    tfs.ColorJitter(0.3, 0.3, 0.3, 0.3),
                    tfs.RandomGrayscale(),
                    tfs.ToTensor(),
                    tfs.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        return 0

    def collate_fn(self, batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs).cuda()
        labels = torch.stack(labels).cuda()
        return imgs, labels

    def get_images_with_cls(self, label, max_num=256):
        imgs = []
        labels = []
        for i in range(len(self.imgs)):
            temp = self.__getitem__(i)
            if temp[1] != label:
                continue
            imgs.append(temp[0])
            labels.append(temp[1])
            if len(imgs) > max_num:
                break

        imgs = torch.stack(imgs, 0).cuda()
        labels = torch.stack(labels, 0).cuda()
        return imgs, labels

class VLCS(BaseDataset):
    def __init__(
        self,
        dataset_root_dir,
        mode,
        domain,
        img_size=224,
    ):
        BaseDataset.__init__(self, dataset_root_dir, mode, domain, img_size)

        self.classes = ["bird", "car", "chair", "dog", "person"]
        self.source_domains = ["Caltech101", "LabelMe", "SUN09", "VOC2007"]

        # random split the image into train, val and test set.
        # test set containing the whole set of images.
        self.split_images()

    def create_split_file(self):
        folder_dir = os.path.join(self.root_dir, self.domain)
        names_val = []
        names_train = []
        num_tot = 0
        for i in os.listdir(folder_dir):
            if "txt" in i:
                continue
            temps = os.listdir(os.path.join(folder_dir, i))
            num_tot += len(temps)
            k = max(3, int(len(temps) * 0.1))
            f = random.sample(temps, k)

            t = [os.path.join(self.domain, i, p) for p in f]
            names_val = names_val + t

            for p in temps:
                if p in f:
                    continue
                names_train.append(os.path.join(self.domain, i, p))
        with open(os.path.join(folder_dir, "train.txt"), "w") as f:
            for p in names_train:
                f.write(p)
                f.write("\n")
        with open(os.path.join(folder_dir, "val.txt"), "w") as f:
            for p in names_val:
                f.write(p)
                f.write("\n")
        with open(os.path.join(folder_dir, "test.txt"), "w") as f:
            for p in names_val:
                f.write(p)
                f.write("\n")
            for p in names_train:
                f.write(p)
                f.write("\n")

    def split_images(self):
        if not os.path.exists(
            os.path.join(self.root_dir, self.domain, self.mode + ".txt")
        ):
            self.create_split_file()

        split_file = os.path.join(self.root_dir, self.domain, self.mode + ".txt")
        with open(split_file, "r") as f:
            for l in f.readlines():
                self.imgs.append(l[:-1])

    def __getitem__(self, index):
        p = self.imgs[index]
        img = self.transforms(Image.open(os.path.join(self.root_dir, p)).convert("RGB"))
        tag = p.split("/")[1].replace("_", " ")
        label = torch.tensor(self.classes.index(tag))

        return img, label


class OfficeHome(BaseDataset):
    def __init__(
        self,
        dataset_root_dir,
        mode,
        domain,
        img_size=224,
    ):
        BaseDataset.__init__(self, dataset_root_dir, mode, domain, img_size)

        classes = "Alarm Clock, Backpack, Batteries, Bed, Bike, Bottle, Bucket, Calculator, Calendar, Candles, Chair, Clipboards, Computer, Couch, Curtains, Desk Lamp, Drill, Eraser, Exit Sign, Fan, File Cabinet, Flipflops, Flowers, Folder, Fork, Glasses, Hammer, Helmet, Kettle, Keyboard, Knives, Lamp Shade, Laptop, Marker, Monitor, Mop, Mouse, Mug, Notebook, Oven, Pan, Paper Clip, Pen, Pencil, Postit Notes, Printer, Push Pin, Radio, Refrigerator, ruler, Scissors, Screwdriver, Shelf, Sink, Sneakers, Soda, Speaker, Spoon, Table, Telephone, Toothbrush, Toys, Trash Can, TV, Webcam"
        self.classes = [c.upper() for c in classes.split(", ")]
        self.source_domains = ["Art", "Clipart", "Real_World", "Product"]

        # random split the image into train, val and test set.
        # test set containing the whole set of images.
        self.split_images()

    def create_split_file(self):
        folder_dir = os.path.join(self.root_dir, self.domain)
        names_val = []
        names_train = []
        num_tot = 0
        for i in os.listdir(folder_dir):
            if "txt" in i:
                continue
            temps = os.listdir(os.path.join(folder_dir, i))
            num_tot += len(temps)
            k = max(3, int(len(temps) * 0.1))
            f = random.sample(temps, k)

            t = [os.path.join(self.domain, i, p) for p in f]
            names_val = names_val + t

            for p in temps:
                if p in f:
                    continue
                names_train.append(os.path.join(self.domain, i, p))
        with open(os.path.join(folder_dir, "train.txt"), "w") as f:
            for p in names_train:
                f.write(p)
                f.write("\n")
        with open(os.path.join(folder_dir, "val.txt"), "w") as f:
            for p in names_val:
                f.write(p)
                f.write("\n")
        with open(os.path.join(folder_dir, "test.txt"), "w") as f:
            for p in names_val:
                f.write(p)
                f.write("\n")
            for p in names_train:
                f.write(p)
                f.write("\n")

    def split_images(self):
        if not os.path.exists(
            os.path.join(self.root_dir, self.domain, self.mode + ".txt")
        ):
            self.create_split_file()

        split_file = os.path.join(self.root_dir, self.domain, self.mode + ".txt")
        with open(split_file, "r") as f:
            for l in f.readlines():
                self.imgs.append(l[:-1])

    def __getitem__(self, index):
        p = self.imgs[index]
        img = self.transforms(Image.open(os.path.join(self.root_dir, p)).convert("RGB"))
        tag = p.split("/")[1].replace("_", " ").upper()
        label = torch.tensor(self.classes.index(tag))

        return img, label


class PACS(BaseDataset):
    def __init__(
        self,
        dataset_root_dir,
        mode,
        domain,
        img_size=224,
    ):
        BaseDataset.__init__(self, dataset_root_dir, mode, domain, img_size)

        self.classes = [
            "dog",
            "elephant",
            "giraffe",
            "guitar",
            "horse",
            "house",
            "person",
        ]
        self.split_images()

    def split_images(self):
        split_dir = self.root_dir + "/splits/"
        for p in os.listdir(split_dir):
            if self.mode in p and self.domain in p:
                with open(split_dir + p, "r") as f:
                    for l in f.readlines():
                        self.imgs.append(l.split(" ")[0])

    def __getitem__(self, index):
        p = self.imgs[index]
        img = self.transforms(
            Image.open(self.root_dir + "/images/kfold/" + p).convert("RGB")
        )
        tag = p.split("/")[1]
        label = torch.tensor(self.classes.index(tag))

        return img, label


class miniDomainNet(BaseDataset):
    def __init__(
        self,
        dataset_root_dir,
        mode,
        domain,
        img_size=96,
    ):
        BaseDataset.__init__(self, dataset_root_dir, mode, domain, img_size)

        self.classes = [
            "umbrella",
            "television",
            "potato",
            "see_saw",
            "zebra",
            "dragon",
            "chair",
            "carrot",
            "sea_turtle",
            "helicopter",
            "teddy-bear",
            "sheep",
            "coffee_cup",
            "grapes",
            "helmet",
            "dolphin",
            "squirrel",
            "drums",
            "guitar",
            "basket",
            "pillow",
            "crocodile",
            "mushroom",
            "dog",
            "table",
            "anvil",
            "peanut",
            "fish",
            "bottlecap",
            "mosquito",
            "camera",
            "elephant",
            "ant",
            "bathtub",
            "butterfly",
            "dumbbell",
            "asparagus",
            "streetlight",
            "cat",
            "purse",
            "penguin",
            "calculator",
            "crab",
            "duck",
            "string_bean",
            "giraffe",
            "lion",
            "ceiling_fan",
            "The_Great_Wall_of_China",
            "fence",
            "alarm_clock",
            "monkey",
            "goatee",
            "pencil",
            "speedboat",
            "truck",
            "mug",
            "screwdriver",
            "train",
            "pear",
            "hammer",
            "castle",
            "flower",
            "skateboard",
            "feather",
            "raccoon",
            "cactus",
            "panda",
            "lipstick",
            "The_Eiffel_Tower",
            "aircraft_carrier",
            "bee",
            "rabbit",
            "eyeglasses",
            "kangaroo",
            "bus",
            "banana",
            "horse",
            "shoe",
            "saxophone",
            "cannon",
            "onion",
            "submarine",
            "computer",
            "flamingo",
            "cruise_ship",
            "lantern",
            "blueberry",
            "strawberry",
            "canoe",
            "spider",
            "compass",
            "foot",
            "broccoli",
            "axe",
            "bird",
            "blackberry",
            "laptop",
            "bear",
            "candle",
            "pineapple",
            "peas",
            "rifle",
            "fork",
            "mouse",
            "toe",
            "watermelon",
            "power_outlet",
            "cake",
            "chandelier",
            "cow",
            "vase",
            "snake",
            "frog",
            "whale",
            "microphone",
            "tiger",
            "cell_phone",
            "camel",
            "leaf",
            "pig",
            "rhinoceros",
            "swan",
            "lobster",
            "teapot",
            "cello",
        ]  # 126

        # follow Dassl
        self.split_images()

    def split_images(self):
        split_file = os.path.join(self.root_dir, f"{self.domain}_{self.mode}.txt")

        with open(split_file, "r") as f:
            for l in f.readlines():
                self.imgs.append(l.split(" ")[0])

    def __getitem__(self, index):
        p = self.imgs[index]
        img = self.transforms(Image.open(os.path.join(self.root_dir, p)).convert("RGB"))
        tag = p.split("/")[1]
        label = torch.tensor(self.classes.index(tag))

        return img, label


class MNISTM(data.Dataset):
    def __init__(self, root, transform):
        self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.transforms = transform
        self.imgs = []
        self.labels = []
        self.root = root

        with open(os.path.join(root, "mnist_m/mnist_m_train_labels.txt"), "r") as f:
            for l in f.readlines():
                l = l.strip("\n")
                self.imgs.append(
                    os.path.join("mnist_m/mnist_m_train/", l.split(" ")[0])
                )
                self.labels.append(int(l.split(" ")[1]))

        with open(os.path.join(root, "mnist_m/mnist_m_test_labels.txt"), "r") as f:
            for l in f.readlines():
                l = l.strip("\n")
                self.imgs.append(os.path.join("mnist_m/mnist_m_test/", l.split(" ")[0]))
                self.labels.append(int(l.split(" ")[1]))

    def __getitem__(self, index):
        p = self.imgs[index]
        img = self.transforms(Image.open(os.path.join(self.root, p)).convert("RGB"))
        label = torch.tensor(self.labels[index])

        return img, label

    def __len__(self):
        return len(self.imgs)


class Digits(BaseDataset):
    def __init__(
        self,
        dataset_root_dir,
        mode,
        domain,
        img_size=32,
    ):
        BaseDataset.__init__(self, dataset_root_dir, mode, domain, img_size)

        self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if domain == "USPS" or domain == "MNIST":
            if mode == "test" or mode == "val":
                self.transforms = tfs.Compose(
                    [
                        tfs.Resize((img_size, img_size)),
                        tfs.ToTensor(),
                        tfs.Lambda(lambda x: x.repeat(3, 1, 1)),
                        tfs.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            elif mode == "train":
                self.transforms = tfs.Compose(
                    [
                        tfs.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                        # tfs.RandomHorizontalFlip(),
                        tfs.ColorJitter(0.3, 0.3, 0.3, 0.3),
                        tfs.RandomGrayscale(),
                        tfs.ToTensor(),
                        tfs.Lambda(lambda x: x.repeat(3, 1, 1)),
                        tfs.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

        if domain == "USPS":
            dataset_train = USPS(
                root=dataset_root_dir, train=True, transform=self.transforms
            )
            dataset_test = USPS(
                root=dataset_root_dir, train=False, transform=self.transforms
            )
            dataset = ConcatDataset([dataset_train, dataset_test])
        elif domain == "MNIST":
            dataset_train = MNIST(
                root=dataset_root_dir, train=True, transform=self.transforms
            )
            dataset_test = MNIST(
                root=dataset_root_dir, train=False, transform=self.transforms
            )
            dataset = ConcatDataset([dataset_train, dataset_test])
        elif domain == "SVHN":
            dataset_train = SVHN(
                root=dataset_root_dir, split="train", transform=self.transforms
            )
            dataset_test = SVHN(
                root=dataset_root_dir, split="test", transform=self.transforms
            )
            dataset = ConcatDataset([dataset_train, dataset_test])
        elif domain == "MNISTM" or domain == "MNIST-M":
            dataset = MNISTM(root=dataset_root_dir, transform=self.transforms)

        trainset_len = int(len(dataset) * 0.9)
        if mode == "test":
            self.dataset = dataset
        else:
            trainset, valset = random_split(
                dataset,
                [trainset_len, len(dataset) - trainset_len],
                generator=torch.Generator().manual_seed(45),
            )  # fix the split
            if mode == "train":
                self.dataset = trainset
            elif "val" in mode:
                self.dataset = valset


    def __getitem__(self, index):
        img, label = self.dataset[index]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        return img, label

    def __len__(self):
        return len(self.dataset)
