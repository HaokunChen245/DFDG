''' 
@copyright Copyright (c) Siemens AG, 2022
@author Haokun Chen <haokun.chen@siemens.com>
SPDX-License-Identifier: Apache-2.0
'''

from dfdg.datasets.datasets import PACS
from dfdg.datasets.datasets import VLCS
from dfdg.datasets.datasets import Digits
from dfdg.datasets.datasets import OfficeHome
from dfdg.datasets.datasets import miniDomainNet


def get_dataset_stats(dataset):
    image_mean = [0.485, 0.456, 0.406]
    image_var = [0.229, 0.224, 0.225]
    return image_mean, image_var


def get_img_size(dataset):
    dataset_size_mapping = {
        'PACS': 224,
        'VLCS': 224,
        'OfficeHome': 224,
        'Digits': 32,
        'miniDomainNet': 96,
    }
    return dataset_size_mapping[dataset]


def get_backbone(dataset):
    backbone = None
    if dataset == 'Digits':
        backbone = 'resnet18'
    else:
        backbone = 'resnet50'
    return backbone


def get_dataset(dataset):
    dataset_mapping = {
        'PACS': PACS,
        'VLCS': VLCS,
        'OfficeHome': OfficeHome,
        'Digits': Digits,
        'miniDomainNet': miniDomainNet,
    }
    assert dataset in dataset_mapping.keys()
    dataset_class = dataset_mapping[dataset]
    return dataset_class


def get_source_domains(dataset):
    if dataset == "PACS":
        return ["art_painting", "cartoon", "photo", "sketch"]
    elif dataset == "Digits":
        return ["MNIST", "MNISTM", "SVHN", "USPS"]
    elif dataset == "VLCS":
        return ["Caltech101", "LabelMe", "SUN09", "VOC2007"]
    elif dataset == "OfficeHome":
        return ["Art", "Clipart", "Product", "Real_World"]
    elif dataset == "miniDomainNet":
        return ["real", "clipart", "painting", "sketch"]


def get_source_domains_abbr(dataset):
    if dataset == "PACS":
        return ["A", "C", "P", "S"]
    elif dataset == "Digits":
        return ["MT", "MM", "SV", "UP"]
    elif dataset == "OfficeHome":
        return ["A", "C", "P", "R"]
    elif dataset == "miniDomainNet":
        return ["R", "C", "P", "S"]


def get_abbr_map(dataset):
    if dataset == "PACS":
        return {
            "art": "A",
            "cartoon": "C",
            "photo": "P",
            "sketch": "S",
            "art_painting": "A",
        }
    elif dataset == "Digits":
        return {
            "MNIST": "MT",
            "MNISTM": "MM",
            "SVHN": "SV",
            "USPS": "UP",
        }
    elif dataset == "OfficeHome":
        return ["A", "C", "P", "R"]

    elif dataset == "miniDomainNet":
        return {
            "real": "R",
            "clipart": "C",
            "painting": "P",
            "sketch": "S",
        }


def get_class_number(dataset):
    if dataset == "PACS":
        return 7
    elif dataset == "VLCS":
        return 5
    elif dataset == "OfficeHome":
        return 65
    elif dataset == "Digits":
        return 10
    elif dataset == "miniDomainNet":
        return 126


def get_class_names(dataset):
    if dataset == "PACS":
        return [
            "dog",
            "elephant",
            "giraffe",
            "guitar",
            "horse",
            "house",
            "person",
        ]
    elif dataset == "VLCS":
        return ["bird", "car", "chair", "dog", "person"]
    elif dataset == "OfficeHome":
        classes = (
            "Alarm Clock, Backpack, Batteries, Bed, Bike, Bottle, Bucket, Calculator, "
            "Calendar, Candles, Chair, Clipboards, Computer, Couch, Curtains, Desk Lamp, Drill, "
            "Eraser, Exit Sign, Fan, File Cabinet, Flipflops, Flowers, Folder, Fork, Glasses, Hammer, ÄHelmet, Kettle, Keyboard, Knives, Lamp Shade, Laptop, Marker, Monitor, Mop, Mouse, Mug, Notebook, Oven, Pan, Paper Clip, Pen, Pencil, Postit Notes, Printer, Push Pin, Radio, Refrigerator, ruler, Scissors, Screwdriver, Shelf, Sink, Sneakers, Soda, Speaker, Spoon, Table, Telephone, Toothbrush, Toys, Trash Can, TV, Webcam"
        )
        return [c.upper() for c in classes.split(", ")]
    elif dataset == "Digits":
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif dataset == "miniDomainNet":
        return [
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
        ]
