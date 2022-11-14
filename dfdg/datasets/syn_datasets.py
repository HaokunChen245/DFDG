''' 
@copyright Copyright (c) Siemens AG, 2022
@author Haokun Chen <haokun.chen@siemens.com>
SPDX-License-Identifier: Apache-2.0
'''

import os

import torch
import torch.utils.data as data
import torchvision.transforms as tfs
from dfdg.datasets.utils import get_abbr_map
from dfdg.datasets.utils import get_class_names
from dfdg.datasets.utils import get_source_domains_abbr


class FAKE(data.Dataset):
    '''Dataset for synthetic images'''

    def __init__(
        self,
        dataset,
        source_domain_a,
        image_root_dir,
        folder_name,
        mode,
        portion=-1,
        augment_in_fake=False,
        target_domain=None,
    ):
        if dataset == "Digits":
            img_size = 32
        elif dataset == "miniDomainNet":
            img_size = 96
        else:
            img_size = 224

        mapp = get_abbr_map(dataset)
        domains = get_source_domains_abbr(dataset)

        if augment_in_fake:
            if dataset == "Digits":
                self.transforms = tfs.Compose(
                    [
                        tfs.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                        # tfs.RandomHorizontalFlip(),
                        tfs.ColorJitter(0.3, 0.3, 0.3, 0.3),
                        tfs.RandomGrayscale(),
                        tfs.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]
                )
            else:
                self.transforms = tfs.Compose(
                    [
                        tfs.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                        tfs.RandomHorizontalFlip(),
                        tfs.ColorJitter(0.3, 0.3, 0.3, 0.3),
                        tfs.RandomGrayscale(),
                        tfs.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]
                )
        else:
            self.transforms = tfs.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        self.class_names = get_class_names(dataset)
        self.num_class = len(self.class_names)

        self.imgs = []
        self.labels = []
        self.index_teachers = []

        # find the correct path of image storage
        self.image_folder_dir = os.path.join(image_root_dir, folder_name)
        assert os.path.exists(
            self.image_folder_dir
        ), f"failure when loading the image: {self.image_folder_dir}"
        if target_domain:
            abbr_target = mapp[target_domain]
        else:
            abbr_target = "----"

        if "stage2" in mode:
            # setting is adabn with multiple teacher,
            # and here the logits are calculated using the teacher from source B.
            # loading all images in that image folder

            for f in os.listdir(self.image_folder_dir):
                if not ".pt" in f or not "img" in f:
                    continue
                if source_domain_a != f.split("_")[1]:
                    continue
                if abbr_target in f or (target_domain and target_domain in f):
                    continue

                img_batch = torch.load(os.path.join(self.image_folder_dir, f))
                label_batch = torch.load(
                    os.path.join(
                        self.image_folder_dir, f.replace("img", "label")
                    )
                )

                source_domain_b = f.split("_")[2]

                temp = torch.Tensor([0, 0, 0, 0])
                # every teacher logits takes weight of 0.5
                temp[domains.index(mapp[source_domain_a])] = 0.5
                temp[domains.index(mapp[source_domain_b])] = 0.5
                self.index_teachers.append(temp)

                for i in range(img_batch.shape[0]):
                    self.imgs.append([f, i])
                    self.labels.append(int(label_batch[i]))
                    self.index_teachers.append(temp)

        elif "stage1" in mode:
            # loading all images in that image folder
            temp = torch.Tensor([0, 0, 0, 0])
            # the only teacher takes 1
            temp[domains.index(mapp[source_domain_a])] = 1

            for f in os.listdir(self.image_folder_dir):
                if not ".pt" in f or not "img" in f:
                    continue
                if source_domain_a != f.split("_")[1]:
                    continue
                img_batch = torch.load(os.path.join(self.image_folder_dir, f))
                label_batch = torch.load(
                    os.path.join(
                        self.image_folder_dir, f.replace("img", "label")
                    )
                )

                for i in range(img_batch.shape[0]):
                    self.imgs.append([f, i])
                    self.labels.append(int(label_batch[i]))
                    self.index_teachers.append(temp)
                del img_batch

        else:
            assert False, "Can not parse argument mode"

        if portion != -1:
            new_len = int(len(self.imgs) * portion)
            self.imgs = self.imgs[:new_len]

        torch.cuda.empty_cache()

    def __getitem__(self, index):
        with torch.no_grad():
            img_batch = torch.load(
                os.path.join(self.image_folder_dir, self.imgs[index][0])
            )
            img = img_batch[self.imgs[index][1]]
            img = self.transforms(img)
            del img_batch
        return (
            img,
            torch.LongTensor([self.labels[index]]),
            self.index_teachers[index],
        )

    def __len__(self):
        return len(self.imgs)

    def collate_fn(self, batch):
        imgs, labels, index_teachers = zip(*batch)
        imgs = torch.stack(imgs).cuda()
        labels = torch.stack(labels).squeeze(-1).cuda()
        index_teachers = torch.stack(index_teachers).cuda()
        return imgs, labels, index_teachers
