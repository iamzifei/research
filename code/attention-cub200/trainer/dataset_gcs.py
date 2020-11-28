import os
import cv2
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from google.cloud import storage

# from google.colab import auth
# auth.authenticate_user()


class CUB(Dataset):
    bucketName = "attention_bucket"
    path = "data/CUB200/CUB_200_2011"
    project = 'ai-lab-290600'
    threshold = 100

    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucketName)

    def __init__(self, train=True, transform=None, target_transform=None):
        self.is_train = train
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = {}

        for idx, line in enumerate(self._read_blob('images.txt').splitlines()):
            image_id, path = line.split()
            self.images_path[image_id] = path
            if idx >= self.threshold:
                break

        self.class_ids = {}
        for idx, line in enumerate(self._read_blob('image_class_labels.txt').splitlines()):
            image_id, class_id = line.split()
            self.class_ids[image_id] = class_id
            if idx >= self.threshold:
                break

        self.data_id = []
        if self.is_train:
            for idx, line in enumerate(self._read_blob('train_test_split.txt').splitlines()):
                image_id, is_train = line.split()
                if int(is_train):
                    self.data_id.append(image_id)
                if idx >= self.threshold:
                    break
        if not self.is_train:
            for idx, line in enumerate(self._read_blob('train_test_split.txt').splitlines()):
                image_id, is_train = line.split()
                if not int(is_train):
                    self.data_id.append(image_id)
                if idx >= self.threshold:
                    break

    def _read_blob(self, file_name, asFile=False):
        # for f in self.bucket.list_blobs():
        # print(f)
        filename = self.path + '/' + file_name
        blob = self.bucket.get_blob(filename)
        print('downloading blob {}.'.format(filename))
        if asFile:
          # create the parent nested folder if not exist
            head_tail = os.path.split(filename)
            Path(head_tail[0]).mkdir(parents=True, exist_ok=True)
            # save the file in file system and return the path
            blob.download_to_filename(filename)
            return filename
        return blob.download_as_string().decode("utf-8")

    def __len__(self):
        return len(self.data_id)

    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        image_id = self.data_id[index]
        class_id = int(self._get_class_by_id(image_id)) - 1
        path = self._get_path_by_id(image_id)
        image = cv2.imread(self._read_blob('images/'+path, True))

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            class_id = self.target_transform(class_id)
        return image, class_id

    def _get_path_by_id(self, image_id):

        return self.images_path[image_id]

    def _get_class_by_id(self, image_id):

        return self.class_ids[image_id]

# if __name__ == '__main__':
#     dataset = CUB()
#     print(len(dataset))
