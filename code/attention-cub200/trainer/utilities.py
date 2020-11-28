import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
from matplotlib import pyplot as plt

from google.cloud import storage
from torchvision.datasets.utils import extract_archive, download_file_from_google_drive


def download_cub(extract_root='./data'):
    file_id = '10A3CXDCYuGSdhAv9aFk-OdNpPbH-2aVK'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    filename = 'CUB_200_2011.tgz'
    archive = os.path.join(extract_root, filename)
    if not os.path.isfile(archive):
        print("Downloading CUB 200 dataset...")
        download_file_from_google_drive(
            file_id, extract_root, filename, tgz_md5)
    checkFile = os.path.join(extract_root, 'attributes.txt')
    if not os.path.isfile(checkFile):
        print("Extracting {} to {}".format(archive, extract_root))
        extract_archive(archive, extract_root)


def download_model(model_path='./model'):
    file_id = '1LxEp7_Hm9RK4oc9UxJ8nu7KUKGFsa4i0'
    tgz_md5 = '90db23115ebec6b2c94499ac1c56ee59'
    filename = 'net.tgz'
    archive = os.path.join(model_path, filename)
    if not os.path.isfile(archive):
        print("Downloading pretrained model...")
        download_file_from_google_drive(
            file_id, model_path, filename, tgz_md5)
    extract_archive(archive, model_path)


def upload_blob(destination_blob_name, fileName):
    """Uploads a file to the bucket."""
    bucketName = "attention_bucket"
    project = 'ai-lab-290600'

    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucketName)
    blob = bucket.blob(destination_blob_name)

    with open(os.path.join(os.curdir, fileName), 'rb') as file:
        blob.upload_from_file(file)

    print(
        "uploaded to {}.".format(
            destination_blob_name
        )
    )


def show_image(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()


def visualize_attn_softmax(I, c, up_factor, nrow):
    # image
    img = I.permute((1, 2, 0)).cpu().numpy()
    # compute the heatmap
    N, C, W, H = c.size()
    a = F.softmax(c.view(N, C, -1), dim=2).view(N, C, W, H)
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor,
                          mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=nrow, normalize=True, scale_each=True)
    attn = attn.permute((1, 2, 0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = 0.6 * img + 0.4 * attn
    return torch.from_numpy(vis).permute(2, 0, 1)


def visualize_attn_sigmoid(I, c, up_factor, nrow):
    # image
    img = I.permute((1, 2, 0)).cpu().numpy()
    # compute the heatmap
    a = torch.sigmoid(c)
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor,
                          mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=nrow, normalize=False)
    attn = attn.permute((1, 2, 0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = 0.6 * img + 0.4 * attn
    return torch.from_numpy(vis).permute(2, 0, 1)


def upload_to_google_storage(bucket_name, project_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client(project=project_name)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )
