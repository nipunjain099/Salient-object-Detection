import argparse
import csv
import torch
from model.dataset import ImageGroundTruthFolder
from model.models import CPD, CPD_A, CPD_darknet19, CPD_darknet19_A, CPD_darknet_A
from torchvision import transforms
from os import walk
import os

parser = argparse.ArgumentParser()
parser.add_argument('--datasets_path', type=str, default='./datasets/test', help='path to datasets, default = ./datasets/test')
parser.add_argument('--imgres', type=int, default=352, help='image input and output resolution, default = 352')
args = parser.parse_args()

transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

gt_transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor()])

f = []
for (dirpath, dirnames, filenames) in walk('pruned/'):
    f.extend(filenames)
    break

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model = CPD().to(device)

dataset = ImageGroundTruthFolder(args.datasets_path, transform=transform, target_transform=gt_transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
datasets = [d.name for d in os.scandir(args.datasets_path) if d.is_dir()]
models = []

for pth in f:

    state_dict = torch.load('pruned/' + pth, map_location=torch.device(device))
    print('Loaded:', pth)
    mae = {ds_name: [] for ds_name in datasets}

    for idx, pack in enumerate(loader):
        img, gt, dataset, img_name, img_res = pack
        if idx % 1000 == 0:
            print('{} - {}'.format(dataset[0], img_name[0]))
        img = img.to(device)
        _, pred = model(img)

        mae[dataset[0]].append(torch.abs(pred - gt).mean().cpu().detach().numpy())

    for d in datasets:
        model.append([pth, np.mean(mae[d])])
