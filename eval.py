import argparse
import csv
import torch
from model.dataset import EvalImageGroundTruthFolder
from model.evaluate import Eval_thread
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--datasets_path', type=str, default='./datasets/test', help='path to datasets, default = ./datasets/test')
parser.add_argument('--pred_path', type=str, default='./results/CPD', help='path to predictions, default = ./results')
arg = parser.parse_args()

dataset = EvalImageGroundTruthFolder(arg.datasets_path, arg.pred_path, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

eval = Eval_thread(arg.datasets_path, loader, method='MAE', dataset='Test', output_dir='./')
results = eval.run()

for d in results:
    print(d)
    for m, r in results[d].items():
        print(m, r)
    filename = '{}/{}.csv'.format(arg.pred_path, d)
    with open(filename, 'w') as outfile:
        writer = csv.writer(outfile)
        for key, val in results[d].items():
            writer.writerow([key, val])
    print()
