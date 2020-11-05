import torch
import torchvision.transforms as transforms
import torch.utils.tensorboard  as tensorboard
import torch.nn.utils.prune as prune

import os, argparse
from datetime import datetime

from model.models import CPD, CPD_A, CPD_darknet19, CPD_darknet19_A, CPD_darknet_A

device = torch.device('cpu')
state_dict = torch.load('CPD-O.pth', map_location=torch.device(device))
model = CPD().to(device)
save_path = 'pruned/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:

    model.load_state_dict(state_dict)
    parameters_to_prune = []
    for name, module in model.named_modules():
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append([module, 'weight'])

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=i,
    )


    nelements = 0
    weight_sum = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            nelements += module.weight.nelement()
            weight_sum += torch.sum(module.weight == 0)
        #     print(
        #     "Sparsity in {}: {:.2f}%".format(name,
        #         100. * float(torch.sum(module.weight == 0))
        #         / float(module.weight.nelement())
        #     )
        # )

    gs = 100 *  weight_sum // nelements
    print('Global sparsity: {:.2f}%'.format(gs))
    for para in parameters_to_prune:
        print(para)
        prune.remove(para[0], para[1])
    torch.save(model.state_dict(), 'pruned/{}_{:.0f}.pth.'.format(model.name, gs))
