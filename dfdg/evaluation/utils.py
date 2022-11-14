''' 
@copyright Copyright (c) Siemens AG, 2022
@author Haokun Chen <haokun.chen@siemens.com>
SPDX-License-Identifier: Apache-2.0
'''

import torch


def evaluate_onebatch(model, batch, y):
    model.eval()
    ans = 0
    tot = 0
    with torch.no_grad():
        o = model(batch).argmax(1)
        tmp_out = y.cpu() == o.cpu()
        ans += int(tmp_out.sum())
        tot += batch.shape[0]
    return ans / tot


def evaluate_oneset(model, loader):
    model.eval()
    ans = 0
    tot = 0
    with torch.no_grad():
        for (imgs, labels) in loader:
            o = model(imgs).argmax(1)
            tmp_out = labels == o
            ans += int(tmp_out.sum())
            tot += imgs.shape[0]
    return ans / tot
