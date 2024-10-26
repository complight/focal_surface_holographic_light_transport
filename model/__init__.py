import os
from importlib import import_module
import torch
import torch.nn as nn
import torch.utils.model_zoo


class Model(nn.Module):
    def __init__(self, settings, ckp):
        super(Model, self).__init__()
        print('Making model...')
        self.settings = settings
        self.device = self.settings["general"]["device"]
        module = import_module('model.' + self.settings["general"]["model"])
        self.model = module.make_model( settings).to(self.device)
        print(self.model, ckp.log_file)


    # def forward(self, x, idx_scale):
    def forward(self, x, y):
        return self.model(x, y)


    def save(self, apath, epoch, is_best = False):
        save_dirs = [os.path.join(apath, 'model_latest.pt'.format(epoch))]
        if is_best:
            save_dirs.append(os.path.join(apath, 'model_{}_best.pt'.format(epoch)))
        for s in save_dirs:
            torch.save(self.model.state_dict(), s)


    def load(self, apath):
        load_from = torch.load(os.path.join(apath, 'model', 'model_latest.pt'))
        self.model.load_state_dict(load_from, strict = False)
