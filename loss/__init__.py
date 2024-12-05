import os
from importlib import import_module
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
matplotlib.use('Agg')


class Loss(nn.modules.loss._Loss):
    def __init__(self, settings, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.setting=settings
        self.loss = []
        self.loss_module = nn.ModuleList()
        self.log = torch.Tensor()
        for loss in self.setting["optimizer"]["loss"].split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('PER') >= 0:
                module = import_module('loss.perceptualLoss')
                loss_function = getattr(module, 'PER')(
                )
            elif loss_type.find('FI') >= 0:
                loss_function = nn.MSELoss()
            elif loss_type.find('FX') >= 0:
                loss_function = nn.MSELoss()

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])
                
        self.log = torch.Tensor()
        device = self.setting["general"]["device"]
        self.loss_module.to(device)

        # if os.path.exists(os.path.join(os.path.join('..', 'experiment'), "model", 'model_latest.pt')):
        #     self.load(apath=ckp.dir)


    def forward(self, recons_focus, target_focus, recons_blur, target_bulr,recons,target):

        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None and l['type'] != 'FI' and l['type'] != 'FP'and l['type'] != 'FX'and l['type'] != 'L1':
                loss = l['function'](recons_focus, target_focus)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'FI':
                loss = l['function'](recons_blur, target_bulr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'FX':
                loss = l['function'](recons, target)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'PER':
                loss = l['function'](recons,target)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'L1':
                loss = l['function'](recons,target)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()
        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.6f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
           return self.loss_module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath):
        kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()
