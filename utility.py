import torch
import torch.optim.lr_scheduler as lrs
import  os
import  time
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import math


def make_optimizer(setting, target):
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': setting["optimizer"]["learning rate"], 'weight_decay': setting["optimizer"]["weight_decay"]}
    optimizer_class = optim.Adam
    kwargs_optimizer['betas'] = setting["optimizer"]["betas"]
    kwargs_optimizer['eps'] = setting["optimizer"]["epsilon"]
    # scheduler
    milestones = setting["optimizer"]["milestones"]
    kwargs_scheduler = {'milestones': milestones, 'gamma':setting["optimizer"]["gamma"]}
    scheduler_class = lrs.MultiStepLR


    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)


        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)


        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))


        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()


        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')


        def schedule(self):
            self.scheduler.step()


        def get_lr(self):
            return self.scheduler.get_lr()[0]


        def get_last_epoch(self):
            return self.scheduler.last_epoch


    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()


    def tic(self):
        self.t0 = time.time()


    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff


    def hold(self):
        self.acc += self.toc()


    def release(self):
        ret = self.acc
        self.acc = 0
        return ret


    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, setting):
        self.setting = setting
        self.log = torch.Tensor()
        self.dir = os.path.join('..', 'experiment')
        if os.path.exists(self.get_path('psnr_log.pt')):

                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
        try:
            os.makedirs(self.dir)
        except OSError:
            if not os.path.isdir(self.dir):
                raise
        try:
            os.makedirs(self.get_path('model'))
        except OSError:
            if not os.path.isdir(self.get_path('model')):
                raise
        try:
            os.makedirs(self.get_path('results'))
        except OSError:
            if not os.path.isdir(self.get_path('results')):
                raise
        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)


    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)


    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)
        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))


    def add_log(self, log):
        self.log = torch.cat([self.log, log])


    def write_log(self, log, refresh=False):
        print(log)
        self.log_file = open(self.get_path('log.txt'), 'a')
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')


    def done(self):
        self.log_file.close()


    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        # for idx_data, d in enumerate(self.args.data_test):
        label = 'test results'
        fig = plt.figure()
        plt.title(label)
        plt.plot(axis, self.log.numpy())
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig(self.get_path('test.pdf'))
        plt.close(fig)


    def save_results(self, dataset, filename, save_list):
            for v, x in zip(save_list, filename):
                name = self.get_path(
                    'results',
                    '{}'.format(x)
                )
                image = v[0].clamp(0, 1).cpu()
                vutils.save_image(image, '{}.png'.format(name))


def calc_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    # PIXEL_MAX = 255.0
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / torch.sqrt(mse))