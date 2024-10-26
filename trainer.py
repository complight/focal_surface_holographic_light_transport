import utility
import torch
import os
from tqdm import tqdm
import torch.utils.data as data
from decimal import Decimal


class Trainer():
    def __init__(self, setting, loaderTrain, loaderTest, model, loss, ckp):
        self.ckp = ckp
        self.loader_train = loaderTrain
        self.loader_test = loaderTest
        self.model = model
        self.loss = loss
        self.setting = setting
        self.device = setting["general"]["device"]
        self.lr = self.setting["optimizer"]["learning rate"]
        self.optimizer = utility.make_optimizer(self.setting, self.model)
        if os.path.exists(os.path.join(os.path.join('..', 'experiment'), "model", 'model_latest.pt')):
            self.optimizer.load(ckp.dir, epoch = len(ckp.log))
            self.model.load(apath = ckp.dir)
            self.loss.load(apath = ckp.dir)


    def train(self):
        self.loss.step()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(self.optimizer.get_last_epoch() + 1, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        trainLoader = data.DataLoader(dataset = self.loader_train, shuffle = True,
                                      batch_size = self.setting["train"]["batch_size"],
                                      num_workers = self.setting["train"]["num_workers"], pin_memory = True)
        for batch, (phase, depth, target, mask, image_id) in enumerate(trainLoader):
            phase = phase.to(self.device)
            depth = depth.to(self.device)
            target = target.to(self.device)
            mask = mask.to(self.device)
            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()
            recons = self.model( depth,phase)
            loss = self.loss(mask * recons, mask * target, (1 - mask) * recons, (1 - mask) * target, recons, target)
            loss.backward()
            self.optimizer.step()
            timer_model.hold()
            if (batch + 1) % self.setting["general"]["print every"] == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.3f}+{:.3f}s'.format(
                    (batch + 1) * self.setting["train"]["batch_size"],
                    len(self.loader_train),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()
        self.loss.end_log(len(self.loader_train))
        self.optimizer.schedule()


    def validation(self):
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1)
        )
        self.model.eval()
        timer_test = utility.timer()
        testLoader = data.DataLoader(dataset = self.loader_test, batch_size = self.setting["train"]["batch_size"],
                                     num_workers = self.setting["train"]["num_workers"], pin_memory = True)
        save_list = []
        name_list = []
        for batch, (phase, depth, target, mask, image_id) in enumerate(tqdm(testLoader, ncols = 80)):
            phase = phase.to(self.device)
            depth = depth.to(self.device)
            target = target.to(self.device)
            recons = self.model( depth,phase)
            if epoch > 100:
                save_list.append(recons)
                name_list.append(image_id)
            self.ckp.log[epoch - 1] += utility.calc_psnr(recons, target)
        self.ckp.log[epoch - 1] /= len(testLoader)
        best = self.ckp.log.max(0)
        self.ckp.write_log(
            '[{:.0f}]\tPSNR: {:.3f} (Best: {:.3f}  @epoch: {})'.format(
                epoch,
                self.ckp.log[epoch - 1],
                best[0],
                best[1] + 1
            )
        )
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        if epoch > 0:
            if best[1] + 1 == epoch:
                self.ckp.save_results(testLoader, name_list, save_list)
        self.ckp.save(self, epoch, is_best = (best[1] + 1 == epoch))
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh = True
        )
        torch.set_grad_enabled(True)
