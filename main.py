import odak
import loss
import model
from trainer import Trainer
from data_loader import DatasetFromFolder
import utility
import os


def main():
    # setting
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    settings_filename = './settings/sample_zero.txt'
    settings = odak.tools.load_dictionary(settings_filename)
    checkpoint = utility.checkpoint(settings)
    train_depth_dir = settings["train"]["train depth file"]
    train_phase_dir = settings["train"]["train phase file"]
    train_target_dir = settings["train"]["train target file"]
    train_mask_dir = settings["train"]["train mask file"]
    test_depth_dir = settings["test"]["test depth file"]
    test_phase_dir = settings["test"]["test phase file"]
    test_target_dir = settings["test"]["test target file"]
    test_mask_dir = settings["test"]["test mask file"]


    # Dataset
    loader = DatasetFromFolder(settings, train_phase_dir, train_depth_dir, train_target_dir, train_mask_dir)
    loaderTest = DatasetFromFolder(settings, test_phase_dir, test_depth_dir, test_target_dir, test_mask_dir)
    _model = model.Model( settings,checkpoint)
    _loss = loss.Loss(settings, checkpoint)
    t = Trainer(settings, loader, loaderTest, _model, _loss, checkpoint)
    for i in range(settings["optimizer"]["epoch"]):
        t.train()
        t.validation()
        checkpoint.done()


if __name__ == '__main__':
    main()
