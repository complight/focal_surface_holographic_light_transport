import glob
import torch
import os
import odak


def load(fn, settings):
    target = odak.learn.tools.load_image(
        fn,
        normalizeby = 2 ** settings["target"]["color depth"],
        torch_style = True
    )
    return target


class DatasetFromFolder():
    def __init__(self, settings, phase_directory, depth_directory, target_directory, mask_directory, key = '.png'):
        self.key = key
        self.phase_directory = phase_directory
        self.depth_directory = depth_directory
        self.target_directory = target_directory
        self.phase_filenames = sorted(
            glob.glob("/scratch/zheng.chua/RGB_dataset_noise_10/train/phase/" + '/**/*{}'.format("_phase_combined.png"),
                      recursive = True))
        self.depth_filenames = sorted(
            glob.glob("/scratch/zheng.chua/RGB_dataset_noise_10/train/depth/" + '/**/*{}'.format(self.key),
                      recursive = True))
        self.target_filenames = sorted(
            glob.glob("/scratch/zheng.chua/RGB_dataset_noise_10/train/target/" + '/**/*{}'.format("_target.png"),
                      recursive = True))
        self.mask_filenames = sorted(
            glob.glob("/scratch/zheng.chua/RGB_dataset_noise_10/train/mask/" + '/**/*{}'.format(self.key),
                      recursive = True))
        self.settings = settings
        self.number_of_planes = settings["target"]["number of planes"]
        self.volume_depth = settings['target']['volume depth']
        self.location_offset = settings['target']['location offset']


    def depth_calculation(self, image_depth):
        distances = torch.linspace(-self.volume_depth / 2., self.volume_depth / 2.,
                                   self.number_of_planes) + self.location_offset
        y = (distances - torch.min(distances))
        distances = y / torch.max(y)
        target_depth = image_depth * (self.number_of_planes - 1)
        target_depth = torch.round(target_depth, decimals = 0)
        for i in range(self.number_of_planes):
            target_depth = torch.where(target_depth == i, distances[i], target_depth)
        return target_depth


    def __getitem__(self, index):
        image_id = os.path.basename(self.target_filenames[index])[0:4]
        image_depth_id = os.path.basename(self.target_filenames[index])[0:6]
        path = os.path.dirname(self.target_filenames[index])[0:-6]
        phase_image = load("{}phase/{}_phase_combined{}".format(path, image_id, ".png"), self.settings)
        target_image = load(self.target_filenames[index], self.settings)
        depth_image = load("{}depth/{}_random_depth{}".format(path, image_depth_id, ".png"), self.settings)
        mask_image = load("{}mask/{}_random_maks{}".format(path, image_depth_id, ".png"), self.settings)
        distance_map = self.depth_calculation(depth_image)
        return phase_image, distance_map.unsqueeze(0), target_image, mask_image.unsqueeze(0), os.path.basename(
            self.target_filenames[index])[0:6]


    def __len__(self):
        return len(self.target_filenames)
