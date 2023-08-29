import os.path
import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as F
import torch.utils.data as data
#from data.base_dataset import BaseDataset, get_transform
#from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
#from util import util

# from omegaconf import OmegaConf
# opt = OmegaConf.load(os.path.join(os.path.dirname(
#     os.path.abspath(__file__)), "config/base.yaml"))["datasets"]



def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class IhdDataset(data.Dataset):
    """A template dataset class for you to implement custom datasets."""
#     @staticmethod
#     def modify_commandline_options(parser, is_train):
#         """Add new dataset-specific options, and rewrite default values for existing options.

#         Parameters:
#             parser          -- original option parser
#             is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

#         Returns:
#             the modified parser.
#         """
#         parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
#         parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0)  # specify dataset-specific default values
#         return parser

    def __init__(self, opt, is_train=True, subset='IhdDataset'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        super(IhdDataset, self).__init__()
        self.opt = opt
        self.image_paths, self.mask_paths, self.gt_paths = [], [], []
        self.isTrain = is_train
        self.image_size  = opt.crop_size
        
        if subset == 'IhdDataset':
            self.datasets = [
                    'HCOCO',
                    'HFlickr',
                    'Hday2night',
                    'HAdobe5k',
                ]
        else:
              self.datasets = [subset]
        
        if self.isTrain==True:
            #self.real_ext='.jpg'
            print('loading training file')
            stage = 'train'
        else:
            print('loading test file')
            stage = 'test'
        dataset_root = self.opt.iharmony
        self.paths = [os.path.join(dataset_root, dataset)
                     for dataset in self.datasets]
 
        for dataset, path in zip(self.datasets, self.paths):
            file = os.path.join(path, f'{dataset}_{stage}.txt')
            with open(file,'r') as f:
                for line in f.readlines():
                    line = 'composite_images/' + line.rstrip()
                    name_parts = line.split('_')
                    mask_path = line.replace('composite_images', 'masks')
                    mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
                    gt_path = line.replace('composite_images', 'real_images')
                    gt_path = gt_path.replace('_'+name_parts[-2]+'_'+name_parts[-1], '.jpg')
                    self.image_paths.append(os.path.join(dataset_root, path, line))
                    self.mask_paths.append(os.path.join(dataset_root, path, mask_path))
                    self.gt_paths.append(os.path.join(dataset_root, path, gt_path))
                    
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        transform_list = [
            transforms.ToTensor(),
            #transforms.Normalize((0, 0, 0), (1, 1, 1))
            #transforms.Lambda(lambda x: x /255.)
        ]
        self.transforms = transforms.Compose(transform_list)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        # comp = util.retry_load_images(path)
        # mask = util.retry_load_images(mask_path)
        # real = util.retry_load_images(target_path)
        
        comp = Image.open(self.image_paths[index]).convert('RGB')
        real = Image.open(self.gt_paths[index]).convert('RGB')
        mask = Image.open(self.mask_paths[index]).convert('1')

        if np.random.rand() > 0.5 and self.isTrain:
            comp, mask, real = tf.hflip(comp), tf.hflip(mask), tf.hflip(real)

        comp = tf.resize(comp, [self.image_size, self.image_size])
        mask = tf.resize(mask, [self.image_size, self.image_size])
        real = tf.resize(real, [self.image_size,self.image_size])
        
            
        comp = self.transforms(comp)
        mask = tf.to_tensor(mask)
        # mask = 1-mask
        real = self.transforms(real)

        # comp = real
        # mask = torch.zeros_like(mask)
        # inputs=torch.cat([real,mask],0)
        inputs=torch.cat([comp,mask],0)
        
        return {'inputs': inputs, 'comp': comp, 'real': real,'img_path': self.image_paths[index],'mask':mask}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
    
    
class FFHQH(data.Dataset):
    def __init__(self, opt, is_train=True, subset='FFHQH'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        super(FFHQH, self).__init__()
        self.opt = opt
        self.image_paths, self.mask_paths, self.gt_paths = [], [], []
        self.isTrain = is_train
        self.image_size  = opt.crop_size
        
        if self.isTrain==True:
            print('loading training file')
            stage = 'train'
        else:
            print('loading test file')
            stage = 'test'
            
        input_list, alpha_list = [], []
 
        file = os.path.join(self.opt.dataset_root, f'{stage}.txt')
        with open(file,'r') as f:
            for line in f.readlines():
                line = line.rstrip()

                input_path = os.path.join(self.opt.dataset_root, 'comp', line)
                mask_path = os.path.join(self.opt.dataset_root, 'alpha', line)
                gt_path = os.path.join(self.opt.ffhq, line)
                
                self.image_paths.append(input_path)
                self.mask_paths.append(mask_path)
                self.gt_paths.append(gt_path)
        
        self.image_paths = self.image_paths
        self.mask_paths = self.mask_paths
        self.gt_paths = self.gt_paths
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        # comp = util.retry_load_images(path)
        # mask = util.retry_load_images(mask_path)
        # real = util.retry_load_images(target_path)
        
        comp = Image.open(self.image_paths[index]).convert('RGB')
        real = Image.open(self.gt_paths[index]).convert('RGB')
        mask = Image.open(self.mask_paths[index])

        comp = tf.resize(comp, [self.image_size, self.image_size])
        mask = tf.resize(mask, [self.image_size, self.image_size])
        real = tf.resize(real, [self.image_size, self.image_size])
        
            
        comp = tf.to_tensor(comp)
        mask = tf.to_tensor(mask)
        # mask = 1-mask
        real = tf.to_tensor(real)

        # comp = real
        # mask = torch.zeros_like(mask)
        # inputs=torch.cat([real,mask],0)
        inputs=torch.cat([comp, mask],0)
        
        return {'inputs': inputs, 'comp': comp, 'real': real, 'mask': mask, 'img_path': self.image_paths[index]}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)