import fire
import os
import torch
import torchvision as tv
from torch.utils.data.sampler import SubsetRandomSampler
from temperature_scaling import ModelWithTemperature
from my_model import train_Dataset
from my_model import My_model
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms

NUM_CLASS = 10

def my_project(save, batch_size=25):
    """
    Applies temperature scaling to a trained model.

    Takes a pretrained DenseNet-CIFAR100 model, and a validation set
    (parameterized by indices on train set).
    Applies temperature scaling, and saves a temperature scaled version.

    NB: the "save" parameter references a DIRECTORY, not a file.
    In that directory, there should be two files:
    - model.pth (model state dict)
    - valid_indices.pth (a list of indices corresponding to the validation set).

    data (str) - path to directory where data should be loaded from/downloaded
    save (str) - directory with necessary files (see above)
    """
    # Load model state dict
    model_filename = os.path.join(save, 'best_model.pth')
    if not os.path.exists(model_filename):
        raise RuntimeError('Cannot find file %s to load' % model_filename)
    state_dict = torch.load(model_filename)

    # Load validation indices
    valid_indices_filename = os.path.join(save, 'valid_indices.pth')
    if not os.path.exists(valid_indices_filename):
        raise RuntimeError('Cannot find file %s to load' % valid_indices_filename)
    valid_indices = torch.load(valid_indices_filename)

    # Regenerate validation set loader
    train = pd.read_csv('./data_fashion_mnist/train.csv', index_col='index') # len 60000

    My_transform = transforms.Compose([
        transforms.ToTensor(), # default : range [0, 255] -> [0.0, 1.0] 스케일링
    ])
    Valid_data = train_Dataset(train, transform=My_transform)
    Valid_dataloader = DataLoader(dataset=Valid_data,
                              batch_size = batch_size,
                              shuffle=False,
                              sampler=SubsetRandomSampler(valid_indices))

    # Load original model
    orig_model = My_model(NUM_CLASS).cuda()
    orig_model.load_state_dict(state_dict)

    # Now we're going to wrap the model with a decorator that adds temperature scaling
    model = ModelWithTemperature(orig_model)

    # Tune the model temperature, and save the results
    model.set_temperature(Valid_dataloader)
    model_filename = os.path.join(save, 'model_with_temperature.pth')
    torch.save(model.state_dict(), model_filename)
    print('Temperature scaled model sved to %s' % model_filename)
    print('Done!')


if __name__ == '__main__':
    """
    Applies temperature scaling to a trained model.

    Takes a pretrained DenseNet-CIFAR100 model, and a validation set
    (parameterized by indices on train set).
    Applies temperature scaling, and saves a temperature scaled version.

    NB: the "save" parameter references a DIRECTORY, not a file.
    In that directory, there should be two files:
    - model.pth (model state dict)
    - valid_indices.pth (a list of indices corresponding to the validation set).

    --data (str) - path to directory where data should be loaded from/downloaded
    --save (str) - directory with necessary files (see above)
    """
    fire.Fire(my_project)
