import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported.
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
                and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


# def create_dataloader(opt, world_size, rank):
#     dataset = find_dataset_using_name(opt.dataset_mode)
#     instance = dataset()
#     instance.initialize(opt)
#     print("dataset [%s] of size %d was created" %
#           (type(instance).__name__, len(instance)))
#
#     if opt.isTrain:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(instance, num_replicas=world_size, rank=rank)
#         dataloader = torch.utils.data.DataLoader(instance,
#                                                  batch_size=opt.batchSize,
#                                                  sampler=train_sampler,
#                                                  shuffle=False,
#                                                  num_workers=int(opt.nThreads),
#                                                  drop_last=opt.isTrain)
#     else:
#         dataloader = torch.utils.data.DataLoader(instance,
#                                                  batch_size=opt.batchSize,
#                                                  shuffle=False,
#                                                  num_workers=int(opt.nThreads),
#                                                  drop_last=opt.isTrain)
#     return dataloader

from torch.utils.data.distributed import DistributedSampler

def create_dataloader(opt):
    dataset = find_dataset_using_name(opt.dataset_mode)
    instance = dataset()
    instance.initialize(opt)
    print("dataset [%s] of size %d was created" %
          (type(instance).__name__, len(instance)))

    if opt.isTrain:
        if opt.multi_thread_gpu:
            dataloader = torch.utils.data.DataLoader(
                instance,
                batch_size=opt.batchSize,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.nThreads),
                drop_last=opt.isTrain,
                sampler=DistributedSampler(instance)
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                instance,
                batch_size=opt.batchSize,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.nThreads),
                drop_last=opt.isTrain
            )
    else:
        dataloader = torch.utils.data.DataLoader(
            instance,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            drop_last=opt.isTrain
        )
    return dataloader
