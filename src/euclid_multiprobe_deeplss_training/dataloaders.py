# Copyright (C) 2026 FHNW, Institute for Data Science

"""
Created February 2026
Author: Tomasz Kacprzak
"""

import warnings
import webdataset
import torch
from euclid_multiprobe_deeplss_training.utils import logger
import glob


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class OntheflyPipeline():
    """
    Sets up a dataset for the onthefly cosmologies.
    """
    def __init__(self, webds_pattern, batch_size, physics_model, smoother=None, downsampler=None, device="cuda", validation=False, num_workers=1, seed=None):

        self.physics_model = physics_model
        self.smoother = smoother
        self.downsampler = downsampler
        self.device = device
        self.batch_size = batch_size
        self.seed = seed
        
        # get webdataset dataset
        list_files = sorted(glob.glob(webds_pattern))
        LOGGER.info(f"found {len(list_files)} files")

        # split the files into training and validation
        validation_fraction = 0.2
        if validation:
            list_files = list_files[:int(len(list_files) * validation_fraction)]
            LOGGER.info(f"using {len(list_files)} files for validation")
        else:
            list_files = list_files[int(len(list_files) * validation_fraction):]
            LOGGER.info(f"using {len(list_files)} files for training")

        dataset = (
            webdataset.WebDataset(
                list_files, 
                shardshuffle=1000,
                seed = self.seed,
                nodesplitter=webdataset.split_by_node,
                workersplitter=webdataset.split_by_worker,
            )
            .decode()
            .to_tuple(
                "maps_float32.pth",
                "vec_int32.pth",
                "vec_float32.pth",
            )
            # .batched(self.batch_size, partial=False)
        )
        
        # get torch DataLoader
        # self.loader = DataLoader(dataset, **kwargs)
        self.loader = webdataset.WebLoader(
                        dataset,
                        pin_memory=True,
                        # batch_size=None,
                        num_workers=num_workers, 
                        prefetch_factor=1, 
                        persistent_workers=True,
                        batch_size=self.batch_size,
                        drop_last=True,
                      )


        # test and get the number of pixels
        for inputs, indices, targets in self.__iter__():
            
            self.num_pixels = inputs.shape[1]
            self.num_channels = inputs.shape[2]
            self.num_targets = targets.shape[1]
            break

        LOGGER.info(f"Created OntheflyPipeline with num_pixels={self.num_pixels}, num_channels={self.num_channels}, num_targets={self.num_targets}")


    def __iter__(self):

        batch_count = 0
        
        for batch in self.loader:

            with torch.no_grad():

                # move to GPU
                batch = tuple(tensor.to(self.device, non_blocking=True) for tensor in batch)

                # Remove the unused fields for now
                maps, vec_int, vec_float = batch

                # initial downsampling 
                if self.downsampler is not None:
                    maps, vec_int, vec_float = batch
                    maps = self.downsampler(maps)
                    batch = (maps, vec_int, vec_float)

                # add physics augmentations
                inputs, targets = self.physics_model(batch)
                
                # final smoothing
                if self.smoother is not None:
                    inputs = self.smoother(inputs)

            LOGGER.debug(f"Batch {batch_count:>6d} physics shape = {inputs.shape}")

            batch_count += 1

            yield inputs, targets, vec_int


