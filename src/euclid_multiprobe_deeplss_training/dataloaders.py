# Copyright (C) 2026 FHNW, Institute for Data Science

"""
Created February 2026
Author: Tomasz Kacprzak
"""

import glob
import warnings

import torch
import webdataset

from euclid_multiprobe_deeplss_training.utils import logger

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class OntheflyPipeline:
    """
    Sets up a dataset for the onthefly cosmologies.

    A smoother must accept dense ``(batch, pixels, channels)`` inputs and
    weights. Channel lists are stacked at that boundary and restored afterwards.
    Weights come from ``physics_model.get_weight_maps(inputs)`` after physics.
    """
    def __init__(
        self, webds_pattern, batch_size, physics_model, smoother=None, downsampler=None,
        device="cuda", validation=False, num_workers=1, seed=None, infer_shapes=True,
    ):

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
        worker_options = {"prefetch_factor": 1, "persistent_workers": True} if num_workers else {}
        self.loader = webdataset.WebLoader(
                        dataset,
                        pin_memory=torch.device(device).type == "cuda",
                        # batch_size=None,
                        num_workers=num_workers, 
                        **worker_options,
                        batch_size=self.batch_size,
                        drop_last=True,
                      )


        # Training needs metadata before constructing its encoder. One-batch
        # diagnostics can skip this extra physics evaluation and allocation.
        if infer_shapes:
            for inputs, targets, _indices in self.__iter__():
                if isinstance(inputs, torch.Tensor):
                    self.num_channels = inputs.shape[-1]
                    self.num_pixels = inputs.shape[1]
                else:
                    self.num_channels = len(inputs)
                    self.num_pixels = inputs[0].shape[1]
                self.num_targets = targets.shape[1]
                break

            LOGGER.info(f"Created OntheflyPipeline with num_pixels={self.num_pixels}, num_channels={self.num_channels}, num_targets={self.num_targets}")


    def __iter__(self):

        batch_count = 0
        
        for batch in self.loader:

            with torch.no_grad():

                # move to GPU
                batch = tuple(tensor.to(self.device, non_blocking=True) for tensor in batch)

                if self.downsampler is not None:
                    batch = (self.downsampler(batch[0]), batch[1], batch[2])

                # add physics augmentations
                inputs, targets = self.physics_model(batch)

                for i, inp in enumerate(inputs):
                    LOGGER.debug(f"OntheflyPipeline: Batch {batch_count:>6d} inputs[{i}].shape={inp.shape} dtype={inp.dtype}")
                
                # Use the statistical weights of the physics channels before smoothing.
                if self.smoother is not None:
                    weights = self.physics_model.get_weight_maps(inputs)
                    if isinstance(inputs, (list, tuple)):
                        # The scale-cut layer uses a dense channel axis. Restore
                        # each channel's dtype afterwards, including scalar maps.
                        smoothed = self.smoother(torch.stack(inputs, dim=-1), torch.stack(weights, dim=-1))
                        inputs = [
                            (channel if original.is_complex() else channel.real).to(original.dtype)
                            for channel, original in zip(smoothed.unbind(dim=-1), inputs, strict=True)
                        ]
                    else:
                        inputs = self.smoother(inputs, weights)

            LOGGER.debug(f"Batch {batch_count:>6d} physics len {len(inputs)}")

            batch_count += 1

            yield inputs, targets, batch[1]


