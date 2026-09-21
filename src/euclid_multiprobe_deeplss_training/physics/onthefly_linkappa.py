# Copyright (C) 2026 FHNW, Institute for Data Science

"""
Created June 2026
Author: Tomasz Kacprzak
"""
# from tkinter import E
import warnings
import numpy as np
import healpy as hp
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader


from euclid_multiprobe_deeplss_training.utils import logger
from euclid_multiprobe_deeplss_training.dataloaders import OntheflyPipeline

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)

def sample_astro_parameters_latin_hypercube(astro_params, seed, n_examples, astro_priors):

    import qmc

    latin_sampler = qmc.LatinHypercube(d=len(astro_params), seed=seed)
    unscaled_samples = latin_sampler.random(n_examples)
    astro_samples = qmc.scale(unscaled_samples, l_bounds=astro_priors[:, 0], u_bounds=astro_priors[:, 1])
    return astro_samples.astype(np.float32)


def galaxy_density_to_count(ng_bar, dg, bg):
    """
    Convert density contrast to mean number counts.
    """

    # get mean number of galaxies per pixel
    ng = (1 + bg * dg) * ng_bar
    # remove negative values
    ng_clip = torch.clamp(ng, min=0, max=1e6)
    # adjust so that the total is conserved
    ng_lambda = ng_clip * torch.sum(ng) / torch.sum(ng_clip)
    # draw Poisson noise
    ng = torch.poisson(ng_lambda)
    return ng

class OntheflyPhysicsModelLinkappa(nn.Module):

    def __init__(self, conf, scalers=False, seed=424344, num_samples_prior=1_000_000, device=None, nside=None, shape_noise_std=0.3, **kwargs):

        super().__init__()

        self.model_name = "linkappa_dmo"
        self.conf = conf
        self.seed = seed
        self.device = device
        self.nside = nside
        self.set_params()
        self.onthefly_samples = self.get_onthefly_params(num_samples_prior)
        self.shape_noise_std = shape_noise_std
        self.num_gal_wl = torch.from_numpy(np.array(self.conf["survey"]["WL"]["n_gal"])).to(self.device)
        self.num_gal_gc = torch.from_numpy(np.array(self.conf["survey"]["GC"]["n_gal"])).to(self.device)
        self.pixel_area = hp.nside2pixarea(self.nside, degrees=True)
        self.sample_uniform_lo = torch.tensor(0., device=self.device, dtype=torch.float32)
        self.sample_uniform_hi = torch.tensor(2 * math.pi, device=self.device, dtype=torch.float32)
        self.num_targets = len(self.params)
        self.num_channels = 18
        self.channel_spins = torch.from_numpy(np.array([0]*18)).to(self.device)
        # self.param_names = ['Om', 's8', 'Ob', 'H0', 'ns', 'w0', 'bary_Mc', 'bary_nu', 'Aia', 'n_Aia', 'bg1', 'bg2', 'bg3', 'bg4', 'bg5', 'bg6', 'bsc1', 'bsc2', 'bsc3', 'bsc4', 'bsc5', 'bsc6']
        self.scalers = scalers
        if self.scalers:
            self.set_scalers()
        LOGGER.info(f"Created physics model Linear, num_channels={self.num_channels}, num_targets={self.num_targets}, apply_scalers={self.scalers}, shape_noise_std={self.shape_noise_std}")
                

    def set_scalers(self):

        shift_channels = 0.
        scale_channels = torch.tensor([1.]*6 + [1./100000.]*6 + [1./10000.]*6, device=self.device)

        list_min = torch.tensor([self.priors[p][0] for p in self.params], device=self.device, dtype=torch.float32)
        list_max = torch.tensor([self.priors[p][1] for p in self.params], device=self.device, dtype=torch.float32)
        shift_targets = -list_min
        scale_targets = 1./(list_max-list_min)

        self.targets_shift = shift_targets.reshape(1, -1).to(self.device)
        self.targets_scale = scale_targets.reshape(1, -1).to(self.device)
        self.channel_shift = shift_channels #.reshape(1, 1, -1).to(self.device)
        self.channel_scale = scale_channels.reshape(1, 1, -1).to(self.device)

    def set_params(self):

        self.params = ['Om', 's8', 'Ob', 'H0', 'ns', 'w0', 'bary_Mc', 'bary_nu', 'bg1', 'bg2', 'bg3', 'bg4', 'bg5', 'bg6', 'Aia1', 'Aia2', 'Aia3', 'Aia4', 'Aia5', 'Aia6', 'bsc1', 'bsc2', 'bsc3', 'bsc4', 'bsc5', 'bsc6']

        # from Table 1 in https://arxiv.org/pdf/2201.07771
        self.priors = {
            'Om':   [0.1, 0.5],
            's8':   [0.4, 1.4],
            'S8':   [0.23094010767585035, 1.8073922282301278],
            'H0':   [64, 82],
            'Ob':   [0.03, 0.06],
            'ns':   [0.87, 1.07],
            'w0':   [-2, -0.333],
            'bary_Mc': [12.0, 15.0], # log10(bary_Mc)
            'bary_nu': [-2, 2.0],
            'bg1':  [0.8, 3.0],
            'bg2':  [0.8, 3.0],
            'bg3':  [0.8, 3.0],
            'bg4':  [0.8, 3.0],
            'bg5':  [0.8, 3.0],
            'bg6':  [0.8, 3.0],
            'Aia1': [-0.5, 1.5],
            'Aia2': [-0.5, 1.5],
            'Aia3': [-0.5, 1.5],
            'Aia4': [-0.5, 1.5],
            'Aia5': [-0.5, 1.5],
            'Aia6': [-0.5, 1.5],
            'bsc1': [1.0, 2.0],
            'bsc2': [1.0, 2.0],
            'bsc3': [1.0, 2.0],
            'bsc4': [1.0, 2.0],
            'bsc5': [1.0, 2.0],
            'bsc6': [1.0, 2.0],
        }

        for param in self.params:
            assert param in self.priors, f"Parameter {param} not found in priors"

        LOGGER.info(f"Model {self.model_name} parameters: {self.params}")


    def get_onthefly_params(self, num_samples_prior):

        self.onthefly_params = ['bg1', 'bg2', 'bg3', 'bg4', 'bg5', 'bg6', 'Aia1', 'Aia2', 'Aia3', 'Aia4', 'Aia5', 'Aia6', 'bsc1', 'bsc2', 'bsc3', 'bsc4', 'bsc5', 'bsc6']
        self.onthefly_priors = {key:self.priors[key] for key in self.onthefly_params}
        onthefly_priors_bounds = np.array([self.onthefly_priors[key] for key in self.onthefly_params])
        
        # Pre-generate samples from latin hypercube
        self.onthefly_samples = sample_astro_parameters_latin_hypercube(
            self.onthefly_params, 
            seed=self.conf['master_seed']+self.seed, 
            n_examples=num_samples_prior, 
            astro_priors=onthefly_priors_bounds)

        LOGGER.info(f"Sampled onthefly parameters: {self.onthefly_samples.shape}")
        for i, param_name in enumerate(self.onthefly_params):
            p_ = self.onthefly_samples[:, i]
            LOGGER.info(f"   {param_name:>20s}  min={np.min(p_): 8.3f} max={np.max(p_): 8.3f} mean={np.mean(p_): 8.3f}")

        return torch.from_numpy(self.onthefly_samples).to(self.device)

    def sample_onthefly_parameters(self, batch_size):

        j = torch.randint(0, self.onthefly_samples.shape[0], (batch_size,), device=self.device)
        return self.onthefly_samples[j].squeeze().to(self.device)

    def forward_physics(self, example):

        #
        # Preliminaries
        #

        # unpack the example
        maps, vec_int, hard_params = example

        # this was created in the postprocessing pipeline, check if we are using the right data
        assert hard_params.shape[1] == 8, f"Expected 8 parameters, got {hard_params.shape[1]}"
        assert maps.shape[2] == 6, f"Expected 6 redshift bins per map, got {maps.shape[2]}"
        assert maps.shape[3] == 4, f"Expected 4 map types, got {maps.shape[3]}"

        # split into lensing, IA, and galaxy clustering maps
        kg, ia, ds, dg = maps.unbind(dim=-1)

        #  we will use kg_tot and as lensing container
        kg_tot = kg

        # get onthefly parameters
        onthefly_params = self.sample_onthefly_parameters(hard_params.shape[0])
        onthefly_params = torch.atleast_2d(onthefly_params)
        targets = torch.cat([hard_params, onthefly_params], dim=1)

        # convert bary_Mc to log10(bary_Mc)
        targets[:, 6] = torch.log10(targets[:, 6])

        #
        # Linear bias map for galaxy clustering (lenses)
        #

        # get mean number of galaxies per pixel
        ng_bar = self.num_gal_gc * self.pixel_area
        ids_bg = [8, 9, 10, 11, 12, 13] # bg1, bg2, bg3, bg4, bg5, bg6
        tomo_bg = targets[:, ids_bg].unsqueeze(1) # shape (batch_size, 1, n_GC_bins)
        assert dg.shape[-1] == tomo_bg.shape[-1], "The number of bias parameters must match the number of tomographic bins"
        # convert density contrast to mean number counts
        ng = galaxy_density_to_count(ng_bar, dg, tomo_bg)
        LOGGER.debug(f'drawn poisson galaxy clustering map ng={ng.shape} min={ng.min():>10.3f} max={ng.max():>10.3f} mean={ng.mean():>10.3f}')

        #
        # Linear bias map for galaxy clustering (sources)
        #

        # get mean number of galaxies per pixel
        ng_bar = self.num_gal_wl * self.pixel_area
        ids_bsc = [20, 21, 22, 23, 24, 25] # bsc1, bsc2, bsc3, bsc4, bsc5, bsc6
        tomo_bsc = targets[:, ids_bsc].unsqueeze(1) # shape (batch_size, 1, n_WL_bins)
        ns = galaxy_density_to_count(ng_bar, ds, tomo_bsc)
        LOGGER.debug(f'drawn poisson galaxy clustering map ns={ns.shape} min={ns.min():>10.3f} max={ns.max():>10.3f} mean={ns.mean():>10.3f}')

        #
        # Add lensing shape noise to kappa
        #
        kg_noise_temp = torch.empty(kg.shape, dtype=torch.float32, device=self.device)
        nn.init.trunc_normal_(kg_noise_temp, mean=0.0, std=self.shape_noise_std/math.sqrt(2.), a=-1.0, b=1.0)
        kg_noise_temp.div_(torch.sqrt(ns))
        kg_noise_temp[ns==0] = 0
        kg.add_(kg_noise_temp)
        

        #
        # Add linear intrinsic alignment to kappa
        #
        ids_ia = [14, 15, 16, 17, 18, 19] # Aia1, Aia2, Aia3, Aia4, Aia5, Aia6
        tomo_Aia = targets[:, ids_ia].unsqueeze(1) # shape (batch_size, 1, num_bins)
        ia.mul_(tomo_Aia)
        kg_tot.add_(ia)
                           

        # 
        # Wrap up
        # 

        # final report
        LOGGER.debug(f'kg_tot.shape={kg_tot.shape}, kg_tot.dtype={kg_tot.dtype}')
        LOGGER.debug(f'ns.shape={ns.shape}, ns.dtype={ns.dtype}')
        LOGGER.debug(f'ng.shape={ng.shape}, ng.dtype={ng.dtype}')

        # Stack probes as channels
        inputs = torch.cat([kg_tot, ns, ng], dim=-1)
        LOGGER.debug(f'inputs shape={inputs.shape} dtype={inputs.dtype}')

        return inputs, targets

    def apply_scalers(self, inputs, targets):

        inputs = (inputs + self.channel_shift) * self.channel_scale
        targets = (targets + self.targets_shift) * self.targets_scale

        return inputs, targets



    
    def forward(self, example):
        
        inputs, targets = self.forward_physics(example)

        if self.scalers:
            inputs, targets = self.apply_scalers(inputs, targets)

        return inputs, targets

    def get_weight_maps(self, inputs):

        weights = torch.ones_like(inputs, device=self.device, dtype=torch.float32)
        weights[..., :6] = inputs[..., 6:12]  # lensing maps weights are set to source clustering pixel count
        return weights



    def unstack_batch_channels(self, inputs):
        """
        Unstack the channels of the input tensor into a list of maps.
        Inputs: 
            inputs: tensor of shape (batch_size, num_pixels, num_channels)
        Outputs:
            maps: list of tensors, each tensor is of shape (batch_size, num_pixels).
        """

        # all maps are scalar, simple unbind is enough
        channel_maps = inputs.unbind(dim=-1)

        return channel_maps

    def preprocess_for_correlations(self, inputs):

        # first 6 channels are shear maps, subtract the mean
        shear_maps = inputs[..., :6]
        select = shear_maps != 0
        shear_maps_mean = shear_maps[select].mean(dim=-1)
        shear_maps = shear_maps - shear_maps_mean
        
        # next 12 channels are clustering maps, convert to density contrast
        source_maps = inputs[..., 6:18]
        select = source_maps != 0
        source_maps_mean = source_maps[select].mean(dim=-1)
        source_maps = source_maps / source_maps_mean - 1

        return torch.cat([shear_maps, source_maps], dim=-1)
        
