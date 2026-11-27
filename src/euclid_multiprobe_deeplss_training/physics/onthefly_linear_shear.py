"""Placeholder for the linear on-the-fly physics model."""

from torch import nn

# Copyright (C) 2026 FHNW, Institute for Data Science

"""
Created June 2026
Author: Tomasz Kacprzak
"""
# from tkinter import E
import os
import warnings
import numpy as np
import healpy as hp
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader


from euclid_multiprobe_deeplss_training.utils import logger
from euclid_multiprobe_deeplss_training.utils.distances import smoothing_sigma
from euclid_multiprobe_deeplss_training.dataloaders import OntheflyPipeline

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)

def sample_astro_parameters_latin_hypercube(astro_params, seed, n_examples, astro_priors):

    from scipy.stats import qmc

    latin_sampler = qmc.LatinHypercube(d=len(astro_params), seed=seed)
    unscaled_samples = latin_sampler.random(n_examples)
    astro_samples = qmc.scale(unscaled_samples, l_bounds=astro_priors[:, 0], u_bounds=astro_priors[:, 1])
    return astro_samples.astype(np.float32)


def galaxy_density_to_count(ng_bar, dg, bg):
    """
    Convert density contrast to mean number counts.
    """

    # get mean number of galaxies per pixel
    ng = torch.where(dg != 0, (1 + bg * dg) * ng_bar, 0)
    # remove negative values
    ng_clip = torch.clamp(ng, min=0, max=1e6)
    # adjust so that the total is conserved
    ng_lambda = ng_clip * torch.sum(ng, axis=1, keepdim=True) / torch.sum(ng_clip, axis=1, keepdim=True)
    # draw Poisson noise
    ng = torch.where(ng_lambda != 0, torch.poisson(ng_lambda), 0)
    return ng

class OntheflyPhysicsModelLinearShear(nn.Module):

    def __init__(self, conf, 
                 scalers=False, 
                 seed=424344, 
                 num_samples_prior=1_000_000, 
                 device=None, 
                 nside=None, 
                 shape_noise_std=0.3, 
                 randomize_astro_params=True, 
                 indices_pixels=None, 
                 small_scale_selection=None, 
                 large_scale_selection=None, 
                 **kwargs):

        super().__init__()

        self.model_name = "linear_shear_dmb"
        self.conf = conf
        self.seed = seed
        self.device = device

        # basics and dimensions
        self.nside = nside
        self.shape_noise_std = shape_noise_std
        self.indices_pixels = indices_pixels
        self.pixel_area = hp.nside2pixarea(self.nside, degrees=True)
        self.num_channels = 18
        self.channel_spins = torch.from_numpy(np.array([2]*6 + [0]*12)).to(self.device)

        # samples properties
        self.num_gal_wl = torch.from_numpy(np.array(self.conf["survey"]["WL"]["n_gal"])).to(self.device)
        self.num_gal_gc = torch.from_numpy(np.array(self.conf["survey"]["GC"]["n_gal"])).to(self.device)
        self.z_med_wl = np.array(self.conf["survey"]["WL"]["z_med"])
        self.z_med_gc = np.array(self.conf["survey"]["GC"]["z_med"])

        # priors and sampling
        self.set_params()
        self.num_targets = len(self.params)
        self.onthefly_samples = self.get_onthefly_params(num_samples_prior)
        self.sample_uniform_lo = torch.tensor(0., device=self.device, dtype=torch.float32)
        self.sample_uniform_hi = torch.tensor(2 * math.pi, device=self.device, dtype=torch.float32)
        self.randomize_astro_params = randomize_astro_params

        # scalers
        self.scalers = scalers
        if self.scalers:
            self.set_scalers()

        # scale selection
        self.small_scale_selection = small_scale_selection
        self.large_scale_selection = large_scale_selection
        self.smoothing = self.small_scale_selection is not None or self.large_scale_selection is not None
        if self.small_scale_selection is not None:
            assert self.indices_pixels is not None, "indices_pixels must be provided for smoothing"
            self.set_small_scale_smoothing_scales()
        if self.large_scale_selection is not None:
            assert self.indices_pixels is not None, "indices_pixels must be provided for smoothing"
            self.set_large_scale_smoothing_scales()

        LOGGER.info(f"created physics model Linear, num_channels={self.num_channels}, num_targets={self.num_targets}, apply_scalers={self.scalers}, shape_noise_std={self.shape_noise_std}")
        LOGGER.debug(f"spins={self.channel_spins}")

    def set_small_scale_smoothing_scales(self):

        from nestconv import nestconv

        self.small_scale_smoothers = nn.ModuleList([None] * self.num_channels)
        # Channels with identical scale cuts share the same sparse geometry.
        smoothers = {}

        def get_smoother(sigma_arcmin, channel_id):

            cache_file=f"small_scale_smoother_channel{channel_id:02d}.pkl"
            if os.path.exists(cache_file):
                return nestconv(verbosity=1,
                                cache_file=cache_file)
            else:
                return nestconv(nside=self.nside, 
                                pixel_ids=self.indices_pixels, 
                                sigma=np.radians(sigma_arcmin / 60.), 
                                weighted=True, 
                                verbosity=1,
                                cache_file=cache_file)
            

        # six WL bins shear
        fwhm_mpc = self.small_scale_selection['WL']
        for i in range(6): 
            j = i + 0
            z_med = self.z_med_wl[i]
            sigma_arcmin = smoothing_sigma(z_med, fwhm_mpc)
            self.small_scale_smoothers[j] = get_smoother(sigma_arcmin, j)
            LOGGER.info(f"small scale smoother for lensing shear channel {j} fwhm_mpc = {fwhm_mpc} z_med = {z_med} sigma_arcmin = {sigma_arcmin}")
            LOGGER.info(self.small_scale_smoothers[j].stage_info())

        # six WL bins source clustering
        fwhm_mpc = self.small_scale_selection['GC']
        for i in range(6): 
            j = i + 6
            z_med = self.z_med_wl[i]
            sigma_arcmin = smoothing_sigma(z_med, fwhm_mpc)
            self.small_scale_smoothers[j] = get_smoother(sigma_arcmin, j)
            LOGGER.info(f"small scale smoother for source clustering channel {j} fwhm_mpc = {fwhm_mpc} z_med = {z_med} sigma_arcmin = {sigma_arcmin}")
            LOGGER.info(self.small_scale_smoothers[j].stage_info())


        # six GC bins galaxy clustering
        fwhm_mpc = self.small_scale_selection['GC']
        for i in range(6): 
            j = i + 12
            z_med = self.z_med_gc[i]
            sigma_arcmin = smoothing_sigma(z_med, fwhm_mpc)
            self.small_scale_smoothers[j] = get_smoother(sigma_arcmin, j)
            LOGGER.info(f"small scale smoother for galaxy clustering channel {j} fwhm_mpc = {fwhm_mpc} z_med = {z_med} sigma_arcmin = {sigma_arcmin}")
            LOGGER.info(self.small_scale_smoothers[j].stage_info())


    def set_large_scale_smoothing_scales(self):

        from nestconv import nestconv
        
        self.large_scale_smoothers = nn.ModuleList([None] * self.num_channels)

        fwhm_deg = self.large_scale_selection
        sigma_deg = fwhm_deg/2.355
        cache_file=f"large_scale_smoother.pkl"
        if os.path.exists(cache_file):
            smoothing = nestconv(verbosity=1, cache_file=cache_file)
        else:
            smoothing = nestconv(nside=self.nside, pixel_ids=self.indices_pixels, sigma=np.radians(sigma_deg), weighted=True, verbosity=1, cache_file=cache_file)

        LOGGER.info(smoothing.stage_info())

        # lensing weighted smoothing
        for j in range(self.num_channels): 
            self.large_scale_smoothers[j] = smoothing


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
        self.channel_scale = scale_channels.to(self.device)

    def set_params(self):

        self.params = ['Om', 's8', 'Ob', 'H0', 'ns', 'w0', 'bary_Mc', 'bary_nu', 'bg1', 'bg2', 'bg3', 'bg4', 'bg5', 'bg6', 'Aia1', 'Aia2', 'Aia3', 'Aia4', 'Aia5', 'Aia6', 'bsc1', 'bsc2', 'bsc3', 'bsc4', 'bsc5', 'bsc6']

        # from Table 1 in https://arxiv.org/pdf/2201.07771
        self.priors = {
            # cosmology parameters
            'Om':   [0.1, 0.5],
            's8':   [0.4, 1.4],
            'S8':   [0.23094010767585035, 1.8073922282301278],
            'H0':   [64, 82],
            'Ob':   [0.03, 0.06],
            'ns':   [0.87, 1.07],
            'w0':   [-2, -0.333],
            # baryonification parameters
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

    def select_onthefly_parameters(self, batch_size, id_cosmo):

        if self.randomize_astro_params:
            j = torch.randint(0, self.onthefly_samples.shape[0], (batch_size,), device=self.device)
        else:
            j = id_cosmo.to(self.device)
        return self.onthefly_samples[j].squeeze().to(self.device)

    def forward_physics(self, example):

        #
        # Preliminaries
        #

        # unpack the example
        maps, vec_int, hard_params = example
        id_cosmo = vec_int[:, 2]

        # this was created in the postprocessing pipeline, check if we are using the right data
        assert hard_params.shape[1] == 8, f"Expected 8 parameters, got {hard_params.shape[1]}"
        assert maps.shape[2] == 6, f"Expected 6 redshift bins per map, got {maps.shape[2]}"
        assert maps.shape[3] == 6, f"Expected 6 map types, got {maps.shape[3]}" # gg*2, ga*2, ds, dg

        # split into lensing, IA, and galaxy clustering maps
        gg1, gg2, ga1, ga2, ds, dg = maps.unbind(dim=-1)

        for i in range(6):
            print(f'gg1[{i}].shape {gg1[:,:,i].shape} gg1[{i}].nnz {(gg1[:,:,i]!=0).sum()}')
            print(f'gg2[{i}].shape {gg2[:,:,i].shape} gg2[{i}].nnz {(gg2[:,:,i]!=0).sum()}')
            print(f'ga1[{i}].shape {ga1[:,:,i].shape} ga1[{i}].nnz {(ga1[:,:,i]!=0).sum()}')
            print(f'ga2[{i}].shape {ga2[:,:,i].shape} ga2[{i}].nnz {(ga2[:,:,i]!=0).sum()}')
            print(f'ds[{i}].shape  {ds[:,:,i].shape}  ds[{i}].nnz  {(ds[:,:,i]!=0).sum()}')
            print(f'dg[{i}].shape  {dg[:,:,i].shape}  dg[{i}].nnz  {(dg[:,:,i]!=0).sum()}')

        # get onthefly parameters
        onthefly_params = self.select_onthefly_parameters(hard_params.shape[0], id_cosmo)
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
        ng = galaxy_density_to_count(ng_bar, dg, tomo_bg).to(torch.float32)
        LOGGER.debug(f'drawn poisson galaxy clustering map ng={ng.shape} dtype={ng.dtype} min={ng.min():>10.3f} max={ng.max():>10.3f} mean={ng.mean():>10.3f} nnz={(ng!=0).sum()} size={ng.size}')

        #
        # Linear bias map for galaxy clustering (sources)
        #

        # get mean number of galaxies per pixel
        ng_bar = self.num_gal_wl * self.pixel_area
        ids_bsc = [20, 21, 22, 23, 24, 25] # bsc1, bsc2, bsc3, bsc4, bsc5, bsc6
        tomo_bsc = targets[:, ids_bsc].unsqueeze(1) # shape (batch_size, 1, n_WL_bins)
        ns = galaxy_density_to_count(ng_bar, ds, tomo_bsc).to(torch.float32)
        LOGGER.debug(f'drawn poisson galaxy clustering map ns={ns.shape} dtype={ns.dtype} min={ns.min():>10.3f} max={ns.max():>10.3f} mean={ns.mean():>10.3f}')

        #
        # Add lensing shape noise to kappa
        #
        gg1_noise_temp = torch.empty(gg1.shape, dtype=torch.float32, device=self.device)
        gg2_noise_temp = torch.empty(gg2.shape, dtype=torch.float32, device=self.device)
        nn.init.trunc_normal_(gg1_noise_temp, mean=0.0, std=self.shape_noise_std/math.sqrt(2.), a=-1.0, b=1.0)
        nn.init.trunc_normal_(gg2_noise_temp, mean=0.0, std=self.shape_noise_std/math.sqrt(2.), a=-1.0, b=1.0)
        ns_sqrt = torch.sqrt(ns)
        select = ns_sqrt==0
        gg1_noise_temp.div_(ns_sqrt)
        gg2_noise_temp.div_(ns_sqrt)
        gg1_noise_temp[select] = 0
        gg2_noise_temp[select] = 0
        gg1.add_(gg1_noise_temp)
        gg2.add_(gg2_noise_temp)
        LOGGER.debug(f'created weak lensing map gg1={gg1.shape} dtype={gg1.dtype} min={gg1.min():>10.3f} max={gg1.max():>10.3f} mean={gg1.mean():>10.3f}')
        LOGGER.debug(f'created weak lensing map gg2={gg2.shape} dtype={gg2.dtype} min={gg2.min():>10.3f} max={gg2.max():>10.3f} mean={gg2.mean():>10.3f}')
        

        #
        # Add linear intrinsic alignment to shear maps
        #
        ids_ia = [14, 15, 16, 17, 18, 19] # Aia1, Aia2, Aia3, Aia4, Aia5, Aia6
        tomo_Aia = targets[:, ids_ia].unsqueeze(1) # shape (batch_size, 1, num_bins)
        ga1.mul_(tomo_Aia)
        ga2.mul_(tomo_Aia)
        gg1.add_(ga1)
        gg2.add_(ga2)
                           

        # 
        # Create inputs maps
        # 

        # final report
        LOGGER.debug(f'gg1.shape={gg1.shape}, gg1.dtype={gg1.dtype}')
        LOGGER.debug(f'gg2.shape={gg2.shape}, gg2.dtype={gg2.dtype}')
        LOGGER.debug(f'ns.shape={ns.shape}, ns.dtype={ns.dtype}')
        LOGGER.debug(f'ng.shape={ng.shape}, ng.dtype={ng.dtype}')

        # Stack probes as channels
        # inputs = torch.cat([kg_tot, ns, ng], dim=-1)
        gg = torch.complex(gg1, gg2)
        inputs = list(gg.unbind(dim=-1) + ns.unbind(dim=-1) + ng.unbind(dim=-1))

        for i, inp in enumerate(inputs):
            LOGGER.debug(f'inputs[{i}].shape={inp.shape} dtype={inp.dtype}')

        # 
        # Create weigths maps
        # 

        # initialize empty weights
        weights = [None] * len(inputs)

        # set lensing weights to source clustering pixel count
        for j in range(6):
            weights[j] = inputs[j+6]
        for j in range(6, self.num_channels):
            # dims = [1] * len(inputs[j].shape)
            # weights[j] = np.ones(dims, dtype=np.float32) # only rank, will broadcast
            weights[j] = torch.where(inputs[j] != 0, 1, 0)

        return inputs, weights, targets


    def apply_scalers(self, inputs, targets):

        # per channel scaler
        for i, inp in enumerate(inputs):
            # print(f'inputs[{i}].shape={inputs[i].shape} dtype={inputs[i].dtype} self.channel_scale.shape={self.channel_scale.shape}', f'self.channel_shift={self.channel_shift}')
            inputs[i] = (inputs[i] + self.channel_shift) * self.channel_scale[i]

        # vectorized targets scaler
        targets = (targets + self.targets_shift) * self.targets_scale

        return inputs, targets


    def apply_smoothing(self, inputs, weights):

        inputs_smoothed = [None] * self.num_channels

        # difference of Gaussians - like bandpass
        # weights are used just for smoothing, not for the final output

        if self.small_scale_selection is not None:
            for j in range(self.num_channels):
                print(f'{j} inputs shape={inputs[j].shape} dtype={inputs[j].dtype} weights shape={weights[j].shape} dtype={weights[j].dtype}')
                m_small, w_small = self.small_scale_smoothers[j](inputs[j], weights[j])
                inputs_smoothed[j] = m_small

        if self.large_scale_selection is not None:
            for j in range(self.num_channels):
                m_large, w_large = self.large_scale_smoothers[j](inputs[j], weights[j])
                inputs_smoothed[j] = inputs_smoothed[j] - m_large
                # inputs_smoothed[j] = m_large

        # # set the zero-weight pixel back to zero
        for i in range(len(inputs_smoothed)):
            print(f"weight {weights[i].shape} num nonzero weights {(weights[i]!=0).sum()}")
            inputs_smoothed[i] = torch.where(weights[i] == 0, 0, inputs_smoothed[i])
       
        return inputs_smoothed
    
    def forward(self, example):
        
        inputs, weights, targets = self.forward_physics(example)

        if self.smoothing:
            inputs = self.apply_smoothing(inputs, weights)

        # if self.scalers:
        #     inputs, targets = self.apply_scalers(inputs, targets)

        return inputs, targets




    def preprocess_for_correlations(self, inputs):

        # first 6 channels are shear maps, subtract the mean
        # this should work on complex tensors
        for i in range(6):
            select = inputs[i] != 0
            inputs[i] = inputs[i] - inputs[i][select].mean(dim=-1, keepdim=True)
        
        # next 12 channels are clustering maps, convert to overdensity rho (not density contrast)
        for i in range(6, 18):
            select = inputs[i] != 0
            inputs[i] = inputs[i] / inputs[i][select].mean(dim=-1, keepdim=True)

        return inputs
        



# Code graveyard:

    # def unstack_batch_channels(self, inputs):
    #     """
    #     Unstack the channels of the input tensor into a list of maps.
    #     Inputs: 
    #         inputs: tensor of shape (batch_size, num_pixels, num_channels)
    #     Outputs:
    #         maps: list of tensors, each tensor is of shape (batch_size, num_pixels).
    #     """

    #     # all maps are scalar, simple unbind is enough
    #     channel_maps = inputs.unbind(dim=-1)

    #     return channel_maps
