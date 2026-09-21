"""Small, readable training utilities for DeepLSS regression experiments."""

from __future__ import annotations

import shutil
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .utils.config import load_config 
from .utils.logger import get_logger

LOGGER = get_logger(__file__)

# PASTED CODE : START #########################################################################################


    LOGGER.timer.start("main")
    LOGGER.info(f"Got index set of size {len(indices)}")

    # I/O delay
    if args.debug:
        args.max_sleep = 0
        LOGGER.warning("debug mode")
    sleep_sec = np.random.uniform(0, args.max_sleep) if args.max_sleep > 0 else 0
    LOGGER.info(f"waiting for {sleep_sec:.2f}s to prevent overloading IO")
    time.sleep(sleep_sec)

    # configuration
    conf = files.load_config(args.config)
    with open(os.path.join(args.dir_out, "config.yaml"), "w") as f:
        yaml.dump(conf, f)

    # directories
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    meta_info_file = os.path.join(repo_dir, conf["files"]["meta_info"])

    cosmo_params_info = cosmogrid.get_cosmo_params_info(meta_info_file, "grid")
    cosmo_dirs = [cosmo_dir.decode("utf-8") for cosmo_dir in cosmo_params_info["path_par"]]
    cosmo_dirs_in = [os.path.join(args.dir_in, "grid", cosmo_dir) for cosmo_dir in cosmo_dirs]

    # CosmoGrid
    n_patches = conf["analysis"]["n_patches"]
    n_cosmos = 2500
    n_cosmos_per_file = args.n_cosmos_per_file
    assert n_cosmos % n_cosmos_per_file == 0
    n_perms_per_cosmo = conf["analysis"]["grid"]["n_perms_per_cosmo"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo
    LOGGER.info(
        f"for every cosmology, theres {n_examples_per_cosmo} examples: {n_patches} patches times {n_perms_per_cosmo} permutations"
    )

    # modeling
    baryonified = conf["analysis"]["modelling"]["baryonified"]

    # analysis files
    pixel_file = files.load_pixel_file(conf)
    nside = int(conf["analysis"]["n_side"])
    nside_down = int(conf["analysis"]["n_side_down"])
    

    # constants
    maps_to_store = conf["survey"]["WL"]["map_types"]["onthefly_store"] + conf["survey"]["GC"]["map_types"]["onthefly_store"]
    full_sky_samples = {"kg":'WL', "ia":'WL', "gg":'WL', "ga":'WL', "ds":'WL', "gd":'WL', "dg":'GC', "qg":'GC'}
    LOGGER.info(f"nside={nside} nside_down={nside_down} maps_to_store={maps_to_store}")

    LOGGER.info(f"starting the main loop trough indices {indices}")

    # index corresponds to a webdataset tar file ###########################################################################
    for index in indices:

        i_cosmo_set = index % n_cosmos
        # # index for the cosmological parameters
        i_cosmo_start = (i_cosmo_set * n_cosmos_per_file) % n_cosmos
        i_cosmo_end = i_cosmo_start + n_cosmos_per_file
        i_perm = (index*n_examples_per_cosmo) // n_cosmos

        LOGGER.info(f"starting index {index} i_cosmo_set={i_cosmo_set:>5d} i_perm={i_perm:>2d}/{n_perms_per_cosmo} i_cosmo_start={i_cosmo_start:>5d}/{n_cosmos} i_cosmo_end={i_cosmo_end:>5d}/{n_cosmos}")
        LOGGER.info(f"cosmo dirs {cosmo_dirs[i_cosmo_start : i_cosmo_end]}")
        LOGGER.timer.start("index")

        if args.debug:
            args.dir_out = os.path.join(args.dir_out, "debug")
            os.makedirs(args.dir_out, exist_ok=True)

        # initialize the webdataset tar file writer
        num_total_examples = 0

        with ExitStack() as stack:

            list_wds_files = []

            for i_ in range(n_patches):

                wds_file = filenames.get_filename_webdataset(
                    args.dir_out,
                    tag=conf["survey"]["name"] + f"_patch{i_:02d}" + args.file_suffix,
                    index=index,
                    simset="grid",
                    with_bary=baryonified,
                )
                LOGGER.info(f"index {index} is writing to {wds_file}")

                list_wds_files.append((wds_file, stack.enter_context(webdataset.TarWriter(wds_file, encoder=True))))

            # loop over the cosmological parameters
            j = 0
            for i_cosmo, cosmo_dir_in in LOGGER.progressbar(
                zip(range(i_cosmo_start, i_cosmo_end), cosmo_dirs_in[i_cosmo_start:i_cosmo_end]),
                at_level="debug",
                desc="looping through cosmologies\n",
                total=i_cosmo_end - i_cosmo_start,
            ):  
                j += 1
                LOGGER.info(f"j={j:>3d}/{n_cosmos_per_file} i_cosmo={i_cosmo:>5d} i_perm={i_perm:>2d} cosmo_dir_in={cosmo_dir_in}")
                LOGGER.timer.start("cosmo")


                # constants
                cosmo = prior.get_hard_parameters(conf, cosmo_params_info, i_cosmo)
                i_sobol = int(cosmo_dir_in[-7:-1])
             

                ##
                ## Main magic - get postprocessed full sky maps
                ##
                full_maps_file = postprocessing._get_full_sky_perm(args, conf, cosmo_dir_in, i_perm)
                full_sky_maps = get_postprocessed_maps(conf, full_maps_file)

                LOGGER.debug('full_sky_maps')
                for m_name in full_sky_maps.keys():
                    for i,m in enumerate(full_sky_maps[m_name]):
                        LOGGER.debug(f"{m_name} {i}: shape={m.shape}, dtype={m.dtype}")
                    
                # write patches
                for i_patch in range(n_patches):

                    patch_maps = {}
                    for m_name in full_sky_maps.keys():

                        patch_maps[m_name] = []
                        for i_z, m in enumerate(full_sky_maps[m_name]):
                            patch_map_ = postprocessing.full_sky_to_patch(m, conf, pixel_file, i_z, i_patch, sample=full_sky_samples[m_name])
                            patch_maps[m_name].append(patch_map_[..., np.newaxis]) # shape n_pix, n_z_bins
                        patch_maps[m_name] = np.concatenate(patch_maps[m_name], axis=-1)

                    # build output dict to be stored
                    # maps_complex = torch.from_numpy(np.concatenate([patch_maps[m_name][..., np.newaxis] for m_name in ['gg', 'ga', 'gd']], axis=-1))
                    # maps_float = torch.from_numpy(np.concatenate([patch_maps[m_name][..., np.newaxis] for m_name in ['ds', 'dg', 'qg']], axis=-1))
                    # vec_int = torch.from_numpy(np.array([i_sobol, i_signal, n_z_WL, n_z_GC]))

                    list_maps_to_store = []
                    list_channels = []

                    for m_name in maps_to_store:
                            
                        m = patch_maps[m_name][..., np.newaxis]

                        LOGGER.debug(f'{m_name} shape={m.shape} dtype={m.dtype}')

                        if np.issubdtype(m.dtype, np.complexfloating):
                            list_maps_to_store.extend([m.real, m.imag])
                            list_channels.extend([m_name+'1', m_name+'2'])
                        elif np.issubdtype(m.dtype, np.floating):
                            list_maps_to_store.append(m)
                            list_channels.append(m_name)
                        else:
                            raise ValueError(f"Unsupported dtype: {m.dtype}")

                    tensor_float = np.concatenate(list_maps_to_store, axis=-1)

                    # LOGGER.warning('TODO: currently the tensor shapes assume that the number of z-bins is the same for all maps, this should be fixed')


                    LOGGER.debug(f'tensor_float.shape={tensor_float.shape}')

                    i_signal = index * n_cosmos * n_perms_per_cosmo * n_patches    \
                                    +  i_cosmo * n_perms_per_cosmo * n_patches   \
                                    +  i_perm * n_patches   \
                                    +  i_patch

                    dict_out = {
                            "__key__": f"{i_signal:09d}",
                            "maps_float32.pth": torch.from_numpy(tensor_float.astype(np.float32)),
                            "vec_int32.pth": torch.from_numpy(np.array([i_signal, i_sobol, i_cosmo, i_perm, i_patch, nside, nside_down]).astype(np.int32)),
                            "vec_float32.pth": torch.from_numpy(cosmo.astype(np.float32)),
                        }

                    # writeout to webdataset
                    wds_file, wds_writer = list_wds_files[i_patch]
                    wds_writer.write(dict_out)
                    del_dict(dict_out)
                    num_total_examples += 1

                    
                    LOGGER.info(f"wrote example to {wds_file} i_cosmo={i_cosmo:>5d} i_perm={i_perm:>2d}, i_patch={i_patch:>2d}, i_signal={i_signal:>8d} channels={list_channels}")
                    for key in patch_maps.keys():
                        LOGGER.debug(f"{key}.shape={patch_maps[key].shape}, dtype={patch_maps[key].dtype}")

                LOGGER.info(f"done with i_cosmo={i_cosmo} i_perm={i_perm} after {LOGGER.timer.elapsed('cosmo')}")
                                   
        LOGGER.info(f"done with index={index} after {LOGGER.timer.elapsed('index')}")

    return num_total_examples


        
def get_postprocessed_maps(conf, full_maps_file):

    # filepaths
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    # constants
    n_side = conf["analysis"]["n_side"]
    kappa2gamma_fac, gamma2kappa_fac, _ = lensing.get_kaiser_squires_factors(3 * n_side - 1)
    z_bins_WL = conf["survey"]["WL"]["z_bins"]
    z_bins_GC = conf["survey"]["GC"]["z_bins"]


    # container
    full_sky_maps = {"kg": [], "ia": [], "gg": [], "ga": [], "gd": [], "ds": [], "dg": [], "qg": []}

    # loop over lensing bins
    for i_z, z_bin in enumerate(z_bins_WL):

        ##
        ## Lensing convergence
        ##

        kg = postprocessing._read_full_sky_bin(conf, full_maps_file, "kg", z_bin)
        full_sky_maps["kg"].append(kg.astype(np.float32))

        ##
        ## Lensing shear
        ##

        # kappa to shear conversion for lensing signal
        g1_, g2_ = lensing.kappa_to_gamma(kg, hp_datapath, kappa2gamma_fac, n_side)
        gg_ = g1_ + 1j*g2_
        full_sky_maps["gg"].append(gg_.astype(np.complex64))

        ##
        ## Linear intrinsic alignment convergence
        ##

        ia = postprocessing._read_full_sky_bin(conf, full_maps_file, "ia", z_bin)
        full_sky_maps["ia"].append(ia.astype(np.float32))


        ##
        ## Linear intrinsic alignment shape
        ##

        # kappa to shear conversion for intrinsic alignment
        g1_, g2_ = lensing.kappa_to_gamma(ia, hp_datapath, kappa2gamma_fac, n_side)
        ga_ = g1_ + 1j*g2_
        full_sky_maps["ga"].append(ga_.astype(np.complex64)) 

        ##
        ## Source sample galaxy counts
        ##

        # source sample galaxy counts for shape noise
        ds_ = postprocessing._read_full_sky_bin(conf, full_maps_file, "dg", z_bin)
        full_sky_maps["ds"].append(ds_.astype(np.float32))

        ##
        ## Delta-NLA intrinsic alignment
        ##

        # delta-NLA component approximation
        gd_ = ga_ * ds_ # approximation
        full_sky_maps["gd"].append(gd_.astype(np.complex64))

    # loop over clustering bins
    for i_z, z_bin in enumerate(z_bins_GC):

        ##
        ## Galaxy counts
        ##

        dg_ = postprocessing._read_full_sky_bin(conf, full_maps_file, "dg", z_bin)
        full_sky_maps["dg"].append(dg_.astype(np.float32))

        ##
        ## Quadratic galaxy counts
        ##

        # quadratic galaxy counts for shape noise
        qg_ = (dg_**2) # approximation
        full_sky_maps["qg"].append(qg_.astype(np.float32))

    return full_sky_maps

def del_dict(dict_):

    for key in list(dict_.keys()):
        del(dict_[key])
    del(dict_)


if __name__ == "__main__":

    args = setup(sys.argv[1:])

    if args.command == 'wds':

        indices = configuration.get_indices(args.indices)
        main(indices=indices, args=args)

    elif args.command == 'test':

        # # test the onthefly_pipeline

        LOGGER.info("Testing the onthefly_pipeline")

        from msfm.onthefly_pipeline import OntheflyPipeline

        loader = OntheflyPipeline().get_loader(
            webds_pattern=os.path.join(args.dir_out, "*.tar"),
            batch_size=8,
        )

        for batch in loader:
            gg, ga, gd, ds, dg, qg, cosmo, i_sobol, i_signal, n_params, n_pix, n_z_WL, n_z_GC = batch
            print(f"gg.shape = {gg.shape}, gg.dtype = {gg.dtype}")
            print(f"ga.shape = {ga.shape}, ga.dtype = {ga.dtype}")
            print(f"gd.shape = {gd.shape}, gd.dtype = {gd.dtype}")
            print(f"ds.shape = {ds.shape}, ds.dtype = {ds.dtype}")
            print(f"dg.shape = {dg.shape}, dg.dtype = {dg.dtype}")
            print(f"qg.shape = {qg.shape}, qg.dtype = {qg.dtype}")
            print(f"cosmo.shape = {cosmo.shape}, cosmo.dtype = {cosmo.dtype}")
            print(f"i_sobol.shape = {i_sobol.shape}, i_sobol.dtype = {i_sobol.dtype}")
            print(f"i_signal.shape = {i_signal.shape}, i_signal.dtype = {i_signal.dtype}")
            print(f"n_params = {n_params}, n_params.dtype = {n_params.dtype}")
            print(f"n_pix.shape = {n_pix.shape}, n_pix.dtype = {n_pix.dtype}")
            print(f"n_z_WL.shape = {n_z_WL.shape}, n_z_WL.dtype = {n_z_WL.dtype}")
            print(f"n_z_GC.shape = {n_z_GC.shape}, n_z_GC.dtype = {n_z_GC.dtype}")
            break

        LOGGER.info("Testing the onthefly_linear physics model")

        from msfm.onthefly_physics.onthefly_linear import OntheflyPhysicsModelLinear
        conf = files.load_config(args.config)
        model = OntheflyPhysicsModelLinear(conf, num_samples_prior=1_000_000)
        loader = model.get_loader(
            webds_pattern=os.path.join(args.dir_out, "*.tar"), 
            batch_size=8
            )

        for batch in loader:
            inputs, targets  = batch
            print(f"inputs = {inputs.shape}")
            print(f"targets = {targets.shape}")
            break

        fname = "inputs.npy"
        np.save(fname, inputs.numpy())
        LOGGER.info(f"Saved inputs to {fname} size={inputs.nbytes/1024**2:.2f} MB")

# PASTED CODE : END #########################################################################################