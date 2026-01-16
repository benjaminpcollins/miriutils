#!/usr/bin/env python
# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "astropy",
#     "h5py",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "photutils",
#     "scipy",
#     "stpsf",
# ]
# ///
"""
MIRI Utils: Photometric Pipeline Module
==========================================

This module provides the MiriPipeline class, a specialized framework designed
to produce publication-ready mid-infrared photometry for the Blue Jay survey.
It automates the transition from raw FITS cutouts in the '_i2d.fits' format to
aperture-corrected Janskys.

Class
-------
    - MiriPipeline: The core engine for batch processing MIRI photometry.

Key Capabilities
----------------
    - Automated "Wide" Table generation (one row per ID, columns per band).
    - Multi-instrument WCS alignment and automated aperture adjustment.
    - Local background modeling using iterative sigma-clipped 2D statistics.
    - High-fidelity aperture corrections using 4x oversampled stpsf models.
    - Rigorous error propagation including nominal detector noise and 
      background modeling uncertainties.
    - Quality flagging for detector artefacts and companion contamination.
    - Stores data for visualising the background modelling in h5py format
      and provides functions for easy reading and plotting.

Workflow
--------
    1. Pre-scans directories to initialize a type-safe, wavelength-ordered table structure.
    2. Aligns science apertures based on Blue Jay (NIRCam F444W) morphology.
    3. Performs exact aperture photometry and background estimation.
    4. Calculates band-specific PSF corrections on an oversampled grid.
    5. Exports a dual-format (FITS/CSV) catalogue with standardised columns.

Example usage
-------------
    from miri_utils import MiriPipeline

    ids_to_process = [7102, 11202, 16874]   # int of galaxy IDs to process
    
    # Initialise the pipeline
    pipeline = BlueJayMiriPipeline(
        all_ids=ids_to_process,
        cutout_dir="./data/cutouts",
        output_dir="./miri_photometry",
        nircam_dir="./NIRCam/cutouts",
        aperture_table="./data/aperture_table.fits"
    )

    # Run full survey photometry and store FITS and CSV format output tables
    pipeline.run_photometry(write_to="Phot_Table_MIRI")
    

Author: Benjamin P. Collins
Date: January 15, 2026
Version: 3.2.0
"""

import os
import time
import warnings
import random
import glob
import json
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from scipy.stats import skew
from scipy.ndimage import binary_dilation

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.table import Table, MaskedColumn
from astropy.stats import sigma_clip, sigma_clipped_stats

import stpsf
from photutils.aperture import (
    EllipticalAperture,
    EllipticalAnnulus,
    aperture_photometry,
)
from photutils.centroids import centroid_com
from photutils.segmentation import detect_sources
from photutils.utils.exceptions import NoDetectionsWarning

# --- Global Configuration & Silencing ---

# Suppress common WCS-related warnings that don't affect photometric accuracy
warnings.simplefilter("ignore", category=FITSFixedWarning)

# Silencing NoDetectionsWarning from segmentation-mapping
warnings.filterwarnings('ignore', category=NoDetectionsWarning)

# --- Standalone Utility Functions ---
class MIRIPipeline:
    def __init__(self, all_ids, cutouts_dir, output_dir, nircam_dir, aperture_table, psf_dir=None, scaling_exceptions_file=None):
        
        # Initialise directories
        self.cutouts_dir = cutouts_dir
        self.output_dir = output_dir
        self.nircam_dir = nircam_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load aperture table
        self.master_table = Table.read(aperture_table)
        
        self.all_ids = all_ids
        
         # 1. Map filters to central wavelengths (microns) for correct physical sorting
        self.wavelength_map = {
            'F560W': 5.6,
            'F770W': 7.7,
            'F1000W': 10.0,
            'F1130W': 11.3,
            'F1280W': 12.8,
            'F1500W': 15.0,
            'F1800W': 18.0,
            'F2100W': 21.0,
            'F2550W': 25.5
        }
        
        # Default PSF directory if none provided
        if psf_dir is None:
            self.psf_dir = os.path.join(self.output_dir, "psfs")
            print(f"Using default PSF directory: {self.psf_dir}")
        else:
            self.psf_dir = psf_dir
            print(f"Found PSF directory {self.psf_dir}")
            
        # Ensure the PSF directory exists
        os.makedirs(self.psf_dir, exist_ok=True)
        
        # 2. Handle Scaling Exceptions File
        if scaling_exceptions_file is None:
            # Default name in the output directory if none provided
            scaling_exceptions_file = os.path.join(self.output_dir, "scaling_config.csv")

        self.scaling_exceptions_path = scaling_exceptions_file
        self.scaling_exceptions = self._initialise_scaling_config()
        
        # Place these in your __init__ or as a config block
        self.quality_config = {
            "exclude_all": [18094, 19307],
            "exclude_filters": {
                "F770W": [16424], "F1000W": [], "F1800W": [12202, 12332, 16419], "F2100W": [7102, 16874],
            },
            "art_filters": {
                "F770W": [7185, 8013, 8469, 8500, 8843, 9517, 11136, 11137, 11494, 11716, 16516, 17793, 19098, 21451],
                "F1000W": [],
                "F1800W": [7102, 11716, 12202, 17793, 19098, 21451],
                "F2100W": [11723, 12175, 12213, 16874, 17984],
            },
            "has_companion": [7136, 7904, 7922, 7934, 8469, 10314, 16424, 17517, 18332, 21452]
        }


    def _initialise_scaling_config(self):
        """Creates a template CSV if missing; otherwise loads existing values."""
        if not os.path.exists(self.scaling_exceptions_path):
            print(f"Creating new scaling config template: {self.scaling_exceptions_path}")
            
            instructions = [
                "# MIRI Photometry Pipeline: Scaling Configuration",
                "# Edit the Scale_Factor column for individual galaxies.",
                "# Default factor is 2.0. Factors are multipliers for the NIRCam aperture size.",
                "# Comments can contain any text. Commas are allowed (Pandas handles quoting).",
                "# --------------------------------------------------------------------------"
            ]
            
            # Create a dataframe with all IDs and a default scale factor of 2.0
            df = pd.DataFrame({
                'ID': self.all_ids,
                'Scale_Factor': [2.0] * len(self.all_ids),
                'Comment': ["" for _ in self.all_ids]
            })
            
            with open(self.scaling_exceptions_path, 'w') as f:
                for line in instructions:
                    f.write(line + "\n")
                # Use index=False to keep it clean. 
                # Pandas will automatically wrap comments in quotes if they contain commas.
                df.to_csv(f, index=False)
                
            return dict(zip(df['ID'], df['Scale_Factor']))
        
        else:
            print(f"Loading existing scaling config: {self.scaling_exceptions_path}")
            df = pd.read_csv(self.scaling_exceptions_path, comment="#")
            # Ensure IDs in the CSV are treated as ints for dictionary matching
            return dict(zip(df['ID'].astype(int), df['Scale_Factor']))
        
    def get_psf(self, filter_name, oversample=4):
        """
        Retrieves a PSF. If the file doesn't exist, it uses stpsf to 
        generate, save, and return it.
        """
        psf_filename = f"PSF_MIRI_{filter_name}.fits"
        psf_path = os.path.join(self.psf_dir, psf_filename)

        # 1. Check if the file already exists
        if not os.path.exists(psf_path):
            print(f"⚡️ PSF for {filter_name} not found. Generating with stpsf (oversample={oversample})...")
            
            # 2. Generate the PSF
            inst = stpsf.MIRI()
            inst.filter = filter_name
            # calc_psf can take a while, so we only do this if necessary
            psf_obj = inst.calc_psf(oversample=oversample)
            
            # 3. Display and save to disc for future use
            stpsf.display(psf_obj)
            psf_obj.writeto(psf_path, overwrite=True)
            print(f"Saved PSF to {psf_path}")

        # 4. Read and return the recommended extension (Ext 3 is the oversampled PSF)
        with fits.open(psf_path) as hdul:
            # Note: Extension 0 = Detector sampled, 3 = Oversampled (usually)
            # Use extension 3 if you are doing precise modeling/deconvolution
            print(f"✅ PSF file found! Loading {psf_path}.")
            return hdul[3].data
    
    def find_files(self, gid, priority=['primer', 'cweb', 'cos3d1', 'cos3d2']):
        """
        Finds one file per filter using the directory structure as the source of truth.
        Structure: .../survey_obs/FILTER/fits/12345_f770w_obs.fits
        """
        # 1. Search recursively using Pathlib
        root = Path(self.cutouts_dir)
        # This finds all FITS files starting with the galaxy ID
        all_paths = list(root.rglob(f"{gid}_*.fits"))
        
        # 2. Group by the folder name (the parent of the parent)
        # path.parent is 'fits', path.parent.parent is 'F1000W'
        files_by_filter = {}
        for p in all_paths:
            filter_name = p.parent.parent.name.upper() # e.g., "F1000W"
            files_by_filter.setdefault(filter_name, []).append(p)
        
        selected_files = {}

        # 3. For each filter, pick the best file based on priority
        for filter_name, path_list in files_by_filter.items():
            best_file_for_filter = None
            
            # This loop enforces the priority: 
            # It checks for 'primer' across ALL files in this filter first.
            # Only if no 'primer' is found does it move to 'cweb'.
            for p_key in priority:
                matches = [p for p in path_list if p_key.lower() in str(p).lower()]
                
                if matches:
                    # We found a match for the highest available priority!
                    # If there's more than one (like primer1 and primer2), matches[0] is fine
                    best_file_for_filter = str(matches[0])
                    break # Stop looking for lower priorities (cweb/cos3d) for this filter
            
            if best_file_for_filter:
                selected_files[filter_name] = best_file_for_filter
        
        # Sort the dictionary by wavelength using our mapping
        # We use .get(k, 99.0) to handle any unexpected filter names by putting them at the end
        sorted_keys = sorted(selected_files.keys(), key=lambda k: self.wavelength_map.get(k, 99.0))
        sorted_files = {k: selected_files[k] for k in sorted_keys}

        return sorted_files if len(sorted_files) > 0 else None
            
    def _parse_path_metadata(self, file_path):
        """
        Extracts metadata from the directory structure.
        Example path: .../cos3d1/F1000W/fits/13297_f1000w_cos3d1.fits
        """
        path_parts = file_path.split(os.sep)
        
        # Extract from directory names for reliability
        survey_obs = path_parts[-4] # e.g., 'cos3d1'
        filter_name = path_parts[-3] # e.g., 'F1000W'
        filename = path_parts[-1]
        galaxy_id = filename.split('_')[0]
        
        return {
            "id": galaxy_id,
            "filter": filter_name,
            "survey_obs": survey_obs,
            "full_path": file_path
        }
        
        
    def get_wcs_rotation(self, wcs):
        # This works regardless of whether the header uses CD or PC
        pc = wcs.wcs.get_pc()
        
        # arctan2(y, x) -> result is in radians
        rot_rad = np.arctan2(pc[0, 1], pc[1, 1])
        
        # Explicit conversion so your 'Apr_Theta' matching is safe
        return np.degrees(rot_rad)
    
    
    def prepare_aperture(self, file_path, rescale=True):
        """
        1. Parse Metadata
        2. Load FITS/WCS
        3. Match with Master Catalogue
        4. Project NIRCam coords -> MIRI pixels
        5. Apply Rotation and Rescaling
        """
        meta = self._parse_path_metadata(file_path)
        gid = int(meta['id'])

        # --- 1. Load Master Aperture ---
        # Assuming self.master_table is already loaded in __init__
        matches = self.master_table[self.master_table["ID"] == str(gid)]
        if len(matches) == 0:
            print(f"Warning: Galaxy {gid} not in catalogue.")
            return None
        row = matches[0]

        # --- 2. Load MIRI and NIRCam WCS ---
        with fits.open(file_path) as hdu_miri:
            header_miri = hdu_miri['SCI'].header
            wcs_miri = WCS(header_miri)
            data_miri = hdu_miri['SCI'].data
            err_miri = hdu_miri['ERR'].data
           
            # Get the pixel area in steradians
            # Defaulting to a MIRI average if the header is missing (though it shouldn't be)
            # 1. Try to get the official pipeline value
            pixel_area_sr = header_miri.get("PIXAR_SR")

            # 2. If it's missing (None), use the geometric derivation
            if pixel_area_sr is None:
                # Use the WCS pixel scale matrix determinant
                # This accounts for the transformation of the pixel area to the sky
                pix_scale_deg = np.sqrt(np.abs(np.linalg.det(wcs_miri.pixel_scale_matrix)))
                pixel_area_sr = (np.radians(pix_scale_deg))**2
                
                # Optional: Log a warning so you know you're using a fallback
                print("WARNING: PIXAR_SR not found. Using WCS geometric derivation.")
    
            # Scalar flux conversion factor from counts/sec to photometric 
            # units for the given dataset mode and photometric reference table.
            photmjsr = header_miri.get("PHOTMJSR", 1.0)
            
            # Get rotation directly from header logic
            miri_rotation = self.get_wcs_rotation(wcs_miri)

        meta["pixel_area_sr"] = pixel_area_sr
        meta["photmjsr"] = photmjsr

        nircam_path = os.path.join(self.nircam_dir, f"{gid}_F444W_cutout.fits")
        with fits.open(nircam_path) as hdu_ni:
            wcs_ni = WCS(hdu_ni['SCI'].header)
            
            # Get rotation directly from header logic
            ni_rotation = self.get_wcs_rotation(wcs_ni)
            
        # --- 3. Coordinate Transformation ---
        # Project NIRCam pixel center to World (RA/Dec) then to MIRI pixels
        sky_coord = wcs_ni.pixel_to_world(row["Apr_Xcenter"], row["Apr_Ycenter"])
        miri_x, miri_y = wcs_miri.world_to_pixel(sky_coord)
        
        # --- 4. Rotation Logic ---
        # The change in rotation between the two images
        delta_rot = miri_rotation - ni_rotation
        new_theta_deg = (row["Apr_Theta"] + delta_rot) % 180 * u.deg
        
        # --- 5. Rescaling Logic ---
        miri_pix_scale = np.sqrt(np.abs(np.linalg.det(wcs_miri.pixel_scale_matrix))) # ~0.11
        ni_pix_scale = np.sqrt(np.abs(np.linalg.det(wcs_ni.pixel_scale_matrix))) # ~0.11
        pixel_conversion = ni_pix_scale / miri_pix_scale
        
        # Lookup the user multiplier (default to 2.0 if ID not in exceptions)
        # self.scaling_exceptions is a dict loaded from a CSV in __init__
        multiplier = self.scaling_exceptions.get(int(gid), 2.0)
        
        # Always rescale pixels from NIRCam to MIRI
        # Only rescale apertures additionally if rescale=True
        total_scale = pixel_conversion * multiplier if rescale else pixel_conversion
        
        # Apply total scaling to apertures
        a_rescaled = row["Apr_A"] * total_scale
        b_rescaled = row["Apr_B"] * total_scale
        
        return {
            "id": gid,
            # Store information about the modified MIRI aperture
            "x": miri_x,
            "y": miri_y,
            "a": a_rescaled/2,
            "b": b_rescaled/2,
            "theta": -np.radians(new_theta_deg),
            # Store information about the original aperture
            #"x_orig": row["Apr_Xcenter"],
            #"y_orig": row["Apr_Ycenter"],
            "a_orig": (row["Apr_A"]/2) * pixel_conversion,
            "b_orig": (row["Apr_B"]/2) * pixel_conversion,
            "theta_orig": -np.radians(row["Apr_Theta"] * u.deg),
            # Store data and metadata
            "data": data_miri, 
            "err": err_miri,
            "meta": meta,
            #"pixel_conversion": pixel_conversion
        }
    
    def plot_apertures_multiband(self, results_list, output_dir=None):
        """
        results_list: A list of dictionaries returned by prepare_aperture for ONE galaxy.
        """
        # Sort by wavelength: Extract the number from 'F770W', 'F1000W', etc.
        # This ensures F770W comes before F1000W
        results_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x['meta']['filter']))))
        n_bands = len(results_list)
        fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 4), squeeze=False)
        
        for i, params in enumerate(results_list):
            ax = axes[0, i]
            data = params["data"]
            filt = params["meta"]["filter"]
            survey_obs = params["meta"]["survey_obs"]
            
            # Robust scaling per band
            vmin, vmax = np.nanpercentile(data, [10, 99.5])
            
            ax.imshow(data, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
            
            # Draw the aperture (remembering photutils needs radians)
            ap = EllipticalAperture((params["x"], params["y"]), 
                                a=params["a"], b=params["b"], 
                                theta=np.radians(params["theta"]))
            ap.plot(ax=ax, color="cyan", lw=2, label="MIRI Adjusted")
            
            if i == 0:
                ap_orig = EllipticalAperture((params["x"], params["y"]), 
                                        a=params["a_orig"], 
                                        b=params["b_orig"], 
                                        theta=np.radians(params["theta_orig"]))
                ap_orig.plot(ax=ax, color="red", lw=1.5, label="NIRCam Original", alpha=0.7)
                ax.legend(loc="upper right", fontsize=8)
                
            # Zoom to 4" x 4" (approx 40x40 MIRI pixels)
            ax.set_xlim(params["x"] - 20, params["x"] + 20)
            ax.set_ylim(params["y"] - 20, params["y"] + 20)
            ax.set_title(f"{filt} | {survey_obs}")
            ax.axis('off')
        
        galaxy_id = results_list[0]['id']
        
        plt.suptitle(f"Galaxy ID: {galaxy_id}", fontsize=14)
        if output_dir is None:
            aperture_dir = os.path.join(self.output_dir, "aperture_plots")
        os.makedirs(aperture_dir, exist_ok=True)
        out_path = os.path.join(aperture_dir, f"{galaxy_id}_all.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        

    def estimate_background(self, aperture_params, sigma_val=2.5, save_vis=None):
        """
        Fits a 2D plane to the image (excluding sources) and calculates 
        local statistics in an elliptical annulus.
        """
        data = aperture_params["data"]
        x, y = aperture_params["x"], aperture_params["y"]
        a, b = aperture_params["a"], aperture_params["b"]
        theta = aperture_params["theta"] # Already in Radians from prepare_aperture
        galaxy_id = aperture_params["id"]
        
        filt = aperture_params["meta"]["filter"]

        # 1. Create Source Mask
        source_ap = EllipticalAperture((x, y), a=a, b=b, theta=theta)
        source_mask = source_ap.to_mask(method='center').to_image(data.shape).astype(bool)
        
        # 2. Initial Sigma Clipping & Source Detection for Masking
        # We ignore the source and NaNs
        # 1. Create an oversized Source Mask (Buffer of 2-3x the aperture)
        # This prevents the target's own light from biasing the background
        a_in, b_in = a + 8, b + 8
        
        bkg_source_ap = EllipticalAperture((x, y), a=a_in, b=b_in, theta=theta)
        source_mask_large = bkg_source_ap.to_mask(method='center').to_image(data.shape).astype(bool)

        # 2. Aggressive Neighbor Detection
        # We use a very low threshold (1.5 sigma) to catch faint wings
        init_mask = source_mask_large | np.isnan(data)
        _, median_init, std_init = sigma_clipped_stats(data, sigma=3.0, mask=init_mask, maxiters=5)

        # Stricter detection threshold for the mask
        detection_threshold = median_init + (2.0 * std_init) 
        segm = detect_sources(data, detection_threshold, npixels=5)

        segm_mask = (segm.data > 0) if segm else np.zeros_like(data, dtype=bool) 
        
        # 3. Final Combined Mask
        combined_mask = source_mask_large | segm_mask | np.isnan(data)
        
        # 3. 2D Plane Fit (Global Gradient)
        yi, xi = np.indices(data.shape)
        fit_mask = ~combined_mask   # fit_mask now includes all data used for the fit
        
        A = np.vstack([xi[fit_mask], yi[fit_mask], np.ones_like(xi[fit_mask])]).T
        z = data[fit_mask]
        
        if len(z) < 10: # Safety check
            coeffs = [0, 0, median_init] # Fallback to flat median
            print("Safety check failed, fallback")
        else:
            coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        
        alpha, beta, gamma = coeffs
        background_plane = alpha * xi + beta * yi + gamma
        data_bkgsub = data - background_plane

        # 4. Define elliptical annulus based on aperture size         
        
        
        # Dynamic outer radius based on image bounds to prevent crashes
        img_h, img_w = data.shape
        dist_to_edge = min(x, img_w - x, y, img_h - y)
        a_out = dist_to_edge - 2    # 2 pixel buffer at the image boundaries
        b_out = a_out * 0.9 # Maintain aspect ratio
        
        annulus = EllipticalAnnulus((x, y), a_in=a_in, a_out=a_out, 
                                    b_in=b_in, b_out=b_out, theta=theta)
        
        ann_mask = annulus.to_mask(method='center').to_image(data.shape).astype(bool)
        
        # Only use pixels that are in the annulus AND not a detected source
        bkg_pixels_mask = ann_mask & ~combined_mask
        
        # 5. Final Stats
        bkg_residuals = data_bkgsub[bkg_pixels_mask]
        
        # 2. Check for Skewness (= presence of artefacts)
        current_skew = skew(bkg_residuals)
        
        # Assign a flag level
        if current_skew > 5.0:
            skew_flag = "CRITICAL_SKEW"  # Likely an artifact or huge spike
            print(f"🔴 ID {galaxy_id} in {filt}: CRITICAL SKEW detected ({current_skew:.3f}). Artefact or huge spike present...")
        elif current_skew > 2.0:
            skew_flag = "HIGH_SKEW"      # Check for unmasked neighbours
            print(f"🟡 ID {galaxy_id} in {filt}: HIGH SKEW detected ({current_skew:.3f}). Check for unmasked neighbours...")
        else:
            skew_flag = "CLEAN"     
        
        if len(bkg_residuals) > 0:
            # Sigma clipping will throw away artefacts with very large skew
            # This identifies which pixels in the residuals were rejected
            clipped_array = sigma_clip(bkg_residuals, sigma=2.5, maxiters=5, cenfunc=np.median)
            # The '.mask' attribute is True where pixels were REJECTED
            clipped_mask_indices = clipped_array.mask

            # 2. Extract stats from the unclipped portions
            clean_median = np.ma.median(clipped_array)
            clean_std = np.ma.std(clipped_array)
            
            # 3. Update mask_vis to show WHERE the clipping happened
            # We need to map the 1D clipped_mask back to the 2D image
            clipped_map_2d = np.zeros_like(data, dtype=bool)
            # Use the same indices that defined bkg_pixels_mask
            clipped_map_2d[bkg_pixels_mask] = clipped_mask_indices
            
            background_median = clean_median
            background_std = clean_std * np.sqrt(source_ap.area)
            
        else:
            print("⚠️ No background residuals available, standard deviation and median defaulting to 0.")
            background_std, background_median = 0, 0
        
        # Initialise mask visualisation for plotting
        mask_vis = np.zeros_like(data, dtype=int)        
        mask_vis[~combined_mask] = 1 # All pixels used by the 2D plane fit
        mask_vis[bkg_pixels_mask] = 2  # Pixels in the annulus/rectangle
        mask_vis[source_mask_large] = 4  # Source pixels
        mask_vis[clipped_map_2d] = 3 # Rejected by additional sigma_clipping
        # All other pixels are 0 -> Excluded/NaN       
        
        aperture_map = {
            "x": float(aperture_params.get("x")),
            "y": float(aperture_params.get("y")),
            "a": float(aperture_params.get("a")),
            "b": float(aperture_params.get("b")),
            "theta": float(aperture_params.get("theta").value if hasattr(aperture_params.get("theta"), 'value') 
                   else aperture_params.get("theta")),
        }
        
        # Store visualisation data
        vis_data = {
            "galaxy_id": galaxy_id,
            "filter": filt,
            "original_data": data,
            "background_plane": background_plane,
            "background_subtracted": data_bkgsub,
            "mask_vis": mask_vis,
            "segmentation_mask": segm_mask,
            "background_region_mask": ann_mask,
            "region_name": "Annulus",
            "source_mask": source_mask_large,
            "aperture_params": aperture_map,
            "a_in": a_in,
            "b_in": b_in,
            "a_out": a_out,
            "b_out": b_out,
            "sigma": sigma_val,
            "coeffs": (alpha, beta, gamma),
        }
        
        # Save visualisation data to .h5 file
        vis_dir = os.path.join(self.output_dir, "vis_data")
        os.makedirs(vis_dir, exist_ok=True)

        vis_path = os.path.join(vis_dir, f"{galaxy_id}_{filt}.h5")
        self.save_vis(vis_data, vis_path)
        
        return {
            "id": galaxy_id,
            "filter": filt,
            "median": background_median,
            "std": background_std,
            "plane": background_plane,
            "subtracted": data_bkgsub,
            "annulus": annulus,
            "source_ap": source_ap,
            "mask_vis": mask_vis,
            "skew_flag": skew_flag
        }

    def plot_background_diagnostic(self, aperture_params, bkg_dict, save_path=None):
        """
        Creates a 2x2 diagnostic mosaic to verify background modeling.
        """
        data = aperture_params["data"]
        gid = aperture_params["id"]
        filt = aperture_params["meta"]["filter"]
        survey_obs = aperture_params["meta"]["survey_obs"]
        
        # Extract calculated objects from bkg_dict
        plane = bkg_dict["plane"]
        subtracted = bkg_dict["subtracted"]
        source_ap = bkg_dict["source_ap"]
        annulus = bkg_dict["annulus"]
        
        
        # Create the figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
        ax1, ax2, ax3, ax4 = axes.flatten()

        # --- 1. Original Data ---
        v1_min, v1_max = np.nanpercentile(data, [2, 98])
        im1 = ax1.imshow(data, origin="lower", cmap="magma", vmin=v1_min, vmax=v1_max)
        source_ap.plot(ax=ax1, color='dodgerblue', lw=2, label="Aperture")
        ax1.set_title(f"Original Data ({filt})")
        #plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # --- 2. 2D Background Plane ---
        v2_min, v2_max = np.nanpercentile(plane, [2, 98])
        im2 = ax2.imshow(plane, origin="lower", cmap="magma", vmin=v2_min, vmax=v2_max)
        ax2.set_title("Fitted Background Plane")
        #plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # --- 3. Subtracted Data + Annulus ---
        v3_min, v3_max = np.nanpercentile(subtracted, [5, 95])
        im3 = ax3.imshow(subtracted, origin="lower", cmap="magma", vmin=v3_min, vmax=v3_max)
        source_ap.plot(ax=ax3, color='dodgerblue', lw=2)
        #annulus.plot(ax=ax3, color='yellow', lw=1.5, ls='--', label="Annulus")
        ax3.set_title("Bkg-Subtracted")
        #ax3.legend(loc='upper right', fontsize=8)
        #plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # --- 4. Mask/Region Visualization ---
        mask_vis = bkg_dict["mask_vis"]
        
        cmap_mask = plt.get_cmap('viridis', 5)
        im4 = ax4.imshow(mask_vis, origin="lower", cmap=cmap_mask, vmin=-0.5, vmax=4.5)
        ax4.set_title("Region Classification")
        
        cbar = plt.colorbar(im4, ax=ax4, ticks=[0, 1, 2, 3, 4], fraction=0.046, pad=0.04)
        cbar.set_ticklabels(["Excluded/NaN", "Fit Region", "Annulus", "Clipped", "Source"])

        plt.suptitle(f"Background Model: Galaxy {gid} | {filt} | {survey_obs}", fontsize=14)
        
        # 1. Determine the destination
        if save_path is None:
            # Default organizational structure
            aperture_dir = os.path.join(self.output_dir, "mosaic_plots", filt)
            os.makedirs(aperture_dir, exist_ok=True)
            save_path = os.path.join(aperture_dir, f"{gid}_bkg.png")

        # 2. Save the figure (do this BEFORE close)
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        
    
    def plot_psf_aperture_review(self, psf_hdul, aperture_params, oversample=4):
        """
        Displays the oversampled PSF but re-scales the axes to 'Science Pixels'.
        This allows direct comparison with the science image aperture.
        """
        filter_name = aperture_params["meta"]["filter"]
        
        psf_data = psf_hdul[3].data
        
        # 1. Create a coordinate grid for the PSF in 'Science Pixel' units
        # If oversample is 4, a 256x256 oversampled array is 64x64 science pixels
        ny, nx = psf_data.shape
        # We center the axes so (0,0) is the PSF peak
        x_axis = (np.arange(nx) - nx/2) / oversample
        y_axis = (np.arange(ny) - ny/2) / oversample

        fig, ax = plt.subplots(figsize=(8, 8))

        # 2. Use 'extent' to force the imshow to display in Science Pixel units
        extent = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]
        
        im = ax.imshow(psf_data, origin='lower', extent=extent,
                    norm=LogNorm(vmin=psf_data.max()*1e-5, vmax=psf_data.max()),
                    cmap='magma')
        
        # 3. Use the ORIGINAL aperture (no multiplication by 4)
        # Since the axes are now in science pixels, the original 'a' and 'b' will match
        ap_pixel = EllipticalAperture(
            positions=(0, 0), # (0,0) because our 'extent' centered the image
            a=aperture_params["a"],
            b=aperture_params["b"],
            theta=aperture_params["theta"]
        )
        
        ap_pixel.plot(ax=ax, color='cyan', lw=2, ls='--')

        # 4. Formatting
        ax.set_xlabel("Offset from Center [Science Pixels]")
        ax.set_ylabel("Offset from Center [Science Pixels]")
        ax.set_title(f"MIRI {filter_name} PSF - Ext 3 (Oversampled x{oversample})")
        
        plt.colorbar(im, label='Normalised Intensity')
        plt.show()
        
            
    def calculate_psf_corr(self, aperture_params, oversample=4, show_plot=False):
        """
        Read MIRI PSF file for the specified filter and calculates 
        the aperture correction factor
        """
        filter_name = aperture_params["meta"]["filter"]
        
        psf_file = os.path.join(self.psf_dir, f"PSF_MIRI_{filter_name}.fits")
        with fits.open(psf_file) as psf_hdul:
            psf_data = psf_hdul[3].data  # Most realistic extension!
            
            if show_plot is True:
                self.plot_psf_aperture_review(psf_hdul, aperture_params)
        
        # 1. Normalise the PSF so the ENTIRE model equals 1.0
        current_sum = np.nansum(psf_data)

        # 2. Find centroid
        x_cen, y_cen = centroid_com(psf_data)

        # 3. Create the MATH aperture (scaled by oversample)
        aperture = EllipticalAperture(
            positions=(x_cen, y_cen),
            a=aperture_params["a"] * oversample,
            b=aperture_params["b"] * oversample,
            theta=aperture_params["theta"],
        )

        # 4. Calculate flux in aperture
        # Since the total sum is 1.0, 'flux_in_aperture' is the Encircled Energy (EE)
        phot_table = aperture_photometry(psf_data, aperture, method="exact")
        ee_fraction = phot_table["aperture_sum"][0]
        
        # 5. The Correction Factor
        # If ee_fraction is 0.9 (90%), correction is 1.11
        correction_factor = current_sum / ee_fraction

        return correction_factor
        
        
    
    @staticmethod  
    def save_vis(vis_data, filename):
        """
        Save visualisation data to HDF5 file.

        Parameters:
        -----------
        vis_data : dict
            Dictionary containing visualization data
        filename : str
            Output filename (should end with .h5 or .hdf5)
        """
        with h5py.File(filename, "w") as f:
            # Save arrays with compression
            for key in [
                "original_data",
                "background_plane",
                "background_subtracted",
                "mask_vis",
                "segmentation_mask",
                "background_region_mask",
                "source_mask",
            ]:
                if key in vis_data and vis_data[key] is not None:
                    f.create_dataset(
                        key, data=vis_data[key], compression="gzip", compression_opts=6
                    )

            # Save scalars
            for key in ["galaxy_id", "a_in", "b_in", "a_out", "b_out", "sigma"]:
                if key in vis_data and vis_data[key] is not None:
                    f.attrs[key] = vis_data[key]

            # Save strings
            for key in ["filter", "region_name"]:
                if key in vis_data and vis_data[key] is not None:
                    f.attrs[key] = (
                        vis_data[key].encode("utf-8")
                        if isinstance(vis_data[key], str)
                        else vis_data[key]
                    )

            # Save coefficients tuple
            if "coeffs" in vis_data and vis_data["coeffs"] is not None:
                f.create_dataset("coeffs", data=np.array(vis_data["coeffs"]))

            # Save aperture_params dict as JSON string
            if "aperture_params" in vis_data and vis_data["aperture_params"] is not None:
                f.attrs["aperture_params"] = json.dumps(vis_data["aperture_params"])

            # Add metadata
            f.attrs["created_date"] = str(np.datetime64("now"))
            f.attrs["data_type"] = "galaxy_visualisation_data"
        
    @staticmethod
    def load_vis(filename):
        """
        Load visualisation data from HDF5 file and reconstruct Photutils objects.
        """
        vis_data = {}

        with h5py.File(filename, "r") as f:
            # 1. Load Arrays
            for key in [
                "original_data",
                "background_plane",
                "background_subtracted",
                "mask_vis",
                "segmentation_mask",
                "background_region_mask",
                "source_mask",
            ]:
                if key in f:
                    vis_data[key] = f[key][:]

            # 2. Load coefficients
            if "coeffs" in f:
                vis_data["coeffs"] = tuple(f["coeffs"][:])

            # 3. Load scalars from attributes
            for key in ["galaxy_id", "a_in", "b_in", "a_out", "b_out", "sigma"]:
                if key in f.attrs:
                    vis_data[key] = f.attrs[key]

            # 4. Load strings and decode if necessary
            for key in ["filter", "region_name"]:
                if key in f.attrs:
                    val = f.attrs[key]
                    vis_data[key] = val.decode("utf-8") if isinstance(val, bytes) else val

            # 5. Load aperture_params dict
            if "aperture_params" in f.attrs:
                vis_data["aperture_params"] = json.loads(f.attrs["aperture_params"])

            # --- RECONSTRUCTION STEP ---
            ap = vis_data.get("aperture_params")
            if ap:
                # Reconstruct the Source Aperture
                # Note: ap['x'], ap['y'] etc. are now plain floats from JSON
                vis_data['source_ap'] = EllipticalAperture(
                    positions=(ap['x'], ap['y']),
                    a=ap['a'],
                    b=ap['b'],
                    theta=ap['theta']
                )

                # Reconstruct the Annulus
                # Uses the a_in, b_out etc stored in the main vis_data dict
                vis_data['annulus'] = EllipticalAnnulus(
                    positions=(ap['x'], ap['y']),
                    a_in=vis_data['a_in'],
                    a_out=vis_data['a_out'],
                    b_in=vis_data['b_in'],
                    b_out=vis_data['b_out'],
                    theta=ap['theta']
                )

        return vis_data

    def measure_flux(self, aperture_params, bkg_results):
        """
        Performs aperture photometry, unit conversion, and error propagation.
        """
        # Extract data from dictionaries
        data = aperture_params["data"]
        err_map = aperture_params["err"] 
        err_map = np.nan_to_num(err_map, nan=0.0, posinf=0.0, neginf=0.0)
        pixel_area_sr = aperture_params["meta"]["pixel_area_sr"] # From PIXAR_SR header
        
        # Get background-subtracted data
        data_bkgsub = bkg_results["subtracted"]
        source_ap = bkg_results["source_ap"]
        
        # Perform photometry
        phot_table = aperture_photometry(data_bkgsub, source_ap, method='exact')
        raw_flux_mjysr = phot_table['aperture_sum'][0]
        
        # 4. Error Propagation
        # A. Detector/Poisson noise from the ERR extension
        ap_mask = source_ap.to_mask(method='exact')
        # Weighted sum of variance (method='exact' accounts for fractional pixels)
        detector_variance = np.nansum(ap_mask.multiply(err_map**2))
        
        # B. Background modelling uncertainty already calculated in estimate_background
        # bkg_std = clean_std * np.sqrt(source_ap.area)
        bkg_err_mjysr = bkg_results["std"]
        
        # C. Combine in quadrature
        total_err_mjysr = np.sqrt(detector_variance + bkg_err_mjysr**2)
        
        # 5. Unit Conversion (MJy/sr -> Jy)
        # Conversion = 1e6 (to Jy) * pixel_area_sr (to strip sr)
        conv = 1e6 * pixel_area_sr
        flux_jy = raw_flux_mjysr * conv
        err_jy = total_err_mjysr * conv
        
        # This is the "Nominal" error in Jy
        nominal_err_jy = np.sqrt(detector_variance) * conv
        
        # Obtain local background and error
        bkg_median_jy = bkg_results["median"] * conv
        bkg_err_jy = bkg_err_mjysr * conv
        
        # 6. Final Results Dictionary
        return {
            "flux_jy": flux_jy,
            "flux_err_jy": err_jy,
            "snr": flux_jy / err_jy if err_jy > 0 else 0,
            "area_pix": source_ap.area,
            "bkg_median_jy": bkg_median_jy,
            "bkg_err_jy": bkg_err_jy,
            "nominal_err_jy": nominal_err_jy,
            "skew_flag": bkg_results["skew_flag"]
        }
    
    def get_filter_column_template(self, filt):
        """Defines the standard set of columns for any single MIRI band."""
        return {
            f"{filt}_flux": np.nan,
            f"{filt}_flux_err": np.nan,
            f"{filt}_abmag": np.nan,
            f"{filt}_apflux": np.nan,
            f"{filt}_apflux_err": np.nan,
            f"{filt}_apflux_errnominal": np.nan,
            f"{filt}_apcorr": np.nan,
            f"{filt}_bkg": np.nan,
            f"{filt}_bkg_err": np.nan,
            f"{filt}_ap_x": np.nan,
            f"{filt}_ap_y": np.nan,
            f"{filt}_ap_theta": np.nan,
            f"{filt}_flag_art": False,  # Boolean default
        }

    def pre_scan_filters(self):
        """
        Crawls the cutouts directory to identify all MIRI filters available in the dataset.
        """
        root = Path(self.cutouts_dir)
        # This looks for the filter folder name (e.g., .../primer1/F770W/fits/)
        # Adjust the glob pattern if your folder structure is different
        filter_dirs = root.glob("*/*/fits")
        
        found_filters = {p.parent.name.upper() for p in filter_dirs}
        
        # Sort them by wavelength using your existing wavelength_map
        sorted_filters = sorted(
            list(found_filters), 
            key=lambda k: self.wavelength_map.get(k, 99.0)
        )
        
        return sorted_filters

    def run_photometry(self, write_to, rescale=True, plot_mosaics=False, plot_psf=False):
        """
        Function to do the heavy lifting. Runs the entire photometry
        """
        
        # Stylized ASCII Header
        print("\n" + "="*60)
        print("""
 ____           _    ____              _ _             _ 
|  _ \ ___   __| |  / ___|__ _ _ __ __| (_)_ __   __ _| |
| |_) / _ \ / _` | | |   / _` | '__/ _` | | '_ \ / _` | |
|  _ <  __/| (_| | | |__| (_| | | | (_| | | | | | (_| | |
|_| \_\___| \__,_|  \____\__,_|_|  \__,_|_|_| |_|\__,_|_|
        """)
        print("                 JWST MIRI PIPELINE v3.0")
        print("                 MIRI Photometry for JWST")
        print("="*60)
        
        # Pre-scan and visual summary
        all_filters = self.pre_scan_filters()
        
        print(f"\n[INFO] Survey Discovery:")
        print(f"  > Found {len(all_filters)} MIRI bands: {', '.join(all_filters)}")
        print(f"  > Target Galaxy Count: {len(self.all_ids)}")
        print(f"  > Initialising 'Wide' table structure...")
        
        # Simple progress bar visualisation for the initialisation
        print("\nPreparing Columns: [", end="")
        for i in range(20):
            time.sleep(0.02) # Just for aesthetic effect
            print("■", end="", flush=True)
        print("] 100%\n")

        if rescale == False:
            print("⚠️ Processing photometry with original aperture sizes based on NIRCam/F444W...")
        
        all_rows = []
        
        for target_id in self.all_ids:
            if target_id in self.quality_config["exclude_all"]:
                continue
                
            # Base identity for the galaxy
            galaxy_row = {
                "ID": target_id,
                "MIRI_ap_a": np.nan,
                "MIRI_ap_b": np.nan,
                "MIRI_ap_npix": np.nan,
                "Flag_Com": target_id in self.quality_config["has_companion"]
            }
            
            # Pre-populate with columns for all filters ALREADY discovered
            for filt in all_filters:
                galaxy_row.update(self.get_filter_column_template(filt))
            
            files = self.find_files(target_id)
            if not files: 
                continue
            
            # Track if we've stored general aperture yet
            ap_geometry_stored = False
            files = self.find_files(target_id)
            
            for filt, file in files.items():
                if target_id in self.quality_config["exclude_filters"].get(filt, []):
                    continue
                
                try:
                    # Perform the photometry steps:
                    
                    # 1. Prepare the apertures for MIRI
                    ap_params = self.prepare_aperture(file, rescale=rescale)
                    
                    # 2. Create background model
                    bkg_res = self.estimate_background(ap_params)
                    
                    if plot_mosaics is True:
                        self.plot_background_diagnostic(ap_params, bkg_res)
                    
                    # 3. Measure fluxes
                    measurements = self.measure_flux(ap_params, bkg_res)
                    
                    # 4. Compute PSF correction
                    psf_corr = self.calculate_psf_corr(ap_params, show_plot=plot_psf)
                    
                    # Get (uncorrected) aperture fluxes
                    apflux = measurements["flux_jy"]                    
                    apflux_err = measurements["flux_err_jy"]
                    
                    # Get PSF-corrected fluxes
                    flux_corr = measurements["flux_jy"] * psf_corr 
                    flux_err_corr = measurements["flux_err_jy"] * psf_corr
                    
                    # --- Convert fluxes into AB magnitudes ---
                    if flux_corr > 0:
                        # constant is 8.90 for Jy and 23.90 for µJy
                        ab_mag = -2.5 * np.log10(flux_corr) + 8.90
                    else:
                        ab_mag = np.nan

                    
                    # Get nominal flux error (from ERR extension)
                    apflux_errnominal = measurements["nominal_err_jy"]
                    
                    # Get local bkg estimates (median + error)
                    n_pix = measurements["area_pix"]
                    bkg_median_jy = measurements["bkg_median_jy"]
                    local_bkg = bkg_median_jy * n_pix
                    bkg_err = measurements["bkg_err_jy"]
                    
                    # --- Fill Filter-Specific Columns ---
                    galaxy_row[f"{filt}_flux"] = flux_corr
                    galaxy_row[f"{filt}_flux_err"] = flux_err_corr
                    galaxy_row[f"{filt}_abmag"] = ab_mag
                    galaxy_row[f"{filt}_apflux"] = apflux
                    galaxy_row[f"{filt}_apflux_err"] = apflux_err
                    galaxy_row[f"{filt}_apflux_errnominal"] = apflux_errnominal
                    galaxy_row[f"{filt}_apcorr"] = psf_corr
                    galaxy_row[f"{filt}_bkg"] = local_bkg
                    galaxy_row[f"{filt}_bkg_err"] = bkg_err
                    
                    # Handle the "varying" parameters per band
                    galaxy_row[f"{filt}_ap_theta"] = float(np.degrees(ap_params["theta"].value))
                    galaxy_row[f"{filt}_ap_x"] = ap_params["x"].item()
                    galaxy_row[f"{filt}_ap_y"] = ap_params["y"].item()
                    
                    # This returns exactly True or False
                    is_artifact = target_id in self.quality_config["art_filters"].get(filt, [])
                    galaxy_row[f"{filt}_flag_art"] = target_id in self.quality_config["art_filters"].get(filt, [])
                    
                    # Store "Scalar" values once
                    if not ap_geometry_stored:
                        galaxy_row["MIRI_ap_a"] = ap_params["a"]
                        galaxy_row["MIRI_ap_b"] = ap_params["b"]
                        galaxy_row["MIRI_ap_npix"] = n_pix
                        ap_geometry_stored = True
                    
                except Exception as e:
                    print(f"Error processing {target_id} in {filt}: {e}")
            
            # Only add the row if we actually measured something
            if len(galaxy_row) > 1:
                all_rows.append(galaxy_row)

        # Convert to DataFrame and save to file
        df = pd.DataFrame(all_rows)
        
        table_dir = os.path.join(self.output_dir, "phot_tables")
        os.makedirs(table_dir, exist_ok=True)
        
        phot_table_path = os.path.join(table_dir, write_to)
        self.save_catalogue(df, phot_table_path)
        
                    
    def save_catalogue(self, df, base_filename):
        """Saves the result in both FITS (Science) and CSV (Human-Readable)."""
        # Save CSV
        csv_path = f"{base_filename}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save FITS
        fits_path = f"{base_filename}.fits"
        table = Table.from_pandas(df)
        
        for col_name in table.colnames:
            # We only want to mask numerical data (Fluxes, Mags, etc.)
            if any(key in col_name for key in ['flux', 'abmag', 'bkg', 'err']):
                data = table[col_name].data
                
                # Create a mask where values are NaN
                mask = np.isnan(data)
                
                # Replace the column with a MaskedColumn
                table[col_name] = MaskedColumn(data, name=col_name, mask=mask, fill_value=np.nan)
            
        table.write(fits_path, format='fits', overwrite=True)
        
        print(f"Catalog exported to:\n1. {csv_path}\n2. {fits_path}")















def create_mosaics(input_dir, mosaic_dir=None, plane_sub_dir=None):
    all_files = glob.glob(os.path.join(input_dir, "*.h5"))
    all_ids = np.unique([os.path.basename(f).split("_")[0] for f in all_files])

    def plot_aperture_overlay(
        ax, data, aperture, cmap="magma", label="", percentile=(5, 95)
    ):
        vmin, vmax = np.nanpercentile(data, percentile)
        im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        # aperture.plot(ax=ax, color='blue', lw=4)
        ax.set_title(label, fontsize=10)
        return im

    if plane_sub_dir:
        os.makedirs(plane_sub_dir, exist_ok=True)

        for file in all_files:
            vis = load_vis(file)
            image_data = vis["original_data"]
            background_plane = vis["background_plane"]
            background_subtracted = vis["background_subtracted"]
            aperture_params = vis["aperture_params"]
            filter = vis["filter"]
            galaxy_id = vis["galaxy_id"]

            if str(galaxy_id) in ["12282", "10128"]:
                aperture = EllipticalAperture(
                    positions=(
                        aperture_params["x_center"],
                        aperture_params["y_center"],
                    ),
                    a=aperture_params["a"],
                    b=aperture_params["b"],
                    theta=aperture_params["theta"],
                )

                # Create figure with three subplots in a horizontal row
                fig, axes = plt.subplots(1, 3, figsize=(14, 4))

                # Plot image1 (original data) on the first subplot
                im0 = plot_aperture_overlay(axes[0], image_data, aperture)
                axes[0].set_title("Original Data", fontsize=14)

                # Plot image2 (background plane) on the second subplot
                im1 = plot_aperture_overlay(axes[1], background_plane, aperture)
                axes[1].set_title("Background Plane", fontsize=14)

                # Plot image3 (background-subtracted) on the third subplot
                im2 = plot_aperture_overlay(axes[2], background_subtracted, aperture)
                axes[2].set_title("Background-Subtracted Data", fontsize=14)

                # Add a minus and equals sign between the images as an annotation
                # fig.text(0.32, 0.45, '$-$', fontsize=30, ha='center', va='center', rotation=0, color='black')
                # fig.text(0.66, 0.45, '$=$', fontsize=30, ha='center', va='center', rotation=0, color='black')

                # Add colorbars
                """
                for ax, im, label in zip(axes, [im0, im1, im2], [
                    'Flux [MJy/(sr pixel)]',
                    'Background Flux [MJy/(sr pixel)]',
                    'Background-subtracted Flux [MJy/(sr pixel)]'
                ]):
                    plt.colorbar(im, ax=ax)#, label=label)
                """
                plt.subplots_adjust(
                    wspace=5
                )  # This will increase the space between the subplots

                # Tight layout and saving the figure
                plt.tight_layout()
                plt.subplots_adjust(
                    top=0.85
                )  # Adjust to prevent overlap with annotation
                # plt.suptitle(f'{filter} - Galaxy ID {galaxy_id}', fontsize=18)
                plt.savefig(
                    os.path.join(plane_sub_dir, f"{galaxy_id}_{filter}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)
                print(
                    f"Saved plane subtraction figure for Galaxy {galaxy_id}, Filter {filter} in {plane_sub_dir}"
                )

    if mosaic_dir:
        os.makedirs(mosaic_dir, exist_ok=True)

        for gid in all_ids:
            print(f"Processing galaxy {gid}...")
            vis_files = glob.glob(os.path.join(input_dir, f"{gid}*.h5"))
            vis_list = [load_vis(f) for f in vis_files]

            filter_order = ["F770W", "F1000W", "F1800W", "F2100W"]
            vis_dict = {v["filter"]: v for v in vis_list}
            vis_sorted = [vis_dict[f] for f in filter_order if f in vis_dict]

            num = len(vis_sorted)
            fig, axes = plt.subplots(3, num, figsize=(4 * num, 12))

            if num == 1:
                axes = np.expand_dims(axes, axis=1)

            for ii, vis in enumerate(vis_sorted):
                ap_params = vis["aperture_params"]
                aperture = EllipticalAperture(
                    positions=(ap_params["x_center"], ap_params["y_center"]),
                    a=ap_params["a"],
                    b=ap_params["b"],
                    theta=ap_params["theta"],
                )

                # Top: original + aperture
                plot_aperture_overlay(
                    axes[0, ii], vis["original_data"], aperture, label=vis["filter"]
                )

                # Middle: background plane
                axes[1, ii].imshow(
                    vis["background_plane"], origin="lower", cmap="viridis"
                )
                axes[1, ii].set_title("Background")

                # Bottom: mask visualisation
                cmap = plt.cm.get_cmap("viridis", 4)
                axes[2, ii].imshow(
                    vis["mask_vis"], origin="lower", cmap=cmap, vmin=-0.5, vmax=3.5
                )
                axes[2, ii].set_title("Mask")

            plt.tight_layout()
            plt.savefig(os.path.join(mosaic_dir, f"{gid}.png"), dpi=150)
            plt.close(fig)


def visualise_background(vis_data, fig_path=None):
    """
    Create visualisations from the background estimation data.

    Parameters
    ----------
    vis_data : dict
        Dictionary containing all data needed for visualisation
    fig_path : str, optional
        Path to save the visualisation figure
    """
    # Extract data from the dictionary
    image_data = vis_data["original_data"]
    background_plane = vis_data["background_plane"]
    background_subtracted = vis_data["background_subtracted"]
    segm_mask = vis_data["segmentation_mask"]
    mask_vis = vis_data["mask_vis"]
    aperture_params = vis_data["aperture_params"]
    sigma = vis_data["sigma"]
    region_name = vis_data["region_name"]
    galaxy_id = vis_data["galaxy_id"]
    filter = vis_data["filter"]

    # Create aperture objects for plotting
    x_center = aperture_params["x_center"]
    y_center = aperture_params["y_center"]
    a = aperture_params["a"]
    b = aperture_params["b"]
    theta = aperture_params["theta"]

    source_aperture = EllipticalAperture(
        positions=(x_center, y_center), a=a, b=b, theta=theta
    )

    # Create visualisations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Original data with aperture
    vmin = np.nanpercentile(image_data, 5)
    vmax = np.nanpercentile(image_data, 95)

    im0 = axes[0, 0].imshow(
        image_data, origin="lower", cmap="magma", vmin=vmin, vmax=vmax
    )
    plt.colorbar(im0, ax=axes[0, 0], label="Flux [MJy/(sr pixel)]")

    # Plot the source aperture
    source_aperture.plot(ax=axes[0, 0], color="blue", lw=4)

    # Overlay the segmentation mask as white outlines
    axes[0, 0].contour(segm_mask, levels=[0.5], colors="green", linewidths=2.5)

    axes[0, 0].set_title("Original Data with Aperture and Masked Regions")

    # Background-subtracted data
    vmin2 = np.nanpercentile(background_subtracted, 5)
    vmax2 = np.nanpercentile(background_subtracted, 95)

    im1 = axes[0, 1].imshow(
        background_subtracted, origin="lower", cmap="magma", vmin=vmin2, vmax=vmax2
    )
    plt.colorbar(
        im1, ax=axes[0, 1], label="Background-subtracted Flux [MJy/(sr pixel)]"
    )
    source_aperture.plot(ax=axes[0, 1], color="blue", lw=4)
    axes[0, 1].set_title("Background-subtracted Data with Aperture")

    # Global 2D background plane
    im2 = axes[1, 0].imshow(background_plane, origin="lower", cmap="viridis")
    plt.colorbar(im2, ax=axes[1, 0], label="Background Flux [MJy/(sr pixel)")
    axes[1, 0].set_title("Global 2D Background Plane")

    # Mask visualisation
    cmap = plt.cm.get_cmap("viridis", 4)
    im3 = axes[1, 1].imshow(mask_vis, origin="lower", cmap=cmap, vmin=-0.5, vmax=3.5)
    cbar = plt.colorbar(im3, ax=axes[1, 1], ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(
        [
            f"Excluded\n(σ={sigma})",
            "Used for fitting",
            f"{region_name} region",
            "Source",
        ]
    )
    axes[1, 1].set_title("Pixel Masks")

    fig.suptitle(f"{filter}", fontsize=18)  # , fontweight='bold')
    plt.tight_layout()

    if fig_path:
        os.makedirs(fig_path, exist_ok=True)
        filepath = os.path.join(fig_path, f"{galaxy_id}_{filter}.png")
        plt.savefig(filepath, dpi=150)
        plt.close(fig)





def get_aperture_params(galaxy_id, filter, aperture_table):
    """
    Retrieve aperture parameters from the CSV table.

    Parameters
    ----------
    galaxy_id : str
        ID of the galaxy
    aperture_table : str
        Path to CSV table with aperture parameters

    Returns
    -------
    dict
        Dictionary with aperture parameters
    """
    df = pd.read_csv(aperture_table)

    # Look for the unique combination of ID and filter
    row = df[(df["ID"] == int(galaxy_id)) & (df["Filter"] == filter)].iloc[0]

    return {
        "x_center": row["Apr_Xcenter"],
        "y_center": row["Apr_Ycenter"],
        "a": row["Apr_A"] / 2,  # Converting diameter to major axis length
        "b": row["Apr_B"] / 2,  # Converting diameter to minor axis length
        "theta": (row["Apr_Theta"] * u.deg).to_value(u.rad),  # Convert to radians
    }




def write_detection_stats(table_path, stats_path=None, nondetections=None):
    """
    Summarise galaxy detection statistics per filter and overall.

    Parameters:
    -----------
    table_path : str
        Path to the FITS table.
    stats_path : str (optional)
        Path to the output statistics file.
    nondetections : dict (optional)
        Dictionary mapping each filter to a list of galaxy IDs that are NOT detected.
    """
    table = Table.read(table_path, format="fits")

    if "Filters" not in table.colnames or "ID" not in table.colnames:
        raise ValueError("FITS table must contain 'Filters' and 'ID' columns.")

    total_galaxies = 153  # Total number of galaxies in the Blue Jay sample

    # Build per-filter detection counts
    filter_detection_counts = {}
    filter_imaged_counts = {}
    detected_in_any = set()
    imaged_in_any = set()

    for row in table:
        filters = [f.strip() for f in row["Filters"].split(",") if f.strip()]
        gid = str(row["ID"])

        if filters:
            imaged_in_any.add(gid)

        for filt in filters:
            filter_imaged_counts[filt] = filter_imaged_counts.get(filt, 0) + 1

            if nondetections is None or int(gid) not in nondetections.get(filt, []):
                filter_detection_counts[filt] = filter_detection_counts.get(filt, 0) + 1
                detected_in_any.add(gid)

    # Write summary
    with open(stats_path, "w") as f:
        f.write("Galaxy Imaging Summary per Filter\n")
        f.write("=" * 40 + "\n")
        for filt, count in filter_imaged_counts.items():
            f.write(
                f"{filt:10s}: {count} / {total_galaxies} galaxies "
                f"({(count / total_galaxies) * 100:.1f}%) imaged\n"
            )

        f.write(
            f"\nImaged in at least one filter: {len(imaged_in_any)} / {total_galaxies} galaxies "
            f"({(len(imaged_in_any) / total_galaxies) * 100:.1f}%)\n"
        )

        f.write("\nGalaxy Detection Summary per Filter\n")
        f.write("=" * 40 + "\n")
        for filt, count in filter_detection_counts.items():
            f.write(
                f"{filt:10s}: {count} / {filter_imaged_counts.get(filt, 0)} galaxies "
                f"({(count / filter_imaged_counts.get(filt, 1)) * 100:.1f}%) detected\n"
            )

        f.write(
            f"\nDetected in at least one filter: {len(detected_in_any)} / {len(imaged_in_any)} galaxies "
            f"({(len(detected_in_any) / len(imaged_in_any)) * 100:.1f}%)\n"
        )

    print(f"Wrote galaxy statistics to {stats_path}")


def plot_galaxy_filter_matrix(
    table_path, fig_path, title=None, nondetections=None, cols=4
):
    """
    Visualise which galaxies are observed and detected in which filters,
    but using a dictionary of *non-detections* instead of detections.

    Parameters:
    -----------
    table_path : str
        Path to the FITS file.
    fig_path : str
        Path to the output file.
    title : str, optional
        Title of the plot.
    nondetections : dict, optional
        Dictionary mapping filter names to lists of galaxy IDs that were NOT detected.
    cols : int
        Number of subplot columns.
    """
    table = Table.read(table_path, format="fits")
    table.info()
    filter_order = ["F770W", "F1000W", "F1800W", "F2100W"]
    pastel_colours = {
        "F770W": "#a6cee3",
        "F1000W": "#b2df8a",
        "F1800W": "#fdbf6f",
        "F2100W": "#fb9a99",
    }

    galaxy_ids = [str(gid) for gid in table["ID"]]
    num_galaxies = len(galaxy_ids)
    chunk_size = (num_galaxies + 3) // cols
    chunks = [
        galaxy_ids[i : i + chunk_size] for i in range(0, num_galaxies, chunk_size)
    ]

    print(f"Number of unique IDs in table: {len(set(str(row['ID']) for row in table))}")
    print(f"Number of IDs in galaxy_ids: {len(galaxy_ids)}")
    print(f"Chunks: {[len(c) for c in chunks]}")

    cell_size = 0.5
    num_cols = len(filter_order)
    num_rows = chunk_size
    fig_width = cell_size * num_cols * cols
    fig_height = cell_size * num_rows * 0.7

    fig, axes = plt.subplots(1, cols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes[0]

    for ax, g_ids in zip(axes, chunks):
        matrix = np.zeros((len(g_ids), len(filter_order)), dtype=int)
        g_index_map = {gid: i for i, gid in enumerate(g_ids)}
        table_id_to_row = {str(row["ID"]): idx for idx, row in enumerate(table)}

        for row in table:
            gid = str(row["ID"])
            if gid not in g_index_map:
                continue
            g_idx = g_index_map[gid]
            filters = row["Filters"]
            if isinstance(filters, (list, np.ndarray)):
                filters = [
                    f.decode() if isinstance(f, bytes) else str(f) for f in filters
                ]
            else:
                filters = [f.strip() for f in str(filters).split(",") if f.strip()]

            for filt in filters:
                if filt in filter_order:
                    f_idx = filter_order.index(filt)

                    # Inverted logic:
                    # Galaxy is marked if it's covered AND not in nondetections for that filter
                    if nondetections is None or int(gid) not in nondetections.get(
                        filt, []
                    ):
                        matrix[g_idx, f_idx] = 1

        # Draw rectangles
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] == 1:
                    base_colour = pastel_colours[filter_order[j]]
                    gid = g_ids[i]
                    row = table[table_id_to_row[gid]]

                    flag_art_array = row["Flag_Art"]
                    flag_art = False
                    if flag_art_array is not None and len(flag_art_array) == len(
                        filter_order
                    ):
                        flag_art = flag_art_array[j]

                    if flag_art:
                        rgb = np.array(mcolors.to_rgb(base_colour))
                        darker_rgb = np.clip(rgb * 0.7, 0, 1)
                        colour = darker_rgb
                    else:
                        colour = base_colour

                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color=colour))

        # Labels with asterisk for companions
        y_labels = []
        for i, gid in enumerate(g_ids):
            row = table[table_id_to_row[gid]]
            label = gid
            if row["Flag_Com"] == True:
                label += "*"
            y_labels.append(label)

        ax.set_xlim(0, len(filter_order))
        ax.set_ylim(len(g_ids), 0)
        ax.set_xticks(np.arange(len(filter_order)) + 0.5)
        ax.set_xticklabels(filter_order, rotation=45, ha="right", fontsize=11)
        ax.set_yticks(np.arange(len(g_ids)) + 0.5)
        ax.set_yticklabels(y_labels, fontsize=11)

        # Add horizontal grid lines
        for y in np.arange(len(g_ids)):
            ax.axhline(
                y=y, color="grey", linestyle="-", linewidth=0.3, alpha=0.5, zorder=10
            )

        # Vertical lines at column boundaries
        for x in np.arange(len(filter_order) + 1):
            ax.axvline(
                x=x, color="grey", linestyle="-", linewidth=0.4, alpha=0.6, zorder=10
            )

        print(f"Plotting {len(g_ids)} galaxies in this panel")

    total_plotted = sum(len(c) for c in chunks)
    print(f"Total plotted galaxies: {total_plotted}")

    # plt.suptitle(title, fontsize=28)
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.show()


def compare_aperture_statistics(
    table_small_path,
    table_big_path,
    fig_path=None,
    summary_doc_path=None,
    non_detections=None,
    scaling=None,
):
    """
    Compare and contrast two photometric tables WITHOUT APERTURE CORRECTION APPLIED
    and create a comprehensive summary plot of all important statistics and write
    the output to a text file.

    Args:
        table_small_path (str):
            Path to table using small apertures
        table_big_path (str):
            Path to table using big apertures
        fig_path (str):
            Output path of the summary plot
        summary_doc_path (str):
            Output path of the summary text file
        scaling (str) optional:
            'log' for logarithmic, default is linear
    """
    # Enhanced Aperture Photometry Comparison
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from astropy.table import Table

    # Set style for better plots
    plt.style.use("default")
    sns.set_palette("husl")

    table_small = Table.read(table_small_path)
    table_big = Table.read(table_big_path)

    # Convert ID columns to string for alignment
    ids_small = [
        id.decode() if isinstance(id, bytes) else str(id) for id in table_small["ID"]
    ]
    ids_big = [
        id.decode() if isinstance(id, bytes) else str(id) for id in table_big["ID"]
    ]

    # Match common IDs
    common_ids = sorted(set(ids_small) & set(ids_big))
    print(f"Found {len(common_ids)} common galaxies")

    # Prepare data structures
    bands = ["F770W", "F1000W", "F1800W", "F2100W"]
    data_comparison = {
        "ID": [],
        "Band": [],
        "Flux_Small_Raw": [],
        "Flux_Big_Raw": [],
        "Flux_Err_Small_Raw": [],
        "Flux_Big_Raw_Err": [],
        "Flux_Small_Corrected": [],
        "Flux_Big_Corrected": [],
        "Flux_Err_Small_Corrected": [],
        "Flux_Big_Corrected_Err": [],
        "Apr_Corr_Small": [],
        "Apr_Corr_Big": [],
        "Flux_Ratio": [],
        "Corrected_Flux_Ratio": [],
        "Flux_Difference": [],
        "Corrected_Flux_Difference": [],
    }

    # Collect all data for comprehensive analysis
    for idx, band in enumerate(
        bands
    ):  # bands = ["F770W", "F1000W", "F1800W", "F2100W"]
        for gid in common_ids:
            index_s = ids_small.index(gid)
            index_b = ids_big.index(gid)

            # Raw fluxes (convert to µJy)
            flux_small = table_small["Flux"][index_s][idx] * 1e6
            flux_big = table_big["Flux"][index_b][idx] * 1e6
            flux_err_small = table_small["Flux_Err"][index_s][idx] * 1e6
            flux_err_big = table_big["Flux_Err"][index_b][idx] * 1e6

            # Aperture corrections
            corr_small = (
                table_small["Apr_Corr"][index_s][idx]
                if "Apr_Corr" in table_small.colnames
                else np.nan
            )
            corr_big = (
                table_big["Apr_Corr"][index_b][idx]
                if "Apr_Corr" in table_big.colnames
                else np.nan
            )

            # Skip if any crucial value is invalid
            if not (
                np.isfinite(flux_small)
                and np.isfinite(flux_big)
                and (flux_small > 0)
                and (flux_big > 0)
                and np.isfinite(flux_err_small)
                and np.isfinite(flux_err_big)
                and np.isfinite(corr_small)
                and np.isfinite(corr_big)
            ):
                continue

            # Calculate corrected fluxes
            flux_small_corr = flux_small * corr_small
            flux_big_corr = flux_big * corr_big
            flux_err_small_corr = flux_err_small * corr_small
            flux_err_big_corr = flux_err_big * corr_big

            # Store all data
            data_comparison["ID"].append(gid)
            data_comparison["Band"].append(band)
            data_comparison["Flux_Small_Raw"].append(flux_small)
            data_comparison["Flux_Big_Raw"].append(flux_big)
            data_comparison["Flux_Err_Small_Raw"].append(flux_err_small)
            data_comparison["Flux_Big_Raw_Err"].append(flux_err_big)
            data_comparison["Flux_Small_Corrected"].append(flux_small_corr)
            data_comparison["Flux_Big_Corrected"].append(flux_big_corr)
            data_comparison["Flux_Err_Small_Corrected"].append(flux_err_small_corr)
            data_comparison["Flux_Big_Corrected_Err"].append(flux_err_big_corr)
            data_comparison["Apr_Corr_Small"].append(corr_small)
            data_comparison["Apr_Corr_Big"].append(corr_big)
            data_comparison["Flux_Ratio"].append(flux_big / flux_small)
            data_comparison["Corrected_Flux_Ratio"].append(
                flux_big_corr / flux_small_corr
            )
            data_comparison["Flux_Difference"].append(flux_big - flux_small)
            data_comparison["Corrected_Flux_Difference"].append(
                flux_big_corr - flux_small_corr
            )

    filename = os.path.join(
        "/Users/benjamincollins/University/Master/Red_Cardinal/photometry/apertures/aperture_comparisons/comparison_data.pkl"
    )

    # Write output to a pickle file
    with open(filename, "wb") as f:
        pkl.dump(data_comparison, f)
        print(f"Saved pickle file to {filename}")

    if fig_path:
        plot_aperture_comparison(data_comparison, fig_path, scaling)
        print(f"Saved output plot to {fig_path}")

    if summary_doc_path:
        write_aperture_summary(
            data_comparison, common_ids, summary_doc_path, non_detections=non_detections
        )


def plot_aperture_comparison(data_comparison, fig_path, scaling=None):
    # Convert to arrays for easier handling
    for key in data_comparison:
        data_comparison[key] = np.array(data_comparison[key])

    # Create comprehensive comparison plots
    fig = plt.figure(figsize=(20, 16))

    bands = ["F770W", "F1000W", "F1800W", "F2100W"]

    # 1. Raw vs Corrected Flux Comparison (Scatter plots)
    for i, band in enumerate(bands):
        mask = data_comparison["Band"] == band

        # Raw fluxes
        ax1 = plt.subplot(4, 4, 1 + i * 2)
        plt.scatter(
            data_comparison["Flux_Small_Raw"][mask],
            data_comparison["Flux_Big_Raw"][mask],
            alpha=0.7,
            s=30,
        )

        # Add 1:1 line
        min_flux = min(
            np.min(data_comparison["Flux_Small_Raw"][mask]),
            np.min(data_comparison["Flux_Big_Raw"][mask]),
        )
        max_flux = max(
            np.max(data_comparison["Flux_Small_Raw"][mask]),
            np.max(data_comparison["Flux_Big_Raw"][mask]),
        )
        plt.plot(
            [min_flux, max_flux], [min_flux, max_flux], "r--", alpha=0.8, label="1:1"
        )
        if scaling:
            plt.loglog()
        plt.xlabel(f"{band} Small Aperture Raw Flux [µJy]")
        plt.ylabel(f"{band} Large Aperture Raw Flux [µJy]")
        plt.title(f"{band} Raw Flux Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Corrected fluxes
        ax2 = plt.subplot(4, 4, 2 + i * 2)
        plt.scatter(
            data_comparison["Flux_Small_Corrected"][mask],
            data_comparison["Flux_Big_Corrected"][mask],
            alpha=0.7,
            s=30,
            color="orange",
        )

        min_flux_corr = min(
            np.min(data_comparison["Flux_Small_Corrected"][mask]),
            np.min(data_comparison["Flux_Big_Corrected"][mask]),
        )
        max_flux_corr = max(
            np.max(data_comparison["Flux_Small_Corrected"][mask]),
            np.max(data_comparison["Flux_Big_Corrected"][mask]),
        )
        plt.plot(
            [min_flux_corr, max_flux_corr],
            [min_flux_corr, max_flux_corr],
            "r--",
            alpha=0.8,
            label="1:1",
        )
        if scaling:
            plt.loglog()
        plt.xlabel(f"{band} Small Aperture Corrected Flux [µJy]")
        plt.ylabel(f"{band} Large Aperture Corrected Flux [µJy]")
        plt.title(f"{band} Corrected Flux Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 2. Flux Ratios (Large/Small aperture)
    for i, band in enumerate(bands):
        mask = data_comparison["Band"] == band

        # Raw flux ratios
        ax3 = plt.subplot(4, 4, 5 + i * 2)
        plt.hist(
            data_comparison["Flux_Ratio"][mask],
            bins=25,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        plt.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Unity")
        plt.axvline(
            np.median(data_comparison["Flux_Ratio"][mask]),
            color="orange",
            linestyle="-",
            linewidth=2,
            label="Median",
        )
        plt.xlabel("Flux Ratio (Large/Small)")
        plt.ylabel("Number of Sources")
        plt.title(f"{band} Raw Flux Ratio Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add statistics text
        ratio_stats = (
            f"Median: {np.median(data_comparison['Flux_Ratio'][mask]):.3f}\n"
            + f"Mean: {np.mean(data_comparison['Flux_Ratio'][mask]):.3f}\n"
            + f"Std: {np.std(data_comparison['Flux_Ratio'][mask]):.3f}"
        )
        plt.text(
            0.95,
            0.95,
            ratio_stats,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Corrected flux ratios
        ax4 = plt.subplot(4, 4, 6 + i * 2)
        plt.hist(
            data_comparison["Corrected_Flux_Ratio"][mask],
            bins=25,
            alpha=0.7,
            color="lightcoral",
            edgecolor="black",
        )
        plt.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Unity")
        plt.axvline(
            np.median(data_comparison["Corrected_Flux_Ratio"][mask]),
            color="orange",
            linestyle="-",
            linewidth=2,
            label="Median",
        )
        plt.xlabel("Corrected Flux Ratio (Large/Small)")
        plt.ylabel("Number of Sources")
        plt.title(f"{band} Corrected Flux Ratio Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add statistics text
        corr_ratio_stats = (
            f"Median: {np.median(data_comparison['Corrected_Flux_Ratio'][mask]):.3f}\n"
            + f"Mean: {np.mean(data_comparison['Corrected_Flux_Ratio'][mask]):.3f}\n"
            + f"Std: {np.std(data_comparison['Corrected_Flux_Ratio'][mask]):.3f}"
        )
        plt.text(
            0.95,
            0.95,
            corr_ratio_stats,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # 3. Aperture Correction Comparison
    for i, band in enumerate(bands):
        mask = data_comparison["Band"] == band

        ax5 = plt.subplot(4, 4, 9 + i * 2)
        plt.scatter(
            data_comparison["Apr_Corr_Small"][mask],
            data_comparison["Apr_Corr_Big"][mask],
            alpha=0.7,
            s=30,
            color="green",
        )

        min_corr = min(
            np.min(data_comparison["Apr_Corr_Small"][mask]),
            np.min(data_comparison["Apr_Corr_Big"][mask]),
        )
        max_corr = max(
            np.max(data_comparison["Apr_Corr_Small"][mask]),
            np.max(data_comparison["Apr_Corr_Big"][mask]),
        )
        plt.plot(
            [min_corr, max_corr], [min_corr, max_corr], "r--", alpha=0.8, label="1:1"
        )
        if scaling:
            plt.loglog()
        plt.xlabel(f"{band} Small Aperture Correction")
        plt.ylabel(f"{band} Large Aperture Correction")
        plt.title(f"{band} Aperture Corrections")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Corrected flux differences
        ax6 = plt.subplot(4, 4, 10 + i * 2)
        plt.hist(
            data_comparison["Corrected_Flux_Difference"][mask],
            bins=25,
            alpha=0.7,
            color="purple",
            edgecolor="black",
        )
        plt.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero")
        plt.axvline(
            np.median(data_comparison["Corrected_Flux_Difference"][mask]),
            color="orange",
            linestyle="-",
            linewidth=2,
            label="Median",
        )
        plt.xlabel("Corrected Flux Difference [µJy]")
        plt.ylabel("Number of Sources")
        plt.title(f"{band} Corrected Flux Difference (Large - Small)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add statistics text
        diff_stats = (
            f"Median: {np.median(data_comparison['Corrected_Flux_Difference'][mask]):.2f} µJy\n"
            + f"Mean: {np.mean(data_comparison['Corrected_Flux_Difference'][mask]):.2f} µJy\n"
            + f"Std: {np.std(data_comparison['Corrected_Flux_Difference'][mask]):.2f} µJy"
        )
        plt.text(
            0.95,
            0.95,
            diff_stats,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # 4. Flux vs Ratio relationships (to identify systematic trends)
    for i, band in enumerate(bands):
        mask = data_comparison["Band"] == band

        ax7 = plt.subplot(4, 4, 13 + i * 2)
        plt.scatter(
            data_comparison["Flux_Small_Raw"][mask],
            data_comparison["Flux_Ratio"][mask],
            alpha=0.6,
            s=30,
            c=data_comparison["Apr_Corr_Small"][mask],
            cmap="viridis",
        )
        plt.colorbar(label="Small Aperture Correction")
        plt.axhline(1.0, color="red", linestyle="--", alpha=0.8, label="Unity")
        if scaling:
            plt.xscale("log")
        plt.xlabel(f"{band} Small Aperture Raw Flux [µJy]")
        plt.ylabel("Flux Ratio (Large/Small)")
        plt.title(f"{band} Flux Ratio vs Brightness")
        plt.legend()
        plt.grid(True, alpha=0.3)

        ax8 = plt.subplot(4, 4, 14 + i * 2)
        plt.scatter(
            data_comparison["Flux_Small_Corrected"][mask],
            data_comparison["Corrected_Flux_Ratio"][mask],
            alpha=0.6,
            s=30,
            c=data_comparison["Apr_Corr_Big"][mask],
            cmap="plasma",
        )
        plt.colorbar(label="Large Aperture Correction")
        plt.axhline(1.0, color="red", linestyle="--", alpha=0.8, label="Unity")
        if scaling:
            plt.xscale("log")
        plt.xlabel(f"{band} Small Aperture Corrected Flux [µJy]")
        plt.ylabel("Corrected Flux Ratio (Large/Small)")
        plt.title(f"{band} Corrected Flux Ratio vs Brightness")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.suptitle("Comprehensive Aperture Photometry Comparison", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(fig_path, dpi=150)
    plt.show()
    plt.close()


def plot_aperture_summary(data_comparison, non_detections=None, scaling=False):
    fig_path = "/Users/benjamincollins/University/Master/Red_Cardinal/photometry/apertures/aperture_comparisons/"

    # Convert to arrays
    for key in data_comparison:
        data_comparison[key] = np.array(data_comparison[key])

    bands = ["F770W", "F1800W"]
    colors = ["#1f77b4", "#ff7f0e"]  # Distinct colors per band
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, band in enumerate(bands):
        mask = data_comparison["Band"] == band

        if non_detections:
            non_det_ids = set(str(gid) for gid in non_detections.get(band, []))
            mask &= ~np.isin(data_comparison["ID"], list(non_det_ids))
            print(f"Excluding {len(non_det_ids)} non-detections for band {band}")

        # --- (1) Corrected Flux Scatter ---
        ax = axes[i, 0]
        ax.scatter(
            data_comparison["Flux_Small_Corrected"][mask],
            data_comparison["Flux_Big_Corrected"][mask],
            alpha=0.7,
            s=30,
            color=colors[i],
        )

        # 1:1 line
        min_flux = min(
            np.min(data_comparison["Flux_Small_Corrected"][mask]),
            np.min(data_comparison["Flux_Big_Corrected"][mask]),
        )
        max_flux = max(
            np.max(data_comparison["Flux_Small_Corrected"][mask]),
            np.max(data_comparison["Flux_Big_Corrected"][mask]),
        )
        ax.plot(
            [min_flux, max_flux], [min_flux, max_flux], "k--", alpha=0.8, label="1:1"
        )
        if scaling:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(f"Corrected Flux small [µJy]", fontsize=12)
        ax.set_ylabel(f"Corrected Flux large [µJy]", fontsize=12)
        ax.set_title(f"{band} Corrected Flux Comparison", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=12)

        # Calculate and display R² correlation
        corr_diff = data_comparison["Corrected_Flux_Difference"][mask]
        flux_small = data_comparison["Flux_Small_Corrected"][mask]
        frac_diff = corr_diff / flux_small
        # ax.text(0.05, 0.9, f"median = {np.median(frac_diff):.3f}" + "\n" + rf"$\sigma$ = {np.std(frac_diff):.3f}",
        #       transform=ax.transAxes, fontsize=10,
        #      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # --- (2) Corrected Flux Ratio Histogram ---
        ax = axes[i, 1]

        # Convert masked data to a NumPy array
        ratios = np.array(data_comparison["Corrected_Flux_Ratio"][mask])

        for rat in ratios:
            if rat > 1.5:
                outlier_id = data_comparison["ID"][mask][ratios == rat][0]
                print(
                    f"Outlier in {band} with ratio {rat:.2f} for galaxy ID {outlier_id}"
                )

        # Plot histogram
        ax.hist(
            ratios,
            bins=25,
            alpha=0.7,
            color=colors[i],
            edgecolor="black",
            range=(0, 3.0),
        )

        # Reference lines
        ax.axvline(
            1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Unity"
        )
        ax.axvline(
            np.median(ratios),
            color="darkred",
            linestyle="-",
            linewidth=1.5,
            label="Median",
        )

        # Add compact statistics
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        num = f"N = {len(ratios)}"
        median_ratio = np.median(ratios)

        stats_text = (
            f"μ={mean_ratio:.2f}\nσ={std_ratio:.2f}\nMed={median_ratio:.2f}\n\n{num}"
        )

        ax.text(
            0.83,
            0.77,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Annotate number of sources
        # ax.text(0.95, 0.95,
        #        f"N = {len(ratios_clipped)} (95th pct of {len(ratios)})",
        #        ha='right', va='top',
        #        transform=ax.transAxes,
        #        fontsize=10,
        #        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Labels and formatting
        ax.set_xlabel("Corrected Flux Ratio (Large/Small)", fontsize=12)
        ax.set_ylabel("Number of Sources", fontsize=12)
        ax.set_title(f"{band} Corrected Flux Ratio Distribution", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=12)

    # plt.suptitle('Aperture Photometry Comparison: Short vs Long λ', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    figname = os.path.join(fig_path, "new_stats_thesis_new.png")
    plt.savefig(figname, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved summary plot to {figname}")

    # Second figure: Flux Ratio vs Brightness

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for i, band in enumerate(bands):
        mask = data_comparison["Band"] == band
        ax = axes[i]

        if non_detections:
            non_det_ids = set(str(gid) for gid in non_detections.get(band, []))
            mask &= ~np.isin(data_comparison["ID"], list(non_det_ids))
            print(f"Excluding {len(non_det_ids)} non-detections for band {band}")

        sc = ax.scatter(
            data_comparison["Flux_Small_Corrected"][mask],
            data_comparison["Corrected_Flux_Ratio"][mask],
            alpha=0.6,
            s=30,
            c=data_comparison["Apr_Corr_Big"][mask],
            cmap="viridis",
            edgecolor="black",
            vmin=1.15,
            vmax=1.45,
        )

        # Proper colorbar
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Aperture Correction Factor (Large)")

        ax.axhline(1.0, color="red", linestyle="--", alpha=0.8, label="Unity")
        ax.axhline(
            np.median(ratios),
            color="darkred",
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
            label="Median",
        )

        if scaling:
            ax.set_xscale("log")
        ax.set_ylim(0, 4)
        ax.set_xlabel(f"{band} Small Aperture Corrected Flux [µJy]", fontsize=12)
        ax.set_ylabel("Flux Ratio (Large/Small)", fontsize=12)
        ax.set_title(f"{band} Flux Ratio vs Brightness", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=12)

    figname = os.path.join(fig_path, "ratio_vs_brightness_thesis_new.png")
    plt.savefig(figname, bbox_inches="tight", pad_inches=0)
    plt.tight_layout()
    plt.show()
    print(f"Saved scatter plot to {figname}")


def write_aperture_summary(
    data_comparison, common_ids, summary_doc_path, non_detections=None
):
    """
    Write a comprehensive aperture comparison summary to a text file.
    Includes raw ratios, aperture corrections, corrected differences,
    fractional differences, and context relative to measurement errors.
    """

    doc_path = os.path.join(
        "/Users/benjamincollins/University/Master/Red_Cardinal/photometry/apertures/aperture_comparisons",
        summary_doc_path,
    )

    df = pd.DataFrame(data_comparison)
    print("Bands in table:", np.unique(data_comparison["Band"]))

    bands = ["F770W", "F1000W", "F1800W", "F2100W"]

    with open(doc_path, "w") as file:
        file.write("\n" + "=" * 80 + "\n")
        file.write("COMPREHENSIVE APERTURE COMPARISON SUMMARY\n")
        file.write("=" * 80 + "\n")

        for band in bands:
            band_data = df[df["Band"] == band]
            mask = np.ones(len(band_data), dtype=bool)

            if non_detections:
                non_det_ids = set(str(gid) for gid in non_detections.get(band, []))
                mask &= ~np.isin(band_data["ID"], list(non_det_ids))
                print(f"Excluding {len(non_det_ids)} non-detections for band {band}")

            band_data = band_data[mask]

            file.write(f"\n{band} FILTER:\n")
            file.write("-" * 40 + "\n")

            # --- Raw flux ratios ---
            flux_ratio = band_data["Flux_Ratio"]
            file.write("Raw Flux Ratios (Large/Small):\n")
            file.write(
                f"  Median: {np.median(flux_ratio):.3f} ± {np.std(flux_ratio):.3f}\n"
            )
            file.write(f"  Mean:   {np.mean(flux_ratio):.3f}\n")
            file.write(
                f"  Range:  {np.min(flux_ratio):.3f} – {np.max(flux_ratio):.3f}\n"
            )

            # --- Corrected flux ratios ---
            corr_ratio = band_data["Corrected_Flux_Ratio"]
            file.write("\nCorrected Flux Ratios (Large/Small):\n")
            file.write(
                f"  Median: {np.median(corr_ratio):.3f} ± {np.std(corr_ratio):.3f}\n"
            )
            file.write(f"  Mean:   {np.mean(corr_ratio):.3f}\n")
            file.write(
                f"  Range:  {np.min(corr_ratio):.3f} – {np.max(corr_ratio):.3f}\n"
            )

            # --- Calculate bias reduction ---
            raw_bias = np.median(flux_ratio) - 1.0
            corrected_bias = np.median(corr_ratio) - 1.0
            bias_reduction = (raw_bias - corrected_bias) / raw_bias * 100

            file.write(f"\nBias Reduction:\n")
            file.write(f"  Initial systematic bias: {raw_bias * 100:.1f}%\n")
            file.write(f"  Residual systematic bias: {corrected_bias * 100:.1f}%\n")
            file.write(f"  Bias reduction achieved: {bias_reduction:.1f}%\n")

            # --- Aperture corrections ---
            file.write("\nAperture Corrections:\n")
            small_corr_med = np.median(band_data["Apr_Corr_Small"])
            big_corr_med = np.median(band_data["Apr_Corr_Big"])
            small_sigma_corr = np.std(band_data["Apr_Corr_Small"])
            big_sigma_corr = np.std(band_data["Apr_Corr_Big"])
            file.write(f"  Small aperture median: {small_corr_med:.3f}\n")
            file.write(f"  Large aperture median: {big_corr_med:.3f}\n")
            file.write(
                f"  Difference (Large–Small): {big_corr_med - small_corr_med:.3f}\n"
            )

            # --- Final corrected flux differences ---
            corr_diff = band_data["Corrected_Flux_Difference"]
            file.write("\nFinal Corrected Flux Differences (Large – Small) [µJy]:\n")
            file.write(
                f"  Median: {np.median(corr_diff):.2f} ± {np.std(corr_diff):.2f}\n"
            )
            file.write(f"  Mean:   {np.mean(corr_diff):.2f}\n")
            higher_flux_pct = np.sum(corr_diff > 0) / len(corr_diff) * 100
            file.write(
                f"  Sources with higher flux in large aperture: {higher_flux_pct:.1f}%\n"
            )

            # --- Fractional differences (preferred) ---
            frac_diff = (
                corr_diff / band_data["Flux_Small_Corrected"]
            )  # ΔFlux / Flux_small
            frac_diff_pct = frac_diff * 100
            file.write("\nFractional Differences ((Large – Small)/Small):\n")
            file.write(
                f"  Median: {np.median(frac_diff_pct):.1f}% ± {np.std(frac_diff_pct):.1f}%\n"
            )
            file.write(
                f"  Range (5–95th pct): {np.percentile(frac_diff_pct, 5):.1f}% – {np.percentile(frac_diff_pct, 95):.1f}%\n"
            )

            # --- Compare to uncertainties ---
            flux_small = band_data["Flux_Small_Corrected"]
            flux_err_small = np.median(band_data["Flux_Err_Small_Corrected"])
            std_flux_err_small = np.std(band_data["Flux_Err_Small_Corrected"])

            # Propagated correction-induced uncertainty for each source
            corr_err = np.sqrt(
                (small_corr_med * flux_err_small) ** 2
                + (flux_small * small_sigma_corr) ** 2
            )

            # Median difference between large and small aperture corrected fluxes
            median_corr_diff = np.median(band_data["Corrected_Flux_Difference"])

            file.write("\nContext vs. Measurement Errors:\n")
            file.write(
                f"  Median corrected aperture difference: {median_corr_diff:.2f} µJy\n"
            )
            file.write(
                f"  Median flux uncertainty: {flux_err_small:.2f} ± {std_flux_err_small:.2f} µJy\n"
            )
            file.write(
                f"  Median propagated uncertainty incl. correction: {np.median(corr_err):.2f} µJy\n"
            )

            if abs(median_corr_diff) < flux_err_small:
                file.write(
                    "  → Aperture differences are smaller than typical measurement errors.\n"
                )
            else:
                file.write(
                    "  → Aperture differences are comparable to or larger than typical measurement errors.\n"
                )

            # --- Compact interpretation line ---
            file.write("\nSummary:\n")
            file.write(
                f"  Corrected photometry converges across apertures, with residuals ≲{np.median(np.abs(frac_diff_pct)):.1f}%.\n"
            )

        # Final count
        file.write(f"\nTotal sources analysed: {len(common_ids)}\n")
        file.write("=" * 80 + "\n")
        print("Wrote summary document to", doc_path)


def plot_appendix_figure(data_comparison, fig_path, non_detections=None, scaling=None):
    """
    Create a compact summary plot for aperture photometry comparison.
    Layout: 4 rows (one per band) × 3 columns
    - Column 1: Corrected flux comparison (scatter)
    - Column 2: Corrected flux ratio distribution (histogram)
    - Column 3: Flux ratio vs brightness (scatter)
    """

    # Convert to arrays for easier handling
    for key in data_comparison:
        data_comparison[key] = np.array(data_comparison[key])

    # Create summary figure optimized for A4 appendix
    fig = plt.figure(figsize=(12, 16))  # Good aspect ratio for A4

    bands = ["F770W", "F1000W", "F1800W", "F2100W"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]  # Distinct colors per band

    for i, band in enumerate(bands):
        mask = data_comparison["Band"] == band
        band_color = colors[i]

        if non_detections:
            non_det_ids = set(str(gid) for gid in non_detections.get(band, []))
            mask &= ~np.isin(data_comparison["ID"], list(non_det_ids))
            print(f"Excluding {len(non_det_ids)} non-detections for band {band}")

        # Column 1: Corrected flux comparison
        ax1 = plt.subplot(4, 3, i * 3 + 1)
        plt.scatter(
            data_comparison["Flux_Small_Corrected"][mask],
            data_comparison["Flux_Big_Corrected"][mask],
            alpha=0.6,
            s=20,
            color=band_color,
        )

        # Add 1:1 line
        min_flux = min(
            np.min(data_comparison["Flux_Small_Corrected"][mask]),
            np.min(data_comparison["Flux_Big_Corrected"][mask]),
        )
        max_flux = max(
            np.max(data_comparison["Flux_Small_Corrected"][mask]),
            np.max(data_comparison["Flux_Big_Corrected"][mask]),
        )
        plt.plot(
            [min_flux, max_flux], [min_flux, max_flux], "k--", alpha=0.7, linewidth=1
        )

        if scaling:
            plt.loglog()
        plt.xlabel("Small Aperture [µJy]")
        plt.ylabel("Large Aperture [µJy]")
        plt.title(f"{band} Corrected Flux", fontsize=11)
        plt.grid(True, alpha=0.3)

        # Calculate and display R² correlation
        corr_coef = np.corrcoef(
            data_comparison["Flux_Small_Corrected"][mask],
            data_comparison["Flux_Big_Corrected"][mask],
        )[0, 1]
        r_squared = corr_coef**2
        plt.text(
            0.05,
            0.9,
            f"R² = {r_squared:.2f}",
            transform=ax1.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Column 2: Corrected flux ratio distribution
        ax2 = plt.subplot(4, 3, i * 3 + 2)
        n, bins, patches = plt.hist(
            data_comparison["Corrected_Flux_Ratio"][mask],
            bins=25,
            alpha=0.7,
            color=band_color,
            edgecolor="black",
            linewidth=0.5,
            range=(0, 2.5),
        )

        # Add vertical lines for key statistics
        median_ratio = np.median(data_comparison["Corrected_Flux_Ratio"][mask])
        plt.axvline(1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
        plt.axvline(median_ratio, color="darkred", linestyle="-", linewidth=1.5)

        plt.xlabel("Flux Ratio (Large/Small)")
        plt.ylabel("N Sources")
        plt.title(f"{band} Ratio Distribution", fontsize=11)
        plt.grid(True, alpha=0.3)

        # Add compact statistics
        mean_ratio = np.mean(data_comparison["Corrected_Flux_Ratio"][mask])
        std_ratio = np.std(data_comparison["Corrected_Flux_Ratio"][mask])
        remark = "2 outliers > 4.0"
        num = f"N = {len(data_comparison['Corrected_Flux_Ratio'][mask])}"

        # if i == len(bands) - 2:
        #    stats_text = f'μ={mean_ratio:.2f}\nσ={std_ratio:.2f}\nMed={median_ratio:.2f}\n\n{num}\n{remark}'
        # else:
        stats_text = (
            f"μ={mean_ratio:.2f}\nσ={std_ratio:.2f}\nMed={median_ratio:.2f}\n\n{num}"
        )

        # plt.text(0.95, 0.95, stats_text, transform=ax2.transAxes, fontsize=8,
        #        verticalalignment='top', horizontalalignment='right')
        # bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.text(
            0.73,
            0.7,
            stats_text,
            transform=ax2.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Column 3: Flux ratio vs brightness
        ax3 = plt.subplot(4, 3, i * 3 + 3)
        scatter = plt.scatter(
            data_comparison["Flux_Small_Corrected"][mask],
            data_comparison["Corrected_Flux_Ratio"][mask],
            alpha=0.6,
            s=15,
            c=data_comparison["Apr_Corr_Small"][mask],
            cmap="viridis",
            vmin=0.8,
            vmax=3.0,
            edgecolor="black",
        )

        plt.axhline(
            1.0, color="red", linestyle="--", alpha=0.8, linewidth=1.5, label="Unity"
        )
        plt.axhline(
            median_ratio,
            color="darkred",
            linestyle="-",
            alpha=0.8,
            linewidth=1,
            label="Median",
        )

        if scaling:
            plt.xscale("log")
        plt.xlabel("Small Aperture Flux [µJy]")
        plt.ylabel("Flux Ratio")
        plt.title(f"{band} Ratio vs Brightness", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 2.5)

        # Add colorbar only for the last band to save space
        if i == len(bands) - 1:
            cbar = plt.colorbar(scatter, ax=ax3, shrink=0.8)
            cbar.set_label("Aperture Correction Factor (Large)", fontsize=10)

    # Overall title and layout adjustment
    # plt.suptitle('Aperture Photometry Summary: Small vs Large Aperture Comparison',
    # fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.35, wspace=0.3)

    # Save with high DPI for appendix quality
    plt.savefig(fig_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.show()
    plt.close()


def analyse_outliers(
    data_comparison,
    flags,
    ratio_col="Corrected_Flux_Ratio",
    threshold=3.0,
    summary=True,
):
    """
    Identify outliers where aperture corrections produced unphysical flux ratios.

    Parameters
    ----------
    data_comparison : dict, pd.DataFrame, or astropy Table
        Must contain columns: 'ID', 'Band', and ratio_col (default: 'Corrected_Flux_Ratio').
    ratio_col : str
        Column name for corrected flux ratios to check (Large/Small).
    threshold : float
        Absolute deviation from unity considered "bad".
        Example: threshold=3 → flags ratios < 1/3 or > 3.
    summary : bool
        If True, prints summary counts per filter.

    Returns
    -------
    outliers : pd.DataFrame
        Subset of rows that are outliers.
    """

    from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval

    # Convert to DataFrame if needed
    if not isinstance(data_comparison, pd.DataFrame):
        data = pd.DataFrame(data_comparison)
    else:
        data = data_comparison.copy()

    # Boolean mask: ratios too extreme
    ratios = data[ratio_col].astype(float)
    bad_mask = (ratios < 1 / threshold) | (ratios > threshold) | ~np.isfinite(ratios)

    outliers = data[bad_mask]

    if summary:
        print("=== Outlier Summary ===")
        for band in outliers["Band"].unique():
            count = np.sum(outliers["Band"] == band)
            print(f"{band}: {count} flagged outliers")
        print(
            f"Total outliers: {len(outliers)} / {len(data)} ({100 * len(outliers) / len(data):.1f}%)"
        )

    for obj in outliers.to_dict(orient="records"):
        objid = obj["ID"]
        band = obj["Band"]
        ratio = obj[ratio_col]

        if int(objid) in flags.get(band, []):
            print(f"⚠️ {objid} in {band} - known nondetection")
        else:
            print("This counts as a detection (apparently)")

        try:
            vis_data = load_vis(
                f"/Users/benjamincollins/University/Master/Red_Cardinal/photometry/vis_data/{objid}_{band}.h5"
            )
        except:
            print(f"❌ No VIS data found for {objid}")
            continue

        img = vis_data["background_subtracted"]

        # Normalisation: auto scale + asinh stretch
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(img)
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())

        plt.figure(figsize=(4, 4))
        plt.imshow(img, origin="lower", cmap="inferno", norm=norm)
        plt.colorbar(label="Flux")
        plt.title(f"Galaxy {objid} - {band}\nRatio = {ratio:.2f}")
        plt.tight_layout()
        plt.show()

    return outliers


def aperture_flux_at(img, aperture_params):
    # aperture shape (scale only, will move centres + vary theta)
    a = aperture_params["a"]
    b = aperture_params["b"]
    theta_ref = aperture_params["theta"]
    x0, y0 = aperture_params["x_center"], aperture_params["y_center"]

    aperture = EllipticalAperture((x0, y0), a, b, theta_ref)

    phot = aperture_photometry(img, aperture, method="exact")
    return phot["aperture_sum"][0]


def empirical_aperture_rms(img, aperture_params, n_random=200, valid_frac=0.25):
    """
    Estimate RMS by placing random elliptical apertures on the image.

    Parameters
    ----------
    img : 2D array
        Background-subtracted + masked cutout image.
    aperture_params : dict
        Dictionary with keys ['a', 'b', 'theta', 'x_center', 'y_center'].
    n_random : int
        Number of random apertures to place.
    valid_frac : float
        Minimum fraction of aperture pixels that must be valid (not NaN)
    Returns
    -------
    rms : float
        Empirical RMS of aperture fluxes.
    """
    ny, nx = img.shape
    aperturesums = []
    attempts = 0
    max_attempts = n_random * 20

    # aperture shape (scale only, will move centres + vary theta)
    a = aperture_params["a"]
    b = aperture_params["b"]
    theta_ref = aperture_params["theta"]
    x0, y0 = aperture_params["x_center"], aperture_params["y_center"]

    while len(aperturesums) < n_random and attempts < max_attempts:
        attempts += 1

        # random centre inside image (avoid edges)
        x = random.uniform(a + 2, nx - a - 2)
        y = random.uniform(b + 2, ny - b - 2)

        # skip if centre falls on masked pixel
        if np.isnan(
            img[int(y), int(x)]
        ):  # NaN is the only value not similar to itself! Important!
            continue

        # random angle variation (around reference theta)
        theta = random.uniform(0, 2 * np.pi)

        aperture = EllipticalAperture((x, y), a, b, theta)
        aperture_mask = aperture.to_mask(method="exact")
        aperture_data = aperture_mask.multiply(img)
        aperture_data_mask = aperture_mask.data

        # Accept apertures with sufficient valid pixels
        n_valid = np.sum(np.isfinite(aperture_data[aperture_data_mask > 0]))
        n_total = np.sum(aperture_data_mask > 0)
        frac_valid = n_valid / n_total if n_total > 0 else 0
        if frac_valid < valid_frac:
            continue

        phot_table = aperture_photometry(img, aperture, method="exact")
        flux = phot_table["aperture_sum"][0]

        if n_valid > 0:
            flux *= n_total / n_valid  # scale correction

        if np.isfinite(flux):
            aperturesums.append(flux)

    if len(aperturesums) < max(10, n_random // 4):
        # fallback: pixel rms scaled to aperture area
        print("📉 Too few valid random apertures, using pixel RMS fallback")
        pixrms = np.nanstd(img)
        area = np.pi * a * b
        return pixrms * np.sqrt(area)

    return np.std(aperturesums, ddof=1)


def recompute_empirical_snr(vis_data, n_random=200):
    """
    Compute flux & empirical S/N at source centre using background-subtracted image
    with combined mask applied.
    """

    galaxy_id = vis_data["galaxy_id"]
    print(f"Recomputing empirical S/N for {galaxy_id}...")

    img = vis_data["background_subtracted"]

    bkg_mask = vis_data["segmentation_mask"] | np.isnan(vis_data["original_data"])
    clean_image = np.where(bkg_mask, np.nan, img)

    combined_mask = vis_data["source_mask"] | bkg_mask
    very_clean_image = np.where(combined_mask, np.nan, img)

    # Load aperture used for photometry
    aperture_params = vis_data["aperture_params"]
    x_center = aperture_params["x_center"]
    y_center = aperture_params["y_center"]
    a = aperture_params["a"]
    b = aperture_params["b"]
    theta = aperture_params["theta"]

    ny, nx = img.shape
    centre = (x_center, y_center)

    flux = aperture_flux_at(clean_image, aperture_params)
    emp_rms = empirical_aperture_rms(
        very_clean_image, aperture_params=aperture_params, n_random=n_random
    )
    sn = flux / emp_rms if emp_rms > 0 else 0.0

    return dict(
        objid=vis_data["galaxy_id"], flux=flux, flux_err=emp_rms, sn=sn, centre=centre
    )


def stack_cutouts(fits_paths, hdu_index=1, method="median"):
    imgs = []
    for p in fits_paths:
        with fits.open(p) as hdul:
            imgs.append(hdul[hdu_index].data.astype(float))
    arr = np.stack(imgs, axis=0)
    if method == "median":
        return np.nanmedian(arr, axis=0)
    return np.nanmean(arr, axis=0)


def show_apertures(objid, band):
    vis_data = load_vis(
        f"/Users/benjamincollins/University/master/Red_Cardinal/photometry/vis_data/{objid}_{band}.h5"
    )
    output_file = os.path.join(
        f"/Users/benjamincollins/University/master/Red_Cardinal/photometry/apertures/aperture_comparisons/{objid}_{band}.png"
    )

    # Extract data from the dictionary
    image_data = vis_data["original_data"]
    background_plane = vis_data["background_plane"]
    background_subtracted = vis_data["background_subtracted"]
    segm_mask = vis_data["segmentation_mask"]
    mask_vis = vis_data["mask_vis"]
    aperture_params = vis_data["aperture_params"]
    sigma = vis_data["sigma"]
    region_name = vis_data["region_name"]
    galaxy_id = vis_data["galaxy_id"]
    filter = vis_data["filter"]

    # Create aperture objects for plotting
    x_center = aperture_params["x_center"]
    y_center = aperture_params["y_center"]
    a = aperture_params["a"]
    b = aperture_params["b"]
    theta = aperture_params["theta"]

    big_aperture = EllipticalAperture(
        positions=(x_center, y_center), a=a, b=b, theta=theta
    )

    small_aperture = EllipticalAperture(
        positions=(x_center, y_center), a=a / 2, b=b / 2, theta=theta
    )

    # Create figure with three subplots in a horizontal row
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Background-subtracted data
    vmin = np.nanpercentile(background_subtracted, 5)
    vmax = np.nanpercentile(background_subtracted, 95)

    im1 = axes[0].imshow(
        background_subtracted, origin="lower", cmap="magma", vmin=vmin, vmax=vmax
    )
    plt.colorbar(im1, ax=axes[0], label="Flux [MJy/(sr pixel)]")
    small_aperture.plot(ax=axes[0], color="black", lw=3, label="Small Aperture")
    axes[0].legend(loc="upper right", fontsize=10)
    # axes[0].set_title(f'Galaxy ID {galaxy_id} - {filter}')

    im2 = axes[1].imshow(
        background_subtracted, origin="lower", cmap="magma", vmin=vmin, vmax=vmax
    )
    plt.colorbar(im2, ax=axes[1], label="Flux [MJy/(sr pixel)]")
    big_aperture.plot(ax=axes[1], color="blue", lw=3, label="Large Aperture")
    axes[1].legend(loc="upper right", fontsize=10)
    # axes[1].set_title(f'Galaxy ID {galaxy_id} - {filter}')

    # Tight layout and saving the figure‚
    plt.tight_layout()
    # plt.suptitle(f'Galaxy ID {galaxy_id} - {filter}', fontsize=14)
    plt.subplots_adjust(top=0.85)  # Adjust to prevent overlap with annotation
    plt.savefig(output_file, dpi=150)
    plt.show()
    plt.close(fig)
