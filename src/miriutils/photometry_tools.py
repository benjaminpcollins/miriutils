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
#     "seaborn",
# ]
# ///
"""
MIRI Utils: Photometric Pipeline Module
==========================================

This module provides the MiriPipeline class, a specialized framework designed
to produce publication-ready mid-infrared photometry for the Blue Jay survey.
It automates the transition from raw FITS cutouts in the '_i2d.fits' format to
aperture-corrected Janskys.

Classes:
--------
    - MiriPipeline: The core engine for batch processing MIRI photometry.

Key Capabilities:
-----------------
    - Automated "Wide" Table generation (one row per ID, columns per band).
    - Multi-instrument WCS alignment and automated aperture adjustment.
    - Local background modeling using iterative sigma-clipped 2D statistics.
    - High-fidelity aperture corrections using 4x oversampled stpsf models.
    - Rigorous error propagation including nominal detector noise and 
      background modeling uncertainties.
    - Quality flagging for detector artefacts and companion contamination.
    - Stores data for visualising the background modelling in h5py format
      and provides functions for easy reading and plotting.
    - Automatically generates detection statistics
    - Provides functions for visualising systematics introduced due to the
      choice of aperture sizes for MIRI.
    - Includes tools for Curve of Growth (CoG) analysis to compare the 
      physical values to PSF model predictions

Workflow:
---------
    1. Pre-scans directories to initialise a type-safe, wavelength-ordered table structure.
    2. Aligns science apertures based on Blue Jay (NIRCam F444W) morphology.
    3. Performs exact aperture photometry and background estimation.
    4. Calculates band-specific PSF corrections on an oversampled grid.
    5. Exports a dual-format (FITS/CSV) catalogue with standardised columns.

Example usage:
--------------
    from miri_utils import MiriPipeline

    ids_to_process = [7102, 11202, 16874]   # int of galaxy IDs to process
    
    # Initialise the pipeline
    pipeline = MiriPipeline(
        all_ids=ids_to_process,
        cutout_dir="./data/cutouts",
        output_dir="./miri_photometry",
        nircam_dir="./NIRCam/cutouts",
        aperture_table="./data/aperture_table.fits"
    )

    # Run full survey photometry and store FITS and CSV format output tables
    pipeline.run_photometry(write_to="Phot_Table_MIRI")
    

Author: Benjamin P. Collins
Date: Jan 2026
Version: 1.0.0
"""

import os
import warnings
import json
from pathlib import Path
import seaborn as sns

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from scipy.stats import skew, kurtosis

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.table import Table, MaskedColumn, join, vstack
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
    def __init__(self, ids_to_process, cutouts_dir, photometry_dir, nircam_dir, aperture_table, psf_dir=None, scaling_exceptions_file=None):
        
        # Load aperture table
        self.master_table = Table.read(aperture_table)
        self.table_path = None  # To be overwritten in save_catalogue()
        
        self.all_ids = ids_to_process
        
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
        
        # Initialise directories
        self.cutouts_dir = cutouts_dir
        self.output_dir = photometry_dir
        self.nircam_dir = nircam_dir        
        
        # Default PSF directory if none provided
        if psf_dir is None:
            self.psf_dir = os.path.join(self.output_dir, "psfs")
            print(f"Using default PSF directory: {self.psf_dir}")
        else:
            self.psf_dir = psf_dir
            print(f"Found PSF directory {self.psf_dir}")
        
        self.aperture_dir = os.path.join(self.output_dir, "apertures")
        self.mosaic_dir = os.path.join(self.output_dir, "mosaic_plots")
        self.phot_table_dir = os.path.join(self.output_dir, "phot_tables")
        self.cog_dir = os.path.join(self.output_dir, "curve_of_growth")
        self.vis_dir = os.path.join(self.output_dir, "vis_data")
        self.detection_dir = os.path.join(self.output_dir, "vis_data")
        
        for dir in [self.aperture_dir, self.mosaic_dir, self.phot_table_dir, self.detection_dir]:
            os.makedirs(dir, exist_ok=True)
        
        # 2. Handle Scaling Exceptions File
        if scaling_exceptions_file is None:
            # Default name in the output directory if none provided
            scaling_exceptions_file = os.path.join(self.output_dir, "scaling_config.csv")

        self.scaling_exceptions_path = scaling_exceptions_file
        self.scaling_exceptions = self._initialise_scaling_config()
        
        # Place these in your __init__ or as a config block
        self.quality_config = {
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
    
    
    def prepare_aperture(self, file_path, rescale):
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
        aperture_dir = self.aperture_dir
        out_path = os.path.join(aperture_dir, f"{galaxy_id}_all.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        

    def estimate_background(self, aperture_params, sigma_val=2.5, n_iters=3, save_vis=False):
        """
        Fits a 2D plane to the image (excluding sources) and calculates 
        local statistics in an elliptical annulus.
        """
        data = aperture_params["data"]
        x_cen, y_cen = aperture_params["x"], aperture_params["y"]
        a, b = aperture_params["a"], aperture_params["b"]
        theta = aperture_params["theta"] # Already in Radians from prepare_aperture
        galaxy_id = aperture_params["id"]
        
        # Load filter and determine if it is a long (>15µm) or short wavelength band
        filt = aperture_params["meta"]["filter"]
        is_long_wl = True if self.wavelength_map[filt] > 14.0 else False
        
        yi, xi = np.indices(data.shape)

        # 1. Create an two source masks (buffer around the aperture)
        source_ap = EllipticalAperture((x_cen, y_cen), a=a, b=b, theta=theta)
        source_mask = source_ap.to_mask(method='center').to_image(data.shape).astype(bool)
        
        # The large source mask prevents the target's own light from biasing the background
        a_in, b_in = a + 8, b + 8
        bkg_source_ap = EllipticalAperture((x_cen, y_cen), a=a_in, b=b_in, theta=theta)
        source_mask_large = bkg_source_ap.to_mask(method='center').to_image(data.shape).astype(bool)

        # 2. Aggressive Neighbor Detection
        # We use a very low threshold (1.5 sigma) to catch faint wings
        init_mask = source_mask_large | np.isnan(data)
        _, median_init, std_init = sigma_clipped_stats(data, sigma=sigma_val, mask=init_mask, maxiters=5)

        # Stricter detection threshold for the mask
        detection_threshold = median_init + (2.0 * std_init) 
        segm = detect_sources(data, detection_threshold, npixels=8)

        segm_mask = (segm.data > 0) if segm else np.zeros_like(data, dtype=bool) 
        
        # 3. Final Combined Mask 
        combined_mask = source_mask_large | segm_mask | np.isnan(data)
        
        if is_long_wl:
            # 1. Calculate Elliptical Distance for weighting
            # Transform coordinates to align with aperture rotation
            dx = xi - x_cen
            dy = yi - y_cen
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            
            # Account for angle of rotation
            x_rot = dx * cos_t + dy * sin_t
            y_rot = -dx * sin_t + dy * cos_t
            
            # Normalised elliptical distance (d=1 is the edge of the 'a' radius)
            d_ell = np.sqrt(x_rot**2 + (y_rot / (b/a))**2)

            # 2. Define the Weighting Scale (sigma)
            # For F1800W/F2100W, we want to focus on a region ~2-3x the aperture size
            # to avoid edge artifacts like in ID 12175
            sigma = a * 2.5
            weights = np.exp(-(d_ell**2) / (2 * sigma**2))
        
        # Initialise coefficients for the loop
        coeffs = [0, 0, median_init] 
        fit_mask = ~combined_mask 
        
        print(f"  > Starting iterative plane fit ({n_iters} iterations)...")
        
        for i in range(n_iters):
            # Prepare the design matrix for pixels in the current mask
            A = np.vstack([xi[fit_mask], yi[fit_mask], np.ones_like(xi[fit_mask])]).T
            z = data[fit_mask]
            
            if len(z) < 10:
                break # Safety break if we mask too much
            
            # For all wavelengths above 15 microns use weighted plane fit!
            if is_long_wl:
                # Apply weights to the design matrix and target vector
                # This is the 'Weighted Least Squares' solution: (A^T W A)^-1 A^T W z
                w = weights[fit_mask]
                W = np.diag(w) # For large images, use A * w[:, np.newaxis] for memory efficiency
                Aw = A * w[:, np.newaxis]
                zw = z * w
                
                # Solve for Plane: z = alpha*x + beta*y + gamma
                coeffs, _, _, _ = np.linalg.lstsq(Aw, zw, rcond=None)
            else:
                # Solve for Plane: z = alpha*x + beta*y + gamma
                coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
                
            alpha, beta, gamma = coeffs
            
            # Calculate residuals of this specific fit
            current_plane = alpha * xi + beta * yi + gamma
            residuals = data - current_plane
            
            # Update the mask using sigma clipping on the residuals
            # We only look at the pixels currently in fit_mask to find new outliers
            res_to_clip = residuals[fit_mask]
            clipped_res = sigma_clip(res_to_clip, sigma=sigma_val, maxiters=5, cenfunc=np.median)
            
            # Refine the fit_mask for the NEXT iteration
            # This removes pixels that were rejected by the sigma clip
            new_fit_mask = fit_mask.copy()
            new_fit_mask[fit_mask] = ~clipped_res.mask
            
            # Check if we've stopped masking new pixels (convergence)
            if np.array_equal(new_fit_mask, fit_mask):
                print(f"    - Converged after {i+1} iterations.")
                break
                
            fit_mask = new_fit_mask

        # Create mask for CoG analysis
        non_sky_mask = ~fit_mask
        kill_mask = non_sky_mask.copy()
        kill_mask[source_mask_large] = False
                
        # Final Plane Generation
        background_plane = alpha * xi + beta * yi + gamma
        data_bkgsub = data - background_plane        
    
        # Define elliptical annulus for local background estimate 
        # Set dynamic outer radius based on image bounds to prevent crashes
        img_h, img_w = data.shape
        dist_to_edge = min(x_cen, img_w - x_cen, y_cen, img_h - y_cen)
        a_out = dist_to_edge - 2    # 2 pixel buffer at the image boundaries
        b_out = a_out * 0.9 # Maintain aspect ratio
        
        annulus = EllipticalAnnulus((x_cen, y_cen), a_in=a_in, a_out=a_out, 
                                    b_in=b_in, b_out=b_out, theta=theta)
        
        ann_mask = annulus.to_mask(method='center').to_image(data.shape).astype(bool)
        
        # Only use pixels that are in the annulus AND not a detected source
        bkg_pixels_mask = ann_mask & ~combined_mask
        
        # Final Stats
        # Extract the 1D arrays of pixels
        plane_vals = np.asarray(background_plane[bkg_pixels_mask])
        res_vals = np.asarray(data_bkgsub[bkg_pixels_mask])

        # Remove any NaNs that might have snuck in
        res_vals = res_vals[~np.isnan(res_vals)]

        # The Level (The pedestal subtracted)
        background_median = np.median(plane_vals)

        # The Noise (Robust RMS / Sigma)
        # We use 1.4826 * MAD to match the scale of a standard deviation
        mad = np.median(np.abs(res_vals - np.median(res_vals)))
        background_rms = 1.4826 * mad 
        
        # Perform quality checks for the background residuals
        global_bkg_res = data_bkgsub[~combined_mask]
        quality_level, quality_reasons, quality_metrics = self.apply_quality_flagging(global_bkg_res)
        
        # Initialise mask visualisation for plotting
        mask_vis = np.zeros_like(data, dtype=int)        
        # Start with everything as "0" (Excluded)

        # Assign the "Sky" pixels (all pixels used in the final iterative fit)
        # This includes the annulus pixels that survived clipping
        mask_vis[new_fit_mask] = 1 

        # Assign the "Source" pixels (your photometry aperture)
        # We do this last so it overwrites anything else
        mask_vis[source_mask] = 2
        
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
        if save_vis:
            vis_dir = self.vis_dir
            os.makedirs(vis_dir, exist_ok=True)

            vis_path = os.path.join(vis_dir, f"{galaxy_id}_{filt}.h5")
            self.save_vis(vis_data, vis_path)
        
        return {
            "id": galaxy_id,
            "filter": filt,
            "median": background_median,
            "median_res": np.median(res_vals),
            "rms": background_rms,
            "plane": background_plane,
            "subtracted": data_bkgsub,
            "annulus": annulus,
            "source_ap": source_ap,
            "mask_vis": mask_vis,
            "kill_mask": kill_mask,
            "quality_level": quality_level,
            "quality_reasons": quality_reasons,
            "quality_metrics": quality_metrics
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

        # --- 4. Mask/Region Visualisation ---
        mask_vis = bkg_dict["mask_vis"]
        
        cmap_mask = plt.get_cmap('viridis', 3)
        im4 = ax4.imshow(mask_vis, origin="lower", cmap=cmap_mask, vmin=-0.5, vmax=2.5)
        ax4.set_title("Region Classification")
        
        # 1. Plot the Annulus Outline (The colorful boundary)
        annulus.plot(ax=ax4, 
                    color='black', 
                    lw=1.2, 
                    ls='-', 
                    alpha=0.9, 
                    label="Annulus")

        # 2. Create the transparent fill by calling .plot again with fill=True
        # This avoids the "no attribute" errors entirely
        annulus.plot(ax=ax4,
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.15,
                    fill=True)
        
        cbar = plt.colorbar(im4, ax=ax4, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
        cbar.set_ticklabels(["Excluded/NaN", "Fit Region", "Source"])

        plt.suptitle(f"Background Model: Galaxy {gid} | {filt} | {survey_obs}", fontsize=14)
        plt.legend()
        
        # 1. Determine the destination
        if save_path is None:
            # Default organizational structure
            filt_dir = os.path.join(self.mosaic_dir, filt)
            os.makedirs(filt_dir, exist_ok=True)
            save_path = os.path.join(filt_dir, f"{gid}_bkg.png")

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
        median_bkg_residuals = bkg_results["median_res"]
        
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
        bkg_err_mjysr = bkg_results["rms"]
        
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
        median_bkg_res_jy = median_bkg_residuals * conv
        
        # 6. Final Results Dictionary
        return {
            "flux_jy": flux_jy,
            "flux_err_jy": err_jy,
            "snr": flux_jy / err_jy if err_jy > 0 else 0,
            "area_pix": source_ap.area,
            "bkg_median_jy": bkg_median_jy,
            "bkg_err_jy": bkg_err_jy,
            "median_bkg_res_jy": median_bkg_res_jy,
            "nominal_err_jy": nominal_err_jy
        }
    
    def measure_flux_cog(self, aperture_params, bkg_results, radii, psf_data=None, cog_dir=None):
        """
        Performs multi-aperture photometry to generate a Curve of Growth.
        """
        
        # 1. Setup Data and Metadata        
        filt = bkg_results["filter"]

        gid = aperture_params["meta"]["id"]
        err_map = aperture_params["err"]
        err_map = np.nan_to_num(err_map, nan=0.0, posinf=0.0, neginf=0.0)
        pixel_area_sr = aperture_params["meta"]["pixel_area_sr"]
        conv = 1e6 * pixel_area_sr
        
        # Extract fixed geometry from the small/original aperture
        x, y = aperture_params["x"], aperture_params["y"]
        theta = aperture_params["theta"]
        b_over_a = aperture_params["b"] / aperture_params["a"]
        
        
        # DYNAMIC DATA SELECTION: Use PSF if provided, otherwise use masked science data
        if psf_data is not None:
            cog_data = psf_data
            # Make sure the aperture is centred on the PSF
            x, y = centroid_com(psf_data)
            is_psf = True
        else:
            kill_mask = bkg_results["kill_mask"]
            cog_data = bkg_results["subtracted"].copy()
            cog_data[kill_mask] = 0.0
            is_psf = False
        
        # Plot and save single filter images
        if cog_dir:
            plt.imshow(cog_data, origin='lower', cmap='viridis')
            plt.title(f"{filt}: Neighbors Zeroed, Target Intact")
            
            fname = os.path.join(cog_dir, f"{filt}_masked.png")
            plt.savefig(fname, dpi=200, bbox_inches='tight')
            plt.close()
        
        cog_results = []

        # 2. Loop over radii to create the Curve of Growth
        for a_val in radii:
            # Scale b proportionally to maintain the galaxy's shape
            effective_a = a_val * 4 if is_psf else a_val
            effective_b = effective_a * b_over_a
            temp_ap = EllipticalAperture((x, y), a=effective_a, b=effective_b, theta=theta)
            
            # A. Sum Flux
            phot_table = aperture_photometry(cog_data, temp_ap, method='exact')
            raw_flux_mjysr = phot_table['aperture_sum'][0]
            
            # B. Error Propagation (Skip for PSF as it has no detector noise)
            if not is_psf:
                err_map = np.nan_to_num(aperture_params["err"], nan=0.0, posinf=0.0, neginf=0.0)
                ap_mask = temp_ap.to_mask(method='exact')
                det_var = np.nansum(ap_mask.multiply(err_map**2))
                bkg_err_mjysr = bkg_results["rms"] * np.sqrt(temp_ap.area)
                total_err_jy = np.sqrt(det_var + bkg_err_mjysr**2) * conv
            else:
                total_err_jy = 0.0
            
            # C. Store measurements for this radius
            cog_results.append({
                "radius_a": a_val,
                "area_pix": temp_ap.area,
                "flux_jy": raw_flux_mjysr * conv if not is_psf else raw_flux_mjysr,
                "flux_err_jy": total_err_jy * conv
            })

        return cog_results
    
    def get_filter_column_template(self, filt):
        """Defines the standard set of columns for any single MIRI band."""
        return {
            # --- Photometry Results ---
            f"{filt}_flux": np.nan,
            f"{filt}_flux_err": np.nan,
            f"{filt}_abmag": np.nan,
            f"{filt}_apflux": np.nan,
            f"{filt}_apflux_err": np.nan,
            f"{filt}_apflux_errnominal": np.nan,
            f"{filt}_apcorr": np.nan,
            
            # --- Background Statistics (Annulus) ---
            f"{filt}_bkg": np.nan,
            f"{filt}_bkg_err": np.nan,
            
            # --- Quality Control (QC) Flags ---
            #f"{filt}_qc_level": "UNKNOWN",   # CLEAN, WARNING, or CRITICAL
            #f"{filt}_qc_reasons": "",       # e.g., "HighSkew|ExtremeTails"
            
            # --- Aperture Geometry ---
            f"{filt}_ap_x": np.nan,
            f"{filt}_ap_y": np.nan,
            f"{filt}_ap_theta": np.nan,
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
    

    def apply_quality_flagging(self, bkg_residuals):
        """
        Performs a multi-variate statistical audit of the background.
        Returns: (str) Level, (str) Reason String, (dict) Raw Metrics
        """
        if len(bkg_residuals) < 200:
            return "WARNING", "InsufficientPixels", {}
        
        if len(bkg_residuals) < 50:
            return "CRITICAL", "InsufficientPixels", {}

        # --- A. Basic Moments ---
        res_skew = skew(bkg_residuals)
        res_kurt = kurtosis(bkg_residuals, fisher=True)
        
        # --- B. Robust RMS Ratio (Clipped vs MAD) ---
        std_val = np.std(bkg_residuals)
        # MAD scaled to match 1-sigma for a Gaussian
        mad = np.median(np.abs(bkg_residuals - np.median(bkg_residuals)))
        robust_std = 1.4826 * mad
        std_ratio = std_val / robust_std if robust_std > 0 else 1.0

        # --- C. Extreme Tail Fraction (>5-sigma) ---
        # Note: We use robust_std for the threshold to avoid the outliers masking themselves
        tail_threshold = 5 * robust_std
        outliers = np.abs(bkg_residuals) > tail_threshold
        tail_frac = np.mean(outliers)

        # --- D. Signed Tail Imbalance (Directionality) ---
        pos_tail = np.sum(bkg_residuals > tail_threshold)
        neg_tail = np.sum(bkg_residuals < -tail_threshold)
        
        # Check for imbalance if we have enough total outliers to be significant
        imbalance = 0
        if (pos_tail + neg_tail) > 2:
            imbalance = abs(pos_tail - neg_tail) / (pos_tail + neg_tail)

        # --- E. Logic-based Flagging ---
        reasons = []
        
        # Thresholds tuned for JWST/MIRI mosaics
        if abs(res_skew) > 2.5:       reasons.append("HighSkew")
        if res_kurt > 10.0:           reasons.append("HighKurtosis") # MIRI has high natural kurtosis
        if std_ratio > 1.5:           reasons.append("NonGaussianWidth")
        if tail_frac > 0.005:         reasons.append("ExtremeTails")
        if imbalance > 0.7:           reasons.append("AsymmetricTails")

        # Classification
        if len(reasons) >= 2 or "AsymmetricTails" in reasons:
            level = "CRITICAL"
        elif len(reasons) == 1:
            level = "WARNING"
        else:
            level = "CLEAN"

        return level, "|".join(reasons), {
            "skew": res_skew, 
            "kurtosis": res_kurt,
            "std_ratio": std_ratio, 
            "tail_frac": tail_frac,
            "imbalance": imbalance
        }

    def run_photometry(self, write_to, rescale=True, plot_mosaics=False, plot_psf=False):
        """
        Function to do the heavy lifting. Runs the entire photometry
        """
        
        # Stylised ASCII Header
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
        
        # Pre-scan for catalogue visualisation
        all_filters = self.pre_scan_filters()

        if rescale == False:
            print("⚠️ Processing photometry with original aperture sizes based on NIRCam/F444W...")
        
        all_rows = []
        
        bkg_floor = []
        
        stored_ids = 0
        
        for target_id in self.all_ids:
                
            files = self.find_files(target_id)
            if not files: 
                continue
            print("\n")
            print(f"========== Processing galaxy ID {target_id} ==========")
            
            # Base identity for the galaxy
            galaxy_row = {
                "ID": target_id,
                "MIRI_ap_a": np.nan,
                "MIRI_ap_b": np.nan,
                "MIRI_ap_npix": np.nan,
                #"Flag_Com": target_id in self.quality_config["has_companion"]
            }
            
            # Pre-populate with columns for all filters ALREADY discovered
            for filt in all_filters:
                galaxy_row.update(self.get_filter_column_template(filt))
            
            # Track if we've stored general aperture yet
            ap_geometry_stored = False
            
            for filt, file in files.items():
                print(f"{filt}:")
                
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
                    local_bkg = measurements["bkg_median_jy"] * n_pix
                    bkg_err = measurements["bkg_err_jy"]
                    
                    bkg_floor.append(measurements["median_bkg_res_jy"])
                    
                    # Get quality flagging
                    quality_level = bkg_res["quality_level"]
                    quality_reasons = bkg_res["quality_reasons"]
                    quality_metrics = bkg_res["quality_metrics"]
                    
                    # --- Store photometric measurements ---
                    galaxy_row[f"{filt}_flux"] = flux_corr
                    galaxy_row[f"{filt}_flux_err"] = flux_err_corr
                    galaxy_row[f"{filt}_abmag"] = ab_mag
                    galaxy_row[f"{filt}_apflux"] = apflux
                    galaxy_row[f"{filt}_apflux_err"] = apflux_err
                    galaxy_row[f"{filt}_apflux_errnominal"] = apflux_errnominal
                    galaxy_row[f"{filt}_apcorr"] = psf_corr
                    
                    # --- Store background statistics ---
                    galaxy_row[f"{filt}_bkg"] = local_bkg
                    galaxy_row[f"{filt}_bkg_err"] = bkg_err
                    
                    # --- Store quality flags ---
                    #galaxy_row[f"{filt}_qc_level"] = quality_level
                    #galaxy_row[f"{filt}_qc_reasons"] = quality_reasons
                    
                    # --- Store aperture geometry ---
                    galaxy_row[f"{filt}_ap_theta"] = float(np.degrees(ap_params["theta"].value))
                    galaxy_row[f"{filt}_ap_x"] = float(ap_params["x"])
                    galaxy_row[f"{filt}_ap_y"] = float(ap_params["y"])
                    
                    # --- Store "Scalar" values once ---
                    if not ap_geometry_stored:
                        galaxy_row["MIRI_ap_a"] = ap_params["a"]
                        galaxy_row["MIRI_ap_b"] = ap_params["b"]
                        galaxy_row["MIRI_ap_npix"] = n_pix
                        ap_geometry_stored = True
            
                except Exception as e:
                    print(f"Error processing {target_id} in {filt}: {e}")
            
            stored_ids += 1
            
            # Only add the row if we actually measured something
            if len(galaxy_row) > 1:
                all_rows.append(galaxy_row)

        print(f"\nMedian local background across {stored_ids} analysed galaxies: ", np.median(bkg_floor)*1e6, "µJy")

        # Convert to DataFrame and save to file
        df = pd.DataFrame(all_rows)
        
        self.save_catalogue(df, write_to)
        
        # Continue by creating detection statistics and storing them in a separate directory
        
        
        
        
        
        
                    
    def save_catalogue(self, df, base_filename):
        """Save photometric catalogue with explicit Astropy masking."""

        # ---------- CSV ----------
        csv_dir = os.path.join(self.output_dir, "phot_tables", "csv")
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"{base_filename}.csv")
        df.to_csv(csv_path, index=False)

        # ---------- FITS ----------
        fits_dir = os.path.join(self.output_dir, "phot_tables", "fits")
        os.makedirs(fits_dir, exist_ok=True)
        self.table_path = os.path.join(fits_dir, f"{base_filename}.fits")

        table = Table()

        for col_name in df.columns:
            # Convert series to a standard numpy array first
            col_data = df[col_name].values 

            # 1. Handle Floats (Fluxes, Mags, Coords) -> MaskedColumn
            if np.issubdtype(df[col_name].dtype, np.floating):
                data = col_data.astype(float)
                mask = np.isnan(data)
                table[col_name] = MaskedColumn(data=data, name=col_name, mask=mask, fill_value=np.nan)

            # 2. Handle Integers (IDs) -> Ensure 1D Array
            elif np.issubdtype(df[col_name].dtype, np.integer):
                table[col_name] = col_data.astype(int)

            # 3. Handle Everything Else (Strings, Booleans, Flags)
            else:
                # Converting to list and then array solves the "unsized object" error
                # for string-based or object-based columns
                table[col_name] = np.array(df[col_name].tolist())

        table.write(
            self.table_path,
            format="fits",
            overwrite=True,
            name="MIRI_PHOTOMETRY"
        )

        print(f"\n💾 Saving photometric catalogue to:\n1. {csv_path}\n2. {self.table_path}")

# ==================================================================================================
# ====================== END OF PHOTOMETRY - START OF ANALYSIS =====================================
# ==================================================================================================

    @staticmethod
    def compare_aperture_statistics(table_small_path, table_big_path, rescale_config_path=None, 
                                    fig_path=None, summary_doc_path=None, 
                                    target_scalings=[1.0, 2.0],
                                    snr=3.0):
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

        # Set style for better plots
        plt.style.use("default")
        sns.set_palette("husl")

        ts = Table.read(table_small_path)
        tb = Table.read(table_big_path)

        # Convert ID columns to string for alignment
        ids_small = [
            id.decode() if isinstance(id, bytes) else str(id) for id in ts["ID"]
        ]
        ids_big = [
            id.decode() if isinstance(id, bytes) else str(id) for id in tb["ID"]
        ]

        # 1. Filter IDs by scaling factor from config
        config = pd.read_csv(rescale_config_path, comment="#")
        # Ensure ID column in config is string for matching
        config['ID'] = config['ID'].astype(str)
        
        # Identify which galaxies have the scaling we care about
        filtered_config = config[config['Scale_Factor'].isin(target_scalings)]
        valid_ids = filtered_config['ID'].tolist()

        # Match common IDs
        common_ids = sorted(set(ids_small) & set(ids_big) & set(valid_ids))
        print(f"Found {len(common_ids)} common galaxies")
        
        # Reduce table to common IDs
        ts = ts[np.isin(ts['ID'], valid_ids)]
        tb = tb[np.isin(tb['ID'], valid_ids)]
        
        # Read bands from either catalogue (should be identical!)
        filters = [c.replace('_flux', '') for c in ts.colnames if c.endswith('_flux') and not c.startswith('ap')]
        
        # Final list to store per-band tables
        all_band_data = []
        
        for filt in filters:    
            # Join tables on ID for this specific band
            # Rename columns during join to avoid collisions
            cols_s = ['ID', f'{filt}_flux', f'{filt}_flux_err', f'{filt}_apflux', f'{filt}_apflux_err', f'{filt}_apcorr', f'{filt}_bkg_err']
            cols_b = ['ID', f'{filt}_flux', f'{filt}_flux_err', f'{filt}_apflux', f'{filt}_apflux_err', f'{filt}_apcorr', f'{filt}_bkg_err']
            
            # Select and rename
            ts_filt = ts[cols_s]
            for c in cols_s[1:]: ts_filt.rename_column(c, c + "_small")
            
            tb_filt = tb[cols_b]
            for c in cols_b[1:]: tb_filt.rename_column(c, c + "_big")

            # Join ensures IDs match perfectly
            matched = join(ts_filt, tb_filt, keys='ID', join_type='inner')
            
            # VECTORIZED CALCULATIONS (No loops!)
            flux_s = matched[f'{filt}_flux_small']
            flux_b = matched[f'{filt}_flux_big']
            apflux_s = matched[f'{filt}_apflux_small']
            apflux_b = matched[f'{filt}_apflux_big']
            apflux_err_s = matched[f'{filt}_apflux_err_small']            
            apflux_err_b = matched[f'{filt}_apflux_err_big']            
            apcorr_s = matched[f'{filt}_apcorr_small']
            apcorr_b = matched[f'{filt}_apcorr_big']
            
            # Rough detection filtering based on SNR
            snr_s = apflux_s / apflux_err_s
            snr_b = apflux_b / apflux_err_b
            
            # Clean NaNs/Negatives
            mask = (flux_s > 0) & (flux_b > 0) & (apflux_s > 0) & (apflux_b > 0) \
                    & np.isfinite(apcorr_s) & np.isfinite(apcorr_b) \
                    & (snr_s > snr) & (snr_b > snr)
            res = matched[mask]
            
            # Add the calculated columns
            res['Band'] = filt
            res['Corrected_Flux_Ratio'] = res[f'{filt}_flux_big'] / res[f'{filt}_flux_small']
            res['Raw_Flux_Ratio'] = res[f'{filt}_apflux_big'] / res[f'{filt}_apflux_small']
            
            # Now contains one compressed table per band!
            all_band_data.append(res)

            print(res[res["Corrected_Flux_Ratio"] > 3.0])
        
        # Combine everything into one master table
        final_table = vstack(all_band_data)
        
        # Convert to dictionary for your existing pickle/plot functions if needed
        data_comparison = {col: final_table[col].tolist() for col in final_table.colnames}
            
        if fig_path:
            MIRIPipeline.plot_aperture_comparison(data_comparison, fig_path)
            print(f"Saved output plot to {fig_path}")
            

    @staticmethod
    def plot_aperture_comparison(data_comparison, fig_dir, scaling='log'):
        # Convert to arrays for vectorized masking
        for key in data_comparison:
            data_comparison[key] = np.array(data_comparison[key])

        # Dynamic filter identification
        bands = []
        for band in data_comparison["Band"]:
            if band not in bands:
                bands.append(band)
        
        n_bands = len(bands)
        
        # Create a grid: 3 columns (Comparison, Ratio, Systematic Trend) x n_bands rows
        fig, axes = plt.subplots(n_bands, 3, figsize=(13, 4 * n_bands), squeeze=False)
        
        colors = {'scatter': '#1f77b4', 'ratio': "#af77e3", 'trend': '#2ca02c'}

        for i, band in enumerate(bands):
            mask = data_comparison["Band"] == band
            
            # Data extraction for brevity
            flux_s = data_comparison[f"{band}_flux_small"][mask] * 1e6
            
            flux_b = data_comparison[f"{band}_flux_big"][mask] * 1e6
            ratio_corr = data_comparison["Corrected_Flux_Ratio"][mask]

            # --- PANEL 1: Corrected Flux Comparison (1:1) ---
            ax = axes[i, 0]
            ax.scatter(flux_s, flux_b, alpha=0.6, s=25, color=colors['scatter'], edgecolors='black', linewidth=0.3)
            
            # 1:1 Line logic
            lims = [np.min([flux_s, flux_b]), np.max([flux_s, flux_b])]
            ax.plot(lims, lims, 'r--', alpha=0.8, zorder=0, label='1:1')
            
            if scaling == 'log':
                ax.set_xscale('log')
                ax.set_yscale('log')
                
            ax.set_title(f"{band}: Corrected Flux Agreement", fontweight='bold')
            ax.set_xlabel("Small Aperture [µJy]")
            ax.set_ylabel("Large Aperture [µJy]")
            ax.grid(True, alpha=0.2)

            # --- PANEL 2: Corrected Flux Ratio Distribution ---
            ax = axes[i, 1]
            ax.hist(ratio_corr, bins=np.linspace(0.5, 3.5, 25), alpha=0.7, color=colors['ratio'], edgecolor='black', linewidth=0.5)
            #ax.hist(ratio_corr, bins=25, alpha=0.7, color=colors['ratio'], edgecolor='black', linewidth=0.5)
            ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="Unity")
            
            med = np.median(ratio_corr)
            std = np.std(ratio_corr)
            ax.axvline(med, color="darkred", linestyle="-", linewidth=1.5, label=f'Med: {med:.3f}')
            
            ax.set_title(f"{band}: Ratio Distribution", fontweight='bold')
            ax.set_xlabel("Ratio (Large/Small)")
            ax.set_ylabel("N Sources")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2)

            # --- PANEL 3: Corrected Ratio vs Brightness (Systematics) ---
            ax = axes[i, 2]
            # Color code by the magnitude of the aperture correction to see if PSF issues drive scatter
            sc = ax.scatter(flux_s, ratio_corr, alpha=0.7, s=30, edgecolor="black",
                            c=data_comparison[f"{band}_apcorr_big"][mask], cmap='viridis')
            ax.axhline(1.0, color="red", linestyle="--", alpha=0.8, label="Unity")
            ax.axhline(med, color="darkred", linestyle="-", linewidth=1.5, label=f'Med: {med:.3f}', alpha=0.6)
            
            ax.set_xscale('log')
            #ax.set_ylim(0.5, 3.0) # Focus on the 50% deviation window
            ax.set_title(f"{band}: Systematic Trends", fontweight='bold')
            ax.set_xlabel("Flux [µJy]")
            ax.set_ylabel("Ratio (Large/Small)")
            ax.set_ylim(0.5)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2)
            
            if i == 0: # Add colorbar only to top row to save space
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label('Apr Corr (Big)', fontsize=10)

        plt.suptitle("Aperture Consistency Diagnostic (Corrected Fluxes Only)", fontsize=16, y=1.02)
        plt.tight_layout()
        
        if fig_dir:
            os.makedirs(fig_dir, exist_ok=True)
            save_path = os.path.join(fig_dir, "apercorrs.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved condensed aperture diagnostic to {save_path}")
        
        plt.show()
        plt.close()


    def run_cog_analysis(self, gid, radii, cog_dir, overplot_psf=False):
        """Function to perform Curve of Growth analysis for extended sources"""

        # Find files associated with galaxy ID
        files = self.find_files(gid)
        
        if files is None:
            return None
        
        if len(files) < 2:
            return None
        
        # Define output directory
        cog_dir = os.path.join(self.output_dir, "apertures", cog_dir)
        os.makedirs(cog_dir, exist_ok=True)
        
        cog_results = {}
        cog_results_psf = {}
        
        for filt, file in files.items():
            try:
                # 1. Prepare the apertures for MIRI
                ap_params = self.prepare_aperture(file, rescale=True)
                
                # 2. Create background model
                bkg_res = self.estimate_background(ap_params)
                
                # 3. Measure fluxes by performing Curve of Growth (CoG) analysis
                measurements = self.measure_flux_cog(ap_params, bkg_res, radii)
            
                cog_results[filt] = measurements
                cog_results["meta"] = ap_params
            
                if overplot_psf:
                    psf_path = os.path.join(self.psf_dir, f"PSF_MIRI_{filt}.fits")
                    with fits.open(psf_path) as psf_hdul:
                        psf_data = psf_hdul[3].data
                    measurements_psf = self.measure_flux_cog(ap_params, bkg_res, radii, psf_data=psf_data)

                    cog_results_psf[filt] = measurements_psf
                    cog_results_psf["meta"] = ap_params
                    
            except Exception as e:
                print(f"Error processing {gid} in {filt}: {e}")
                
        # Plotting Logic
        if not cog_results:
            return
        
        plt.figure(figsize=(8, 5))
        
        # --- AUTOMATED COLOUR LOGIC ---
        # Setup normalisation based on MIRI wavelength range (5 to 26 microns)
        norm = mcolors.Normalize(vmin=5.6, vmax=25.5)
        colormap = cm.get_cmap('jet')
        
        # Track available filters to avoid plotting the 'meta' key
        available_filters = [f for f in cog_results.keys() if f != "meta"]
        ap_meta = cog_results["meta"]
        
        # Sort filters by wavelength so the legend looks organized
        available_filters.sort(key=lambda x: self.wavelength_map.get(x, 0))

        for filt in available_filters:
            w_val = self.wavelength_map.get(filt, 15.0)
            line_color = colormap(norm(w_val))
            
            if overplot_psf:
                # Use normalised flux for direct shape comparison
                galaxy_flux = [r['flux_jy'] for r in cog_results[filt]]
                psf_flux = [r['flux_jy'] for r in cog_results_psf[filt]]
                
                y_galaxy = galaxy_flux / np.max(galaxy_flux)
                y_psf = psf_flux / np.max(psf_flux)
                
                plt.plot(radii, y_galaxy, label=f'{filt}', color=line_color, 
                        marker='o', markersize=3, lw=2, alpha=0.9)
                plt.plot(radii, y_psf, color=line_color, 
                        linestyle='--', lw=1.5, alpha=0.7)
                plt.ylabel("Normalised Cumulative Flux")
            else:
                # Use absolute units if not overplotting
                y_galaxy = [r['flux_jy'] * 1e6 for r in cog_results[filt]]
                plt.plot(radii, y_galaxy, label=filt, color=line_color, 
                        marker='o', markersize=3, lw=2, alpha=0.9)
                plt.ylabel("Cumulative Flux [µJy]")

        # Use 'a' for Big Aperture and your previous 'a' (before +8) for Small
        # Adjust these keys based on how you stored them in prepare_aperture
        small_limit = ap_meta["a_orig"]
        big_limit = ap_meta["a"]

        plt.axvline(small_limit, color='orange', linestyle='--', label='Small Aperture Limit', alpha=0.8)
        plt.axvline(big_limit, color='dodgerblue', linestyle='--', label='Big Aperture Limit', alpha=0.8)
        
        plt.xlabel("Semi-major axis (pixels)")
        plt.ylabel("Cumulative Flux [µJy]")
        plt.title(f"Curve of Growth (CoG) Analysis: {gid}")
        plt.legend()
        plt.grid(alpha=0.2)
        
        fname = f"{gid}_cog_psf.png" if overplot_psf else f"{gid}_cog.png"
        save_path = os.path.join(cog_dir, fname)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close() # Close figure to free memory
        
        return cog_results
    
    @staticmethod
    def get_detection_stats(table_path, out_dir, snr=3.0):
        """Code to generate detection statistics of a given photometric table"""
        
        # Read table and generate output directory
        table = Table.read(table_path)
        os.makedirs(out_dir, exist_ok=True)

        # Read bands from either catalogue (should be identical!)
        filters = [c.replace('_flux', '') for c in table.colnames if c.endswith('_flux') and not c.startswith('ap')]
        
        nondetections = {}
        
        for filt in filters:    
            # Join tables on ID for this specific band
            # Rename columns during join to avoid collisions
            cols = ['ID', f'{filt}_flux', f'{filt}_flux_err', f'{filt}_apflux', f'{filt}_apflux_err', f'{filt}_apcorr', f'{filt}_bkg_err']
            
            # VECTORISED CALCULATIONS (No loops!)
            flux = table[f'{filt}_flux']
            flux_err = table[f'{filt}_flux_err']
            apflux = table[f'{filt}_apflux']
            apflux_err = table[f'{filt}_apflux_err']            
            
            # Rough detection filtering based on SNR
            signal_to_noise = apflux / apflux_err
            
            # Clean NaNs/Negatives
            mask = (flux > 0) & (apflux > 0) & (signal_to_noise >= snr)
            
            nondetected = table[~mask]
            
            # Extract table IDs
            ids = [id.decode() if isinstance(id, bytes) else str(id) for id in nondetected["ID"]]
            
            nondetections[filt] = ids
            
        return nondetections
            
    @staticmethod
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
            table_id_to_row = {str(row["ID"]): ii for ii, row in enumerate(table)}

            for row in table:
                gid = str(row["ID"])
                if gid not in g_index_map:
                    continue
                g_ii = g_index_map[gid]
                filters = row["Filters"]
                if isinstance(filters, (list, np.ndarray)):
                    filters = [
                        f.decode() if isinstance(f, bytes) else str(f) for f in filters
                    ]
                else:
                    filters = [f.strip() for f in str(filters).split(",") if f.strip()]

                for filt in filters:
                    if filt in filter_order:
                        f_ii = filter_order.index(filt)

                        # Inverted logic:
                        # Galaxy is marked if it's covered AND not in nondetections for that filter
                        if nondetections is None or int(gid) not in nondetections.get(
                            filt, []
                        ):
                            matrix[g_ii, f_ii] = 1

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
 