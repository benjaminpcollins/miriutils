#!/usr/bin/env python
# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "astropy",
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
      physical values to PSF model predictions and identify potential detector
      artefacts
    - Creates heat map plots of observations and detections across all MIRI bands

Workflow:
---------
    1. Pre-scans directories to initialise a type-safe, wavelength-ordered table structure.
    2. Aligns science apertures based on Blue Jay (NIRCam F444W) morphology.
    3. Performs exact aperture photometry and background estimation.
    4. Calculates band-specific PSF corrections on an oversampled grid.
    5. Exports a dual-format (FITS/CSV) catalogue with standardised columns.
    6. Creates heat map plots for observation and detection statistics across all MIRI bands.

Example usage:
--------------
    from miri_utils import MiriPipeline

    ids_to_process = [7102, 11202, 16874]   # int of galaxy IDs to process
    
    # Initialise the pipeline
    pipeline = MiriPipeline(
        table_name="Phot_Table_MIRI",
        all_ids=ids_to_process,
        cutout_dir="./data/cutouts",
        output_dir="./miri_photometry",
        nircam_dir="./NIRCam/cutouts",
        aperture_table="./data/aperture_table.fits"
    )

    # Run full survey photometry and store FITS and CSV format output tables
    pipeline.run_photometry()


Author: Benjamin P. Collins
Date: April 2026
Version: 2.1.0
"""

import os
import warnings
import json
from pathlib import Path
import seaborn as sns
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from scipy.stats import skew, kurtosis, median_abs_deviation

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
    def __init__(self, 
                 table_name,
                 ids_to_process, 
                 cutouts_dir, 
                 photometry_dir, 
                 nircam_dir, 
                 aperture_table, 
                 psf_dir=None, 
                 scaling_exceptions_file="scaling_config.csv"):
        
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
        
        # Load aperture table
        self.master_table = Table.read(aperture_table)
        
        # Output directories for the photometry tables (both CSV and FITS)
        self.csv_dir = os.path.join(self.output_dir, "phot_tables", "csv")
        self.fits_dir = os.path.join(self.output_dir, "phot_tables", "fits")   
        
        # Specify table path
        self.table_path = os.path.join(self.fits_dir, f"{table_name}.fits")
        self.table_name = table_name
        
        # Default PSF directory if none provided
        if psf_dir is None:
            self.psf_dir = os.path.join(self.output_dir, "psfs")
            print(f"Using default PSF directory: {self.psf_dir}")
        else:
            self.psf_dir = psf_dir
            print(f"Found PSF directory {self.psf_dir}")
        
        self.aperture_dir = None
        self.mosaic_dir = None
        self.cog_dir = None
        self.detection_dir = None

        self.phot_table_dir = os.path.join(self.output_dir, "phot_tables")
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.fits_dir, exist_ok=True)
        
        # 2. Handle Scaling Exceptions File
        if scaling_exceptions_file is None:
            # Default name in the output directory if none provided
            scaling_exceptions = os.path.join(self.output_dir, "scaling_config.csv")
        else:
            scaling_exceptions = os.path.join(self.output_dir, scaling_exceptions_file)
        
        self.scaling_exceptions_path = scaling_exceptions
        self.scaling_exceptions = self._initialise_scaling_config()
        
        # Placeholder for adding a custom scaling factor to the fluxes if needed
        self.flux_scaling_factor = 1.0
        self.added_flux_error = 0.0

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
            "multiplier": multiplier
            #"pixel_conversion": pixel_conversion
        }
    
    def plot_apertures_multiband(self, ap_params):
        """
        results_list: A list of dictionaries returned by prepare_aperture for ONE galaxy.
        """
        
        self.aperture_dir = os.path.join(self.output_dir, "apertures")
        os.makedirs(self.aperture_dir, exist_ok=True)
        
        # Sort by wavelength: Extract the number from 'F770W', 'F1000W', etc.
        # This ensures F770W comes before F1000W
        ap_params.sort(key=lambda x: int(''.join(filter(str.isdigit, x['meta']['filter']))))
        n_bands = len(ap_params)
        fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 4), squeeze=False)
        
        for i, params in enumerate(ap_params):
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
        
        galaxy_id = ap_params[0]['id']
        
        plt.suptitle(f"Galaxy ID: {galaxy_id}", fontsize=14)
        out_path = os.path.join(self.aperture_dir, f"{galaxy_id}_all.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        

    def estimate_background(self, aperture_params, sigma_val=2.5, n_iters=3, n_random=200):
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
        
        img_h, img_w = data.shape
        yi, xi = np.indices(data.shape)

        # 1. Create an two source masks (buffer around the aperture)
        source_ap = EllipticalAperture((x_cen, y_cen), a=a, b=b, theta=theta)
        source_mask = source_ap.to_mask(method='center').to_image(data.shape).astype(bool)
        
        # The large source mask prevents the target's own light from biasing the background
        a_in, b_in = a + img_h//10, b + img_h//10
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
                break
                
            fit_mask = new_fit_mask

        # Create mask for CoG analysis
        non_sky_mask = ~fit_mask
        kill_mask = non_sky_mask.copy()
        kill_mask[source_mask_large] = True # THIS DEFINITELY HAS TO BE TRUE TO EXCLUDE THE SOURCE!!!
        
        # Final Plane Generation
        background_plane = alpha * xi + beta * yi + gamma
        data_bkgsub = data - background_plane
        
        # Insert section for empirical rms caculation
        noise_image = data_bkgsub.copy()
        noise_image[kill_mask] = np.nan        
        
        emp_rms = self.empirical_aperture_rms(
            noise_image, 
            aperture_params, 
            n_random=n_random
            )        

        # Reset kill mask for the CoG analysis
        kill_mask[source_mask_large] = False
        
        # Define elliptical annulus for local background estimate 
        # Set dynamic outer radius based on image bounds to prevent crashes
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

        background_median = np.median(plane_vals)

        # The Noise (Robust RMS / Sigma)
        # We use 1.4826 * MAD to match the scale of a standard deviation
        mad = np.median(np.abs(res_vals - np.median(res_vals)))
        background_rms = 1.4826 * mad 
        
        # Initialise mask visualisation for plotting
        mask_vis = np.zeros_like(data, dtype=int)   # Invalid pixels
        mask_vis[new_fit_mask] = 1                  # Annulus without sigma clipping
        mask_vis[source_mask] = 2                   # Aperture   
        
        aperture_map = {
            "x": float(aperture_params.get("x")),
            "y": float(aperture_params.get("y")),
            "a": float(aperture_params.get("a")),
            "b": float(aperture_params.get("b")),
            "theta": float(aperture_params.get("theta").value if hasattr(aperture_params.get("theta"), 'value') 
                   else aperture_params.get("theta")),
        }        
        
        return {
            "id": galaxy_id,
            "filter": filt,
            "median": background_median,
            "median_res": np.median(res_vals),
            "rms": background_rms,
            "emp_rms": emp_rms if emp_rms is not None else 0.0,
            "data": data,
            "bkg_plane": background_plane,
            "subtracted": data_bkgsub,
            "annulus": annulus,
            "source_ap": source_ap,
            "mask_vis": mask_vis,
            "kill_mask": kill_mask
        }

    def plot_background_diagnostic(self, aperture_params, bkg_dict):
        """
        Creates a 2x2 diagnostic mosaic to verify background modeling.
        """
        
        # Initialise mosaic directory
        self.mosaic_dir = os.path.join(self.output_dir, "mosaic_plots")
        
        data = aperture_params["data"]
        gid = aperture_params["id"]
        filt = aperture_params["meta"]["filter"]
        survey_obs = aperture_params["meta"]["survey_obs"]
        
        # Extract calculated objects from bkg_dict
        plane = bkg_dict["bkg_plane"]
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
    
        # Default directory structure
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
        

    def measure_flux(self, aperture_params, bkg_results):
        """
        Performs aperture photometry, unit conversion, and error propagation.
        """
        # Extract data from dictionaries
        ap_meta = aperture_params["meta"]
        filt = ap_meta["filter"]
        gid = ap_meta["id"]
        pixel_area_sr = ap_meta["pixel_area_sr"] # From PIXAR_SR header
        
        # Get data from dictionaries
        data_bkgsub = bkg_results["subtracted"]
        err_map = np.nan_to_num(aperture_params["err"], nan=0.0)
        background_plane = bkg_results["bkg_plane"]
        
        # Get aperture object and background statistics
        source_ap = bkg_results["source_ap"]
        median_bkg_residuals = bkg_results["median_res"]
        emp_rms = bkg_results["emp_rms"]
        
        # Perform photometry on the background-subtracted data
        phot_table = aperture_photometry(data_bkgsub, source_ap, method='exact')
        raw_flux_mjysr = phot_table['aperture_sum'][0]
        
        # Perform photometry on the background plane
        phot_table_plane = aperture_photometry(background_plane, source_ap, method='exact')
        raw_bkg_flux_mjysr = phot_table_plane['aperture_sum'][0]
        
        # Detector/Poisson noise from the ERR extension
        ap_mask = source_ap.to_mask(method='exact')
        
        # Weighted sum of variance (method='exact' accounts for fractional pixels)
        detector_variance = np.nansum(ap_mask.multiply(err_map**2))
        
        # Background modelling uncertainty already calculated in estimate_background
        bkg_err_mjysr = bkg_results["rms"] * np.sqrt(source_ap.area)
        
        # Combine in quadrature (Detector + Background Modelling)
        total_err_mjysr = np.sqrt(detector_variance + bkg_err_mjysr**2)
        
        # --- NEW SCALING AND ERROR PADDING LOGIC ---
        # 1. Apply the empirical scaling factor (e.g. 1.182)
        flux_scaled_mjysr = raw_flux_mjysr * self.flux_scaling_factor
        
        # 2. Add the Intrinsic Scatter (MAD) in quadrature to the error
        # This accounts for the ~11% uncertainty in the "extendedness" correction
        # We use the scaled flux as the base for the systematic error component
        intrinsic_scatter = 0.110  # This matches your MAD result
        systematic_err_mjysr = flux_scaled_mjysr * intrinsic_scatter
        
        # Update the total error budget
        total_err_scaled_mjysr = np.sqrt(total_err_mjysr**2 + systematic_err_mjysr**2)
        
        # Conversion = 1e6 (to Jy) * pixel_area_sr (to strip sr)
        conv = 1e6 * pixel_area_sr
        
        # Unit Conversion (MJy/sr -> Jy)
        flux_jy = flux_scaled_mjysr * conv
        bkg_flux_jy = raw_bkg_flux_mjysr * conv
        total_err_jy = total_err_scaled_mjysr * conv
        emp_rms_jy = emp_rms * conv
        
        print("Emp_RMS:", emp_rms_jy, "Propagated Error:",total_err_jy)
        
        # This is the "Nominal" error in Jy
        nominal_err_jy = np.sqrt(detector_variance) * conv
        
        # Obtain local background and error
        bkg_median_jy = bkg_results["median"] * conv
        bkg_err_jy = bkg_err_mjysr * conv
        median_bkg_res_jy = median_bkg_residuals * conv
        
        # Add-in: Curve of Growth analysis!!
        nx, ny = data_bkgsub.shape
        max_ap_size = nx//3
        
        # Start from 2 pixels and increase towards the max_ap_size
        radii = np.linspace(2, max_ap_size, 25)
        
        # Get aperture geometry
        x, y = aperture_params["x"], aperture_params["y"]
        theta = aperture_params["theta"]
        a, b = aperture_params["a"], aperture_params["b"]
        b_over_a = b/a
        
        # Get kill mask and recreate noise image
        kill_mask = bkg_results["kill_mask"]
        cog_data = data_bkgsub.copy()
        cog_data[kill_mask] = 0.0
        
        # 1. Pre-calculate PSF metadata once (outside the loop)
        psf_path = os.path.join(self.psf_dir, f"PSF_MIRI_{filt}.fits")
        with fits.open(psf_path) as psf_hdul:
            psf_data = psf_hdul[3].data
            px, py = centroid_com(psf_data)

        cog_fluxes = []
        cog_fluxes_psf = []
        cog_radii = []

        # 2. Combined Loop
        for r in radii:
            # Define both apertures
            ap_sci = EllipticalAperture((x, y), a=r, b=r*b_over_a, theta=theta)
            ap_psf = EllipticalAperture((px, py), a=4*r, b=4*r*b_over_a, theta=theta)
            
            # Photometry (exact method is computationally expensive, so we group them)
            phot_sci = aperture_photometry(cog_data, ap_sci, method='exact')
            phot_psf = aperture_photometry(psf_data, ap_psf, method='exact')
            
            cog_fluxes.append(phot_sci['aperture_sum'][0] * conv)
            cog_fluxes_psf.append(phot_psf['aperture_sum'][0])
            cog_radii.append(r)
        
        return {
            # Science fluxes and uncertainties
            "flux_jy": flux_jy,
            "flux_err_jy": max(total_err_jy, emp_rms_jy), # Use propagated error if it's bigger than the empirical RMS
            
            # Background statistics and aperture area (just to be sure)
            "n_pix": source_ap.area,
            "bkg_flux_jy": bkg_flux_jy,
            "bkg_median_jy": bkg_median_jy,
            "bkg_err_jy": bkg_err_jy,
            "median_bkg_res_jy": median_bkg_res_jy,
            "nominal_err_jy": nominal_err_jy,
            
            # CoG results
            "cog_fluxes": cog_fluxes,
            "cog_fluxes_psf": cog_fluxes_psf,
            "cog_radii": cog_radii
        }
        
    def _plot_cog(self, cog_dict):
        """Function to perform Curve of Growth analysis for extended sources"""

        self.cog_dir = os.path.join(self.output_dir, "curve_of_growth")
        os.makedirs(self.cog_dir, exist_ok=True)

        plt.figure(figsize=(8, 5))
        
        # --- AUTOMATED COLOUR LOGIC ---
        # Setup normalisation based on MIRI wavelength range (5 to 26 microns)
        norm = mcolors.Normalize(vmin=5.6, vmax=25.5)
        colormap = cm.get_cmap('turbo')
        
        # Track available filters to avoid plotting the 'meta' key
        available_filters = [f for f in cog_dict.keys() if f != "ap_params" and f != "qc_flag"]
        if len(available_filters) == 0:
            return None
        ap_params = cog_dict["ap_params"]
        gid = ap_params["meta"]["id"]
        
        # Sort filters by wavelength so the legend looks organized
        available_filters.sort(key=lambda x: self.wavelength_map.get(x, 0))

        for filt in available_filters:
            w_val = self.wavelength_map.get(filt, 15.0)
            line_color = colormap(norm(w_val))
            
            # Use normalised flux for direct shape comparison
            cog_data = cog_dict[filt]
            galaxy_flux = cog_data['cog_fluxes']
            psf_flux = cog_data['cog_fluxes_psf']
            radii = cog_data['cog_radii']
            qc_flag = cog_data["qc_flag"]
            
            # Normalise by maximum absolute value instead of just positive maximum!
            y_galaxy = galaxy_flux / np.max(np.abs(galaxy_flux))
            y_psf = psf_flux / np.max(psf_flux)
            
            plt.plot(radii, y_galaxy, label=f'{filt} ({qc_flag})', color=line_color, 
                    marker='o', markersize=3, lw=2, alpha=0.9)
            plt.plot(radii, y_psf, color=line_color, 
                    linestyle='--', lw=1.5, alpha=0.7)
        
        # Get aperture parameters
        small_ap = ap_params["a_orig"]
        big_ap = ap_params["a"]

        plt.axvline(small_ap, color='orange', linestyle='--', label='Small Aperture Limit', alpha=0.8)
        plt.axvline(big_ap, color='dodgerblue', linestyle='--', label='Big Aperture Limit', alpha=0.8)
        
        plt.xlabel("Semi-major axis (pixels)")
        plt.ylabel("Normalised Cumulative Flux")        
        plt.title(f"Curve of Growth (CoG) Analysis: {gid}")
        plt.legend()
        plt.grid(alpha=0.2)
        
        # Save file to cog directory
        fname = f"{gid}_cog_psf.png"
        save_path = os.path.join(self.cog_dir, fname)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close() 
        
    
    def _apply_quality_flagging(self, flux_dict):
        """
        Performs a quality based on the curve of growth.
        """
        
        # Return data:
        # Quality flag, flux_jy, flux_err_jy
        
        # Actual flux of the observation
        flux_jy = flux_dict["flux_jy"]          # Measured flux
        flux_err_jy = flux_dict["flux_err_jy"]  # Propagated error
        
        # Extract CoG-fluxes
        fluxes = np.array(flux_dict["cog_fluxes"])
        radii = np.array(flux_dict["cog_radii"])
        
        # Get important diagnostic flux values
        f_max = np.max(fluxes)          # Peak flux in the CoG
        f_min = np.min(fluxes)          # Minimum flux in the CoG
        idx_max = np.argmax(fluxes)     # Position of the peak
        idx_min = np.argmin(fluxes)     # Position of the minimum
        f_final = fluxes[-1]            # Flux value at largest aperture
        num_points = len(fluxes)        # Number of points in the curve
        tail_fluxes = fluxes[idx_max:]  # Flux range from the peak to the end
        diffs = np.diff(fluxes)         # Differences between consecutive fluxes
        neg_fraction = np.sum(diffs < 0) / len(diffs)   # Fraction of aperture steps adding negative flux
        pos_fraction = np.sum(diffs > 0) / len(diffs)   # Fraction of aperture steps adding positive flux
        
        # Quality Flagging Logic
        # ---------------------------------------------------------
        
        # If the curve is significantly negative at the end AND the majority of steps are negative, 
        # it's likely an oversubtraction issue
        is_deeply_negative = f_min < -3.0 * flux_err_jy
        is_consistently_falling = neg_fraction > 0.80
        if f_final < 0 and is_deeply_negative and is_consistently_falling:
            return "OVERSUB", 1

        # If peak is at the end AND the growth in the last 3 steps is still high
        is_highly_positive = f_max > 3.0 * flux_err_jy
        is_consistently_rising = pos_fraction > 0.80
        last_growth = (fluxes[-1] - fluxes[-4]) / fluxes[-1] if fluxes[-1] > 0 else 0
        if idx_max >= num_points - 4 and last_growth > 0.05 and is_consistently_rising and is_highly_positive:
            return "UNDERSUB", 2

        # The dip & bump (background gradient/stripe)
        tail_fluxes_peak = fluxes[idx_max:]        
        f_min_after_peak = np.min(tail_fluxes_peak)
        depth_absolute = f_max - f_min_after_peak
        is_significant_dip = depth_absolute > (3.0 * flux_err_jy)
        depth_fraction = depth_absolute / f_max if f_max > 0 else 0
        
        # 20% loss after peak is highly unphysical
        if idx_max != len(fluxes)-1 and depth_fraction > 0.20 and is_significant_dip:  
            return "ARTEFACT", 3
        
        # Clean data
        return "CLEAN", 0
    
    
    def _get_filter_column_template(self, filt):
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
            
            # --- Quality Control (QC) Flag ---
            f"{filt}_qc_flag": 0,
            f"{filt}_ap_flag": 0,
            
            # --- Aperture Geometry ---
            f"{filt}_ap_x": np.nan,
            f"{filt}_ap_y": np.nan,
            f"{filt}_ap_theta": np.nan,
        }

    def _pre_scan_filters(self):
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
    

    

    def run_photometry(self, 
                       rescale_apertures=True,      # Whether to rescale the apertures
                       save_mosaic=False,           # Whether to save the background diagnostic mosaic
                       plot_psf=False,              # Whether to plot the PSF with the aperture overlay
                       save_cog=False,              # Whether to save the Curve of Growth plots
                       snr_thresh=3.0,              # Relevant for the detection plot
                       plot_matrix=False,
                       plot_apertures=False,
                       flux_scaling_factor=None
                       ):
        """
        Function to do the heavy lifting. Runs the entire photometry
        """
        
        # Stylised ASCII Header
        print("\n" + "="*60)
        print(r"""
 ____           _    ____              _ _             _ 
|  _ \ ___   __| |  / ___|__ _ _ __ __| (_)_ __   __ _| |
| |_) / _ \ / _` | | |   / _` | '__/ _` | | '_ \ / _` | |
|  _ <  __/| (_| | | |__| (_| | | | (_| | | | | | (_| | |
|_| \_\___| \__,_|  \____\__,_|_|  \__,_|_|_| |_|\__,_|_|
        """)
        print("                 JWST MIRI PIPELINE v2.1.0")
        print("                 MIRI Photometry for JWST")
        print("="*60)
        
        # Pre-scan for catalogue visualisation
        all_filters = self._pre_scan_filters()

        if rescale_apertures == False:
            print("⚠️ Processing photometry with original aperture sizes based on NIRCam/F444W...")
        
        if flux_scaling_factor is not None and flux_scaling_factor != 1.0:
            print(f"⚠️ Attention: Rescaling fluxes by a user-specified factor of {flux_scaling_factor} (default is 1.0)")
            self.flux_scaling_factor = flux_scaling_factor[0]
            self.added_flux_error = flux_scaling_factor[1]
        
        all_rows = []
        
        stored_ids = 0
        
        for target_id in self.all_ids:
                
            files = self.find_files(target_id)
            if not files: 
                continue
            
            print(f"Processing galaxy ID {target_id}...")
            
            galaxy_cog_dict = {}
            
            # Base identity for the galaxy
            galaxy_row = {
                "ID": target_id,
                "MIRI_ap_a": np.nan,
                "MIRI_ap_b": np.nan,
                "MIRI_ap_npix": np.nan
            }
            
            # Pre-populate with columns for all filters ALREADY discovered
            for filt in all_filters:
                galaxy_row.update(self._get_filter_column_template(filt))
            
            # Track if we've stored general aperture yet
            ap_geometry_stored = False
            
            for filt, file in files.items():
                
                try:
                                        
                    # --- 1. Prepare the apertures for MIRI ---
                    ap_params = self.prepare_aperture(file, rescale=rescale_apertures)                      
                    
                    if plot_apertures is True:
                        self.plot_apertures_multiband(ap_params)
                    
                    # --- 2. Create and subtract background model ---
                    bkg_dict = self.estimate_background(ap_params)
                    
                    if save_mosaic is True:
                        self.plot_background_diagnostic(ap_params, bkg_dict)
                    
                    # --- 3. Measure fluxes ---
                    flux_dict = self.measure_flux(ap_params, bkg_dict)
                    
                    qc_flag, qc_identifier = self._apply_quality_flagging(flux_dict)              
                    galaxy_cog_dict[filt] = flux_dict
                    galaxy_cog_dict[filt]["qc_flag"] = qc_flag

                    # Ensure that the aperture params are available for the plotter
                    if "ap_params" not in galaxy_cog_dict:
                        galaxy_cog_dict["ap_params"] = ap_params                
                    
                    # --- 5. Compute PSF correction ---
                    psf_corr = self.calculate_psf_corr(ap_params, show_plot=plot_psf)
                    
                    # --- 6. Group all results and correct them for the PSF ---
                    apflux = flux_dict["flux_jy"]            
                    apflux_err = flux_dict["flux_err_jy"]
                    apflux_errnominal = flux_dict["nominal_err_jy"]
                    
                    flux_corr = flux_dict["flux_jy"] * psf_corr
                    flux_err_corr = flux_dict["flux_err_jy"] * psf_corr
                    
                    # Quick check of SNR for the user 
                    snr = flux_corr / flux_err_corr if flux_err_corr > 0 else 0
                    
                    print(f"  - {filt}: Flux = {flux_corr*1e6:.3f} µJy | SNR = {snr:.2f} | QC = {qc_flag} ({qc_identifier})")
                    
                    # --- 7. Convert fluxes into AB magnitudes ---
                    if flux_corr > 0:
                        # constant is 8.90 for Jy and 23.90 for µJy
                        ab_mag = -2.5 * np.log10(flux_corr) + 8.90
                    else:
                        ab_mag = np.nan
                    
                    # 8. --- Get background estimates ---
                    n_pix = flux_dict["n_pix"]
                    local_bkg = flux_dict["bkg_flux_jy"]
                    bkg_err = flux_dict["bkg_err_jy"]
                                        
                    # --- 9. Store photometric measurements ---
                    galaxy_row[f"{filt}_flux"] = flux_corr
                    galaxy_row[f"{filt}_flux_err"] = flux_err_corr
                    galaxy_row[f"{filt}_abmag"] = ab_mag
                    galaxy_row[f"{filt}_apflux"] = apflux
                    galaxy_row[f"{filt}_apflux_err"] = apflux_err
                    galaxy_row[f"{filt}_apflux_errnominal"] = apflux_errnominal
                    galaxy_row[f"{filt}_apcorr"] = psf_corr
                    
                    # --- 10. Store background statistics ---
                    galaxy_row[f"{filt}_bkg"] = local_bkg
                    galaxy_row[f"{filt}_bkg_err"] = bkg_err
                    
                    # --- 11. Store quality flag ---
                    galaxy_row[f"{filt}_qc"] = qc_identifier
                    if ap_params["multiplier"] < 2.0:
                        galaxy_row[f"{filt}_ap_flag"] = 1  # Flag for small aperture multiplier
                    
                    # --- 12. Store aperture geometry ---
                    galaxy_row[f"{filt}_ap_theta"] = float(np.degrees(ap_params["theta"].value))
                    galaxy_row[f"{filt}_ap_x"] = float(ap_params["x"])
                    galaxy_row[f"{filt}_ap_y"] = float(ap_params["y"])
                    
                    # --- 13. Store "Scalar" values once ---
                    if not ap_geometry_stored:
                        galaxy_row["MIRI_ap_a"] = ap_params["a"]
                        galaxy_row["MIRI_ap_b"] = ap_params["b"]
                        galaxy_row["MIRI_ap_npix"] = n_pix
                        ap_geometry_stored = True
            
                except Exception as e:
                    print(f"Error processing {target_id} in {filt}: {e}")
            
            # Optionally save curve of growth plots
            if save_cog:
                self._plot_cog(galaxy_cog_dict)    
            
            stored_ids += 1
            
            # Only add the row if we actually measured something
            if len(galaxy_row) > 1:
                all_rows.append(galaxy_row)
                
        # --- 14. Convert to DataFrame and save to file ---
        df = pd.DataFrame(all_rows)
        
        # --- 15. Save the catalogue with proper masking ---
        self._save_catalogue(df)
        
        # --- 16. Plot galaxy filter matrix ---
        if plot_matrix is True:
            # MIRI Coverage
            self.plot_galaxy_filter_matrix()
            
            # MIRI Detections based on SNR threshold
            self.plot_galaxy_filter_matrix(snr_thresh=snr_thresh)
        
        # --- 17. Final message ---
        print("-" * 40)
        print(f"✅ FINAL SUMMARY: {stored_ids} galaxies processed")
        print(f"🛰️ Instrument: JWST/MIRI")
        print(f"📊 Quality Control: CoG Active")
        print("-" * 40)
        print("🎉 Success! Your photometric heart can continue beating peacefully now ❤️‍🩹.")
        
        
    def _save_catalogue(self, df):
        """Save photometric catalogue with explicit Astropy masking."""

        # ---------- CSV ----------
        csv_path = os.path.join(self.csv_dir, f"{self.table_name}.csv")
        df.to_csv(csv_path, index=False)

        # ---------- FITS ----------
        table_path = os.path.join(self.fits_dir, f"{self.table_name}.fits")

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
            table_path,
            format="fits",
            overwrite=True,
            name="MIRI_PHOTOMETRY"
        )

        print(f"\n💾 Saving photometric catalogue to:\n1. {csv_path}\n2. {self.table_path}")


    def plot_galaxy_filter_matrix(self, snr_thresh=None, figname=None, cols=3):
        """
        Visualise which galaxies are observed and detected in each band.

        Parameters:
        -----------
        snr_thresh : float or None
            If set, only observations with SNR above this threshold will be coloured.
             If None, all observations will be coloured regardless of SNR.
        figname : str, optional
            Optionally provide a name for the figure.
        cols : int, optional
            Number of subplot columns. Defaults to 3.
        """
        
        self.detection_dir = os.path.join(self.output_dir, "detection_plots")
        os.makedirs(self.detection_dir, exist_ok=True)
        
        if os.path.exists(self.table_path) is False:
            print("❌ Error: Photometric table not found. Please run photometry first.")
            return
        
        table = Table.read(self.table_path, format="fits")
        
        all_bands = [c.replace('_qc', '') for c in table.colnames if c.endswith('_qc')]
        
        print("Plotting observation heat map for the following filters:\n", all_bands)
        
        pastel_colours = {
            "F770W": "#a6cee3",
            "F1000W": "#b2df8a",
            "F1800W": "#fdbf6f",
            "F2100W": "#fb9a99",
        }

        # Safety check so the code doesn't crash if other filters are added in the future
        for band in all_bands:
            if band not in pastel_colours:
                pastel_colours[band] = "#cccccc"  # Default grey for unknown bands
        
        galaxy_ids = [str(gid) for gid in table["ID"]]
        
        sorted_ids = sorted(galaxy_ids, key=lambda x: int(x))
        num_galaxies = len(galaxy_ids)
        chunk_size = (num_galaxies + 3) // cols
        chunks = [sorted_ids[i : i + chunk_size] for i in range(0, num_galaxies, chunk_size)]

        cell_size = 0.6
        num_filters = len(all_bands)
        num_rows = chunk_size
        fig_width = cell_size * num_filters * cols
        fig_height = cell_size * num_rows * 0.7

        fig, axes = plt.subplots(1, cols, figsize=(fig_width, fig_height), squeeze=False)
        axes = axes[0]

        # Create a mapping for quick row lookup
        table_id_to_idx = {str(row["ID"]): i for i, row in enumerate(table)}

        for ax, g_ids in zip(axes, chunks):
            matrix = np.zeros((len(g_ids), num_filters), dtype=int)
            
            y_labels = []
            for i, gid in enumerate(g_ids):
                
                #Extract galaxy row from the photometric table
                row_idx = table_id_to_idx[gid]
                row = table[row_idx]
                
                # Create y-labels
                label = gid
                y_labels.append(label)
                available_filters = [
                    filt for filt in all_bands 
                    if f"{filt}_flux" in table.colnames and not table[f"{filt}_flux"].mask[row_idx]
                ]
                
                for j, filt in enumerate(all_bands):
                    
                    if filt in available_filters:
                        # Check for Signal-to-noise and quality flag
                        flux_jy = row[f"{filt}_flux"]
                        flux_err_jy = row[f"{filt}_flux_err"]
                        qc_id = row[f"{filt}_qc"]

                        # Determine color (checking for artefacts)
                        base_colour = pastel_colours.get(filt, "#grey")

                        if qc_id > 0:   # Artefact or background-subtraction issue
                            rgb = np.array(mcolors.to_rgb(base_colour))
                            darker_rgb = np.clip(rgb * 0.7, 0, 1)
                            colour = darker_rgb
                        else:
                            colour = base_colour

                        if snr_thresh is not None:
                            snr = flux_jy / flux_err_jy if flux_err_jy > 0 else 0
                            if snr >= snr_thresh:
                                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=colour))
                        else:
                            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=colour))

            ax.set_xlim(0, num_filters)
            ax.set_ylim(len(g_ids), 0)
            ax.set_xticks(np.arange(num_filters) + 0.5)
            ax.set_xticklabels(all_bands, rotation=45, ha="right", fontsize=15)
            ax.set_yticks(np.arange(len(g_ids)) + 0.5)
            ax.set_yticklabels(y_labels, fontsize=15)

            # Add horizontal grid lines
            for y in np.arange(len(g_ids)):
                ax.axhline(y=y, color="grey", linestyle="-", linewidth=0.3, alpha=0.5, zorder=10)

            # Vertical lines at column boundaries
            for x in np.arange(num_filters + 1):
                ax.axvline(x=x, color="grey", linestyle="-", linewidth=0.4, alpha=0.6, zorder=10)
        
        if snr_thresh is not None:
            plt.suptitle(f"MIRI Detections (SNR > {snr_thresh})", fontsize=28, y=0.99)
            if figname is not None:
                figname = os.path.join(self.detection_dir, figname)
            else:
                figname = os.path.join(self.detection_dir, f"miri_obs_snr_{snr_thresh}.png")
        else:
            plt.suptitle("MIRI Observations", fontsize=28, y=0.99)
            if figname is not None:
                figname = os.path.join(self.detection_dir, figname)
            else:
                figname = os.path.join(self.detection_dir, f"miri_obs.png")
            
        plt.tight_layout()
        fig_path = os.path.join(self.detection_dir, figname)
        plt.savefig(fig_path, dpi=150)
        plt.show()
        
        print(f"\n💾 Saved observation matrix to: {fig_path}\n")



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
    def empirical_aperture_rms(img, aperture_params, n_random=200, valid_frac=0.8):
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
        x0, y0 = aperture_params["x"], aperture_params["y"]

        while len(aperturesums) < n_random and attempts < max_attempts:
            attempts += 1

            # random centre inside image (avoid edges)
            x = random.uniform(a + 2, nx - a - 2)
            y = random.uniform(b + 2, ny - b - 2)

            # skip if centre falls on masked pixel
            if np.isnan(img[int(y), int(x)]):
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
            print("         ⚠️ Too few valid random apertures, using propagated errors")
            return None
        
        return median_abs_deviation(aperturesums, scale='normal')
            
