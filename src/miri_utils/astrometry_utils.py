#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIRI Utils Astrometric Offset Module
====================================

This module provides functions for computing and visualising astrometric offsets between NIRCam and MIRI cutouts, 
which is critical for ensuring accurate positional alignment in multi-wavelength analyses. 
The module includes tools for:
- Computing centroids for cutouts with optional smoothing to reduce noise.
- Saving alignment figures to visually inspect centroid matching.
- Calculating RA/Dec offsets for entire galaxy catalogues.
- Exporting statistical summaries of offset distributions.
- Shifting MIRI FITS files to correct systematic positional offsets.

Dependencies
------------
Dependencies:
- astropy (for FITS I/O, WCS transformations, and coordinate calculations)
- matplotlib (for visualisation)
- photutils (for centroiding)
- scipy (for image smoothing)
- numpy (for array manipulations)
- json (for exporting statistics)
 
Requirements
------------
- astropy
- numpy
- scipy
- pandas

Usage
-----
- Ensure cat is defined with the expected structure (including 'id', 'ra', 'dec' columns).
- Prepare offset columns by calling the prepare_cols function.
- Call the compute_offset function to compute centroids and offsets for each galaxy in the catalogue.
- Use save_alignment_figure for visual verification of alignment.
- Export summary statistics with write_offset_stats.
- Shift MIRI cutouts to correct for systematic offsets using shift_miri_fits.

Author: Benjamin P. Collins
Date: May 15, 2025
Version: 3.0
"""

import numpy as np
import scipy
import os
import glob
import warnings
import pandas as pd
from pathlib import Path

from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, FITSFixedWarning
import astropy.units as u
from astropy.nddata import Cutout2D
from photutils import centroids
from .cutout_tools import load_cutout
import shutil

# Suppress common WCS-related warnings that don't affect functionality
warnings.simplefilter("ignore", category=FITSFixedWarning)

def get_path(template, **kwargs):
    """
    Formats a template and searches for a matching file on disk.
    Supports wildcards (*).
    """
    try:
        # 1. Fill in the variables (id, survey_name, filter, etc.)
        target_pattern = template.format(**kwargs)
        
        # 2. Use glob to find files matching the pattern
        # This handles the messy MAST names automatically
        matches = glob.glob(target_pattern)
        
        if matches:
            # Return the first match (usually there's only one i2d per filter)
            return matches[0]
        else:
            return None
    except KeyError as e:
        print(f"Missing variable for template: {e}")
        return None

def compute_centroid(cutout, smooth_sigma, good_frac_cutout, smooth_miri):
    """Compute the centroid of a given cutout image using quadratic fitting.
    
    Parameters:
    cutout (Cutout2D): The 2D image cutout for centroid computation.
    smooth_sigma (float): Sigma for Gaussian smoothing.
    good_frac_cutout (float): Fraction of the cutout used for centroid fitting.
    smooth_miri (bool): Whether to apply additional smoothing to MIRI data.
    
    Returns:
    SkyCoord: The computed centroid in world coordinates, or None if the centroid could not be determined.
    """
    if cutout is None:
        return None

    # Decide whether to smooth MIRI or not
    if smooth_miri == True:
        smoothed_data = scipy.ndimage.gaussian_filter(cutout.data, smooth_sigma)
    else: 
        smoothed_data = cutout.data
    
    # Makes sure the boxsize is an odd number
    search_boxsize = int(np.floor(good_frac_cutout * cutout.shape[0]) // 2 * 2 + 1)

    centroid_pix = centroids.centroid_quadratic(
        smoothed_data,
        xpeak=cutout.shape[0] // 2,
        ypeak=cutout.shape[1] // 2,
        search_boxsize=search_boxsize,
        fit_boxsize=5
    )

    return cutout.wcs.pixel_to_world(centroid_pix[0], centroid_pix[1]) if not np.isnan(centroid_pix).any() else None

def save_alignment_figure(g, cutout_nircam, cutout_miri, centroid_nircam, centroid_miri, output_dir, filter):
    """Save a side-by-side comparison of NIRCam and MIRI cutouts, with centroids marked.
    
    Parameters:
    g (dict): Galaxy metadata including ID.
    cutout_nircam (Cutout2D): NIRCam cutout image.
    cutout_miri (Cutout2D): MIRI cutout image.
    centroid_nircam (SkyCoord): Centroid of the NIRCam cutout.
    centroid_miri (SkyCoord): Centroid of the MIRI cutout.
    output_dir (str): Directory to save the figure.
    survey (str): Survey name.
    filter (str): MIRI filter used.
    """
    
    fig, axs = plt.subplots(1, 2, figsize=[10, 5])

    axs[0].imshow(scipy.ndimage.gaussian_filter(cutout_nircam.data, 1.0), origin='lower')
    axs[0].plot(*cutout_nircam.wcs.world_to_pixel(centroid_nircam), 'x', color='red')
    axs[0].set(title=f"{g['id']} - NIRCam/F444W")

    axs[1].imshow(scipy.ndimage.gaussian_filter(cutout_miri.data, 1.0), origin='lower')
    axs[1].plot(*cutout_miri.wcs.world_to_pixel(centroid_miri), 'o', color='orange')
    axs[1].set(title=f"{g['id']} - MIRI/{filter}")
    
    # show expected position of the centroid
    expected_position_pix = cutout_miri.wcs.world_to_pixel(centroid_nircam)
    axs[1].plot(expected_position_pix[0], expected_position_pix[1], 'x', color='red')

    output_path = os.path.join(output_dir, f"{g['id']}.png")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path)
    plt.close()
    
    

def compute_offset(cat, survey, filter_name, output_base, miri_template, nircam_template, save_fig=True, smooth_miri=True):
    """Computes the astrometric offset between NIRCam and MIRI for each galaxy."""
    
    # 1. Convert output_base to a Path object for easier handling
    base_path = Path(output_base)
    
    # 2. Extract survey name and observation number
    if survey[-1].isdigit():
        survey_name = survey[:-1]  # e.g., "primer"
        obs = survey[-1]           # e.g., "1"
    else:
        survey_name = survey
        obs = ''
    
    filter_l = filter_name.lower()

    # 3. Define the new Hierarchical Structure: base/survey/filter/
    # This creates a specific folder for the survey AND the filter
    work_dir = base_path / survey / filter_name
    plot_dir = work_dir / "diagnostic_plots"
    
    # Create all directories at once (parents=True creates survey and filter folders)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 4. Define the CSV paths for later use
    csv_path = work_dir / f"{survey}_{filter_name}_offsets.csv"
    
    print(f"Directory structure initialized at: {work_dir}")
    
    # 1. Initialise an empty list to store the offsets for each survey
    offset_results = []
    
    # Loop through all galaxies
    for i, gal in enumerate(cat):
        
        # 1. Generate paths using the templates
        m_path = get_path(miri_template, id=gal['id'], survey_name=survey_name, obs=obs, filter=filter_l)
        n_path = get_path(nircam_template, id=gal['id'])
        
        # 2. Safety check: Ensure both paths were successfully found AND exist on disk
        if m_path is None or n_path is None:
            # Optional: print(f"File missing for ID {gal['id']}")
            continue

        if not (os.path.exists(m_path) and os.path.exists(n_path)):
            continue    
        
        ref_position = SkyCoord(ra=gal['ra'], dec=gal['dec'], unit=u.deg)
        cutout_size = (2.5 * u.arcsec, 2.5 * u.arcsec)
        smooth_sigma, good_frac_cutout = 1.0, 0.7
        
        # Load MIRI cutout
        miri_data, miri_wcs = load_cutout(m_path)
        if miri_data is None:
            continue
        cutout_miri = Cutout2D(miri_data, ref_position, cutout_size, wcs=miri_wcs)

        # Load NIRCam cutout
        nircam_data, nircam_wcs = load_cutout(n_path)
        if nircam_data is None:
            continue
        cutout_nircam = Cutout2D(nircam_data, ref_position, cutout_size, wcs=nircam_wcs)

        # Compute centroids
        centroid_nircam = compute_centroid(cutout_nircam, smooth_sigma, good_frac_cutout, smooth_miri)
        centroid_miri = compute_centroid(cutout_miri, smooth_sigma, good_frac_cutout, smooth_miri)

        if centroid_nircam is None or centroid_miri is None:
            print(f"{gal['id']}: Centroid not found - Skipping...")
            continue
        
        if save_fig == True:
            # Save alignment figure
            save_alignment_figure(gal, cutout_nircam, cutout_miri, centroid_nircam, centroid_miri, plot_dir, filter_name)
        
        # Compute offsets
        dra, ddec = centroid_nircam.spherical_offsets_to(centroid_miri)
        
        # 2. Append a dictionary for each galaxy
        offset_results.append({
            'galaxy_id': gal['id'],
            'dra_arcsec': dra.to(u.arcsec).value,
            'ddec_arcsec': ddec.to(u.arcsec).value
        })
    
    if offset_results:
        # 3. Create a fresh DataFrame and save to CSV
        df_offsets = pd.DataFrame(offset_results)
        df_offsets.to_csv(csv_path, index=False)
            
        # 4. Create empty survey flag sheet if it doesn't exist
        generate_flag_sheet(work_dir, survey, filter_name)
    else:
        print("No offsets were computed; no output CSV generated.")
    

def generate_flag_sheet(work_dir, survey, filter_name):
    """Creates a blank flagging CSV if one doesn't exist."""
    
    flag_csv = work_dir / f"{survey}_{filter_name}_flags.csv"
    offset_csv = work_dir / f"{survey}_{filter_name}_offsets.csv"
    
    # 1. Define the header instructions
    instructions = [
        "# ASTROMETRY QUALITY CONTROL SHEET",
        f"# Survey: {survey} | Filter: {filter_name}",
        "# --------------------------------------------------",
        "# INSTRUCTIONS:",
        "# 1. Review the diagnostic plots in the 'diagnostic_plots' folder.",
        "# 2. In the 'use' column: Set to 1 for GOOD, 0 for BAD (outliers/contamination).",
        "# 3. Use the 'notes' column to document why a source was excluded.",
        "# 4. SAVE THIS FILE AS A CSV. Do not change column names.",
        "# --------------------------------------------------\n\n"
    ]
    
    if os.path.exists(offset_csv) and not os.path.exists(flag_csv):
        df = pd.read_csv(offset_csv)
        # Create a new dataframe with just ID and a 'use' column (1 = Good, 0 = Bad)
        flag_df = df[['galaxy_id']].copy()
        
        flag_df['use'] = 1  # Default everything to 'good'
        flag_df['notes'] = "" # Space for your manual comments
        
        flag_df.to_csv(flag_csv, index=False)
        print(f"Flag sheet created: {flag_csv}")
        
        # Write instructions first, then the data
        with open(flag_csv, 'w') as f:
            f.write("\n".join(instructions))
            flag_df.to_csv(f, index=False)
        print(f"Flag sheet created with instructions: {flag_csv}")



def get_survey_stats(output_base, survey, filter_name):
    """Calculates clean statistics for a single survey/filter combination."""
    # This automatically handles the subfolder logic
    # Path: output_base / survey / filter_name
    base = Path(output_base)
    folder = base / survey / filter_name
    offset_path = folder / f"{survey}_{filter_name}_offsets.csv"
    flag_path = folder / f"{survey}_{filter_name}_flags.csv"

    if not Path(offset_path).exists():
        return None
    
    try:
        # Load offsets and flags
        df = pd.read_csv(offset_path)
        
        if flag_path.exists() and flag_path.stat().st_size > 0:
            # We use quoting=3 to ignore stray quotes that cause ParserError
            flags = pd.read_csv(flag_path, 
                                comment='#', 
                                quotechar='"',
                                skipinitialspace=True)
            df = pd.merge(df, flags[['galaxy_id', 'use']], on='galaxy_id')
            df_clean = df[df['use'] == 1]
        else:
            df_clean = df
        
        if df_clean.empty: return None
        
        # Compute Statistics
        return {
            'survey': survey,
            'filter': filter_name,
            'n_sources': len(df_clean),
            'dra_mean': df_clean['dra_arcsec'].mean(),
            'dra_std': df_clean['dra_arcsec'].std(),
            'ddec_mean': df_clean['ddec_arcsec'].mean(),
            'ddec_std': df_clean['ddec_arcsec'].std(),
            'total_mag': np.sqrt(df_clean['dra_arcsec'].mean()**2 + df_clean['ddec_arcsec'].mean()**2)
        }
            
    except Exception as e:
        print(f"Skipping {survey}_{filter_name} due to error: {e}")
        return None   

    

def generate_master_summary(output_base, survey_config):
    """
    survey_config: dict of {survey: [filters]}
    Example: {"primer1": ["F770W", "F1800W"]}
    """
    all_stats = []
    
    for survey, filters in survey_config.items():
        for filt in filters:
                
            res = get_survey_stats(output_base, survey, filt)
            if res:
                all_stats.append(res)

    # Create the comparison table
    summary_df = pd.DataFrame(all_stats)
    
    # Save it for your paper/thesis
    summary_df.to_csv(Path(output_base) / "astrometry_summary.csv", index=False)
    return summary_df

def apply_wcs_shift(input_file, dra_arcsec, ddec_arcsec, output_file=None):
    """
    Applies an astrometric shift to a FITS file.
    
    Args:
        input_file (str): Path to original FITS.
        dra_arcsec (float): RA shift (arcsec).
        ddec_arcsec (float): Dec shift (arcsec).
        output_file (str, optional): If provided, saves a copy here. 
                                     If None, updates input_file in-place.
    """
    # 1. Handle File Copying if necessary
    target_file = input_file
    if output_file and output_file != input_file:
        shutil.copy2(input_file, output_file)
        target_file = output_file

    # 2. Open and Update
    with fits.open(target_file, mode='update') as hdul:
        # Most JWST data has WCS in the 'SCI' extension
        header = hdul[0].header
        if 'CRVAL1' not in header and len(hdul) > 1:
            header = hdul['SCI'].header

        wcs = WCS(header)
        if not wcs.has_celestial:
            print(f"⚠️ No celestial WCS in {target_file}")
            return

        # Apply shift (degrees)
        header['CRVAL1'] -= dra_arcsec / 3600.0
        header['CRVAL2'] -= ddec_arcsec / 3600.0

        # 3. Add Metadata (Crucial for Package Users!)
        header['HISTORY'] = f"Astrometry corrected: dRA={dra_arcsec:.4f}\", dDec={ddec_arcsec:.4f}\""
        header['ASTRO_CORR'] = True # Custom keyword for easy filtering
        
        hdul.flush() 
    
    print(f"✅ {'Updated' if not output_file else 'Created'} {target_file}")