#!/usr/bin/env python
# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "astropy",
#     "matplotlib",
#     "numpy",
#     "scipy",
#     "photutils",
#     "seaborn",
# ]
# ///
"""
MIRI Utils: Astrometric Calibration & Alignment Module
======================================================

A comprehensive toolkit for identifying and correcting systematic astrometric 
offsets between JWST NIRCam (reference) and MIRI (target) imaging.

Key Capabilities:
-----------------
- Automated cross-matching of centroids between multi-wavelength cutouts.
- Hierarchical data management (Survey/Filter structure).
- Interactive quality control via auto-generated flagging sheets.
- Global statistical analysis and visualisation of pointing errors.
- High-fidelity WCS correction of Stage 3 mosaics with spherical geometry 
  correction (RA Cosine) and idempotency checks.

Module Features:
----------------
- compute_offset: Core loop for processing galaxy catalogues.
- generate_flag_sheet: Creates non-destructive CSVs for manual QC.
- display_offsets: Multi-panel visualisation of survey-wide residuals.
- apply_wcs_shift: Spherical-aware WCS correction for FITS mosaics.

Dependencies:
-------------
- astropy (FITS, WCS, Units, Coordinates)
- photutils (Centroiding algorithms)
- pandas (Dataframe management and CSV I/O)
- scipy (Gaussian smoothing)
- matplotlib & seaborn (Diagnostic visualisation)

Author: Benjamin P. Collins
Date: Dec 2025 (Updated for Version 3.1)
Version: 3.1
"""

import os
import glob
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import seaborn as sns
from matplotlib import pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata import Cutout2D
from photutils import centroids

# Internal package imports
from .cutout_tools import load_cutout

# Suppress common WCS-related warnings that don't affect functionality
warnings.simplefilter('ignore', category=FITSFixedWarning)



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
        smoothed_data = ndimage.gaussian_filter(cutout.data, smooth_sigma)
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

    axs[0].imshow(ndimage.gaussian_filter(cutout_nircam.data, 1.0), origin='lower')
    axs[0].plot(*cutout_nircam.wcs.world_to_pixel(centroid_nircam), 'x', color='red')
    axs[0].set(title=f"{g['id']} - NIRCam/F444W")

    axs[1].imshow(ndimage.gaussian_filter(cutout_miri.data, 1.0), origin='lower')
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
    """
    Computes systematic astrometric offsets between NIRCam and MIRI images for a galaxy catalog.

    This function iterates through a catalogue of sources, locates corresponding 
    NIRCam (reference) and MIRI (target) images using path templates, and 
    calculates the RA/Dec shift. It organises results into a hierarchical 
    directory structure: `output_base/survey/filter/`.

    Parameters
    ----------
    cat : astropy.table.Table or pandas.DataFrame
        A catalogue containing source coordinates and IDs. Must include 'id', 
        'ra', and 'dec' columns.
    survey : str
        The name of the survey/programme (e.g., 'cos3d1', 'primer2'). This is 
        used for directory naming and path formatting.
    filter_name : str
        The MIRI filter being processed (e.g. 'F770W', 'F1800W').
    output_base : str or Path
        The root directory where the 'astrometry/' folder structure will be 
        initialised and results saved.
    miri_template : str
        An f-string compatible path template for MIRI files. 
        Example: "data/MIRI/{survey_name}/{survey_name}_{obs}/*{filter}*_i2d.fits"
    nircam_template : str
        An f-string compatible path template for NIRCam reference files.
        Example: "data/NIRCam/mosaics/*{id}*_nircam.fits"
    save_fig : bool, default True
        If True, saves diagnostic plots showing the alignment of centroids 
        for each processed galaxy.
    smooth_miri : bool, default True
        If True, applies a Gaussian kernel to the MIRI image before 
        centroiding to mitigate noise in faint mid-infrared sources.

    Returns
    -------
    None
        Results are saved to disk as:
        1. `{survey}_{filter}_offsets.csv`: Raw calculated shifts.
        2. `{survey}_{filter}_flags.csv`: Quality control sheet for manual review.
        3. `diagnostic_plots/`: Folder containing PNGs for visual verification.

    Notes
    -----
    The function uses a 'non-destructive' approach. If a flagging file 
    already exists, it will not be overwritten, allowing for iterative 
    updates to the offset calculations without losing manual annotations.
    """
    
    # 1. Convert output_base to a Path object for easier handling
    base_path = Path(output_base)
    
    # 2. Extract survey name and observation number
    if survey[-1].isdigit():
        survey_name = survey[:-1]  # e.g. "primer"
        obs = survey[-1]           # e.g. "1"
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
        dra_med = df_clean['dra_arcsec'].median()
        ddec_med = df_clean['ddec_arcsec'].median()

        # Calculate MAD: median(|x - median(x)|)
        # We use 1.4826 to make it comparable to a standard Gaussian sigma
        dra_mad = np.median(np.abs(df_clean['dra_arcsec'] - dra_med)) * 1.4826
        ddec_mad = np.median(np.abs(df_clean['ddec_arcsec'] - ddec_med)) * 1.4826

        return {
            'survey': survey,
            'filter': filter_name,
            'n_sources': len(df_clean),
            
            # RA Statistics
            'dra_mean': df_clean['dra_arcsec'].mean(),
            'dra_std': df_clean['dra_arcsec'].std(),
            'dra_med': dra_med,
            'dra_mad': dra_mad,  # Robust spread
            
            # Dec Statistics
            'ddec_mean': df_clean['ddec_arcsec'].mean(),
            'ddec_std': df_clean['ddec_arcsec'].std(),
            'ddec_med': ddec_med,
            'ddec_mad': ddec_mad,  # Robust spread
            
            # Combined Magnitude (using medians for a more robust total shift)
            'total_shift_med': np.sqrt(dra_med**2 + ddec_med**2)
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


def display_offsets(output_base, summary_df):
    """
    Creates a multi-panel visualization of all astrometric offsets.
    """
    # Define this at the top of your function or as a global config
    FILTER_MARKERS = {
        "F770W": "o",   # Circle
        "F1000W": "^",  # Triangle
        "F1130W": "p",  # Pentagon
        "F1280W": "v",  # Downward Triangle
        "F1500W": "P",  # Thick Plus
        "F1800W": "s",  # Square
        "F2100W": "D",  # Diamond
    }
    
    # Mapping variable names to human-readable names
    PROGRAMME_MAP = {
        "cos3d1": "COSMOS-3D (Obs 1)", "cos3d2": "COSMOS-3D (Obs 2)",
        "cweb1": "COSMOS-Web (Obs 1)",  "cweb2": "COSMOS-Web (Obs 2)",
        "primer1": "PRIMER (Obs 1)",    "primer2": "PRIMER (Obs 2)"
    }

    # Define a paired color palette: Dark for Obs 1, Light for Obs 2
    # 'Paired' provides sets of (Light, Dark) colors
    colors = sns.color_palette("Paired")
    COLOUR_MAP = {
        "COSMOS-3D (Obs 1)": colors[1], "COSMOS-3D (Obs 2)": colors[0],
        "COSMOS-Web (Obs 1)": colors[3], "COSMOS-Web (Obs 2)": colors[2],
        "PRIMER (Obs 1)": colors[5],    "PRIMER (Obs 2)": colors[4]
    }
    
    master_data = []
    
    # 1. Collect all "Good" data points
    for _, row in summary_df.iterrows():
        folder = Path(output_base) / row['survey'] / row['filter']
        off_path = folder / f"{row['survey']}_{row['filter']}_offsets.csv"
        flag_path = folder / f"{row['survey']}_{row['filter']}_flags.csv"
        
        if off_path.exists():
            df = pd.read_csv(off_path)
            if flag_path.exists():
                flags = pd.read_csv(flag_path, comment='#', sep=',')
                df = pd.merge(df, flags, on='galaxy_id')
                df = df[df['use'] == 1] # Only plot the ones you verified
            
            # Clean up labels using the maps
            df['Programme'] = PROGRAMME_MAP.get(row['survey'], row['survey'])
            df['Filter'] = row['filter'].upper()
            master_data.append(df)

    full_df = pd.concat(master_data)

    # 2. Setup Plotting Style
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(14, 10))
    
    # --- Part 1: Global Overplot (Left) ---
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1)
    # Plot individual galaxies with lower alpha
    scatter = sns.scatterplot(
        data=full_df, x='dra_arcsec', y='ddec_arcsec', 
        hue='Programme', style='Filter', markers=FILTER_MARKERS, 
        palette=COLOUR_MAP, alpha=0.4, s=30, ax=ax1
    )
    
    # 3. Plot Medians with human-readable logic
    for prog in full_df['Programme'].unique():
        for filt in full_df[full_df['Programme'] == prog]['Filter'].unique():
            sub = full_df[(full_df['Programme'] == prog) & (full_df['Filter'] == filt)]
            m_style = FILTER_MARKERS.get(filt, 'X')
            m_color = COLOUR_MAP.get(prog, 'black')
            
            ax1.plot(
                sub['dra_arcsec'].mean(), sub['ddec_arcsec'].mean(), 
                marker=m_style, color=m_color, markersize=14, 
                markeredgecolor='white', markeredgewidth=1.5
            )

    # Styling the main plot
    ax1.axhline(0, color='black', linestyle='--', alpha=0.8)
    ax1.axvline(0, color='black', linestyle='--', alpha=0.8)
    ax1.set_title("Global Offset Distribution", fontsize=18, pad=15)
    ax1.set_xlabel(r"$\Delta$RA [arcsec]", fontsize=14)
    ax1.set_ylabel(r"$\Delta$Dec [arcsec]", fontsize=14)
    ax1.set_xlim(-0.3, 0.3)
    ax1.set_ylim(-0.3, 0.3)

    # --- Part 2: Facet-like behavior (Right Top/Bottom) ---
    # Cleanup the legend (Separate colors from shapes)
    handles, labels = ax1.get_legend_handles_labels()
    # Find indices for the headers in the legend to split them
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # --- Part 4: KDE Density Panels (Right) ---
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    sns.kdeplot(data=full_df, x='dra_arcsec', hue='Programme', palette=COLOUR_MAP, ax=ax2, common_norm=False)
    ax2.set_title("RA Offset Stability", fontsize=18, pad=15)

    ax3 = plt.subplot2grid((2, 2), (1, 1))
    sns.kdeplot(data=full_df, x='ddec_arcsec', hue='Programme', palette=COLOUR_MAP, ax=ax3, common_norm=False)
    ax3.set_title("Dec Offset Stability", fontsize=18, pad=15)

    plt.tight_layout()
    plt.savefig(Path(output_base) / "astrometry_diagnostic_plot.png", dpi=300)
    plt.show()


def apply_wcs_shift(input_file, dra_arcsec, ddec_arcsec, output_file=None):
    """
    Applies a systematic astrometric shift to the WCS of a JWST FITS file.

    This function calculates the coordinate shift in degrees, incorporating a 
    cosine declination correction to ensure physical accuracy. It updates the 
    CRVAL1 and CRVAL2 keywords in both the Primary and SCI headers. 
    
    The function is 'idempotent': it checks for the 'ASTRO_COR' keyword and 
    will skip files that have already been corrected to prevent double-shifting.

    Parameters
    ----------
    input_file : str or Path
        Path to the original Stage 3 JWST FITS mosaic (e.g. an i2d.fits file).
    dra_arcsec : float
        The systematic RA offset to subtract, measured in arcseconds on the sky.
    ddec_arcsec : float
        The systematic Dec offset to subtract, measured in arcseconds.
    output_file : str or Path, optional
        If provided, the original file is copied to this path before shifting.
        If None (default), the function performs an 'in-place' update on the 
        input_file. Use with caution.

    Returns
    -------
    None
        The FITS file is updated with:
        1. Shifted CRVAL1/2 coordinates.
        2. A 'HISTORY' entry detailing the specific dRA/dDec applied.
        3. 'ASTRO_COR = True' added to the header as a completion flag.

    Notes
    -----
    The RA correction is calculated as:
    delta_RA_deg = (dra_arcsec / 3600.0) / cos(Declination)
    
    This ensures that the applied coordinate shift accounts for the 
    convergence of longitudinal lines toward the poles.
    """
    target_file = input_file
    if output_file and output_file != input_file:
        if os.path.exists(output_file):
            target_file = output_file
        else:
            shutil.copy2(input_file, output_file)
            target_file = output_file

    # --- THE SAFETY CHECK ---
    with fits.open(target_file) as hdul:
        # Check both Primary and SCI headers for our custom flag
        is_corrected = hdul[0].header.get('ASTRO_COR', False)
        if not is_corrected and 'SCI' in hdul:
            is_corrected = hdul['SCI'].header.get('ASTRO_COR', False)
        
        if is_corrected:
            print(f"⏭️ Skipping {target_file}: Already corrected.")
            return

    with fits.open(target_file, mode='update') as hdul:
        # 1. Identify the WCS header (prefer SCI extension)
        ext = 'SCI' if 'SCI' in hdul else 0
        header = hdul[ext].header
        
        wcs = WCS(header)
        if not wcs.has_celestial:
            print(f"⚠️ No celestial WCS in {target_file}")
            return

        # 2. RA Cosine Correction
        # dra_deg = dra_arcsec / (3600 * cos(dec))
        # Note: This is only relevant for fields close to the poles where longitudinal lines appear squished
        dec_ref = header.get('CRVAL2', 0)
        cos_dec = np.cos(np.deg2rad(dec_ref))
        
        dra_deg = (dra_arcsec / 3600.0) / cos_dec
        ddec_deg = ddec_arcsec / 3600.0

        # 3. Apply shift to BOTH Primary and SCI headers if they exist
        # This ensures tools like DS9 and Python scripts stay in sync
        for h in [hdul[0].header, hdul[ext].header]:
            if 'CRVAL1' in h:
                h['CRVAL1'] -= dra_deg
                h['CRVAL2'] -= ddec_deg
                h['HISTORY'] = f"Astrometry corrected: dRA={dra_arcsec:.4f}\", dDec={ddec_arcsec:.4f}\""
                h['ASTRO_COR'] = True

        hdul.flush() 
    
    print(f"✅ {'Updated' if not output_file else 'Created'} {target_file}")