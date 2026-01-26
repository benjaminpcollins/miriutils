#!/usr/bin/env python
# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "astropy",
#     "matplotlib",
#     "numpy",
#     "reproject",
#     "scipy",
# ]
# ///
"""
MIRI Utils: Visualisation & RGB Composition Module
=================================================

This module provides the RGBComposer class, designed to automate the creation 
of 3-color composite stamps from multi-instrument JWST data (MIRI & NIRCam). 
It handles spatial alignment via reprojection, North-up rotation, asinh scaling, 
and automated labeling with filter legends and scale bars.

Classes
-------
    - RGBComposer: The main engine for finding, aligning, and rendering RGB stamps.

Key Capabilities
----------------
    - Automatic filter-to-channel mapping (Recipe generation).
    - Sub-pixel alignment using Astropy Reproject.
    - Publication-ready plotting with color-coded legends.

Example usage
-------------
    from miri_utils.vis import RGBComposer

    # Initialise the composer
    composer = RGBComposer(miri_dir="./miri_fits", 
                           nircam_dir="./nircam_fits", 
                           output_dir="./stamps")

    # Process a specific galaxy
    files = composer.find_files_for_galaxy("18332")
    recipe = composer.determine_recipe(files)
    
    if recipe:
        arrays, target_wcs = composer.process_and_align(recipe, rotate_north=True)
        rgb_data = composer.create_rgb(arrays, stretch=0.5)
        composer.save_stamp("18332", rgb_data, recipe)

Author: Benjamin P. Collins
Date: January 10, 2026
Version: 3.1
"""

import os
import re
import glob
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from reproject import reproject_interp
from astropy.wcs import WCS
from astropy.visualization import (AsinhStretch, LinearStretch, 
                                   ManualInterval, PercentileInterval, 
                                   make_rgb)

class RGBComposer:
    def __init__(self, cutout_dir, nircam_dir, output_dir):
        self.cutout_dir = cutout_dir
        self.nircam_dir = nircam_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define our filter "anchor"
        self.nircam_anchor = 'F444W'
        self.miri_pattern = re.compile(r'F\d{3,4}W')

    def get_filter_wavelength(self, filter_name):
        """Extracts numeric wavelength, e.g., 'F2100W' -> 2100"""
        match = re.search(r'\d+', filter_name)
        return int(match.group()) if match else 0

    def find_files(self, gid):
        """
        Scans directories to see what FITS files actually exist for this ID.
        Returns a dictionary: {filter_name: path_to_file}
        """
        available = {}
        
        # 1. Check NIRCam dir for F444W
        nircam_match = glob.glob(os.path.join(self.nircam_dir, f"{gid}_{self.nircam_anchor}*.fits"))
        if nircam_match:
            available[self.nircam_anchor] = nircam_match[0]
            
        # 2. Search MIRI directory recursively
        # We look for the ID and the specific f[digits]w pattern
        miri_pattern = os.path.join(self.cutout_dir, "**", f"fits/{gid}_f*w*.fits")
        miri_matches = glob.glob(miri_pattern, recursive=True)
        
        for path in miri_matches:
            fname = os.path.basename(path)
            # Find the filter part (e.g., F2100W)
            match = self.miri_pattern.search(fname.upper())
            if match:
                filter_name = match.group()
                available[filter_name] = path
            
        return available

    def get_recipe(self, available_files):
        """
        Applies your logic to map available files to R, G, B channels.
        """
        # Separate MIRI filters and sort by wavelength
        miri_filters = sorted(
            [f for f in available_files.keys() if f != self.nircam_anchor],
            key=self.get_filter_wavelength
        )                
                
        count = len(miri_filters)
        
        # Helper to create a channel entry
        def get_chan(f_name, files_dict):
            if f_name is None or f_name not in files_dict:
                return {'path': None, 'name': 'N/A'}
            return {'path': files_dict[f_name], 'name': f_name}

        recipe = {
            'R': {'path': None, 'name': 'N/A'},
            'G': {'path': None, 'name': 'N/A'},
            'B': {'path': None, 'name': 'N/A'}
        }

        if count == 0:
            return None 
            
        if count == 1:
            recipe['R'] = get_chan(miri_filters[0], available_files)
            recipe['B'] = get_chan(self.nircam_anchor, available_files)
            
        elif count == 2:
            # Longest MIRI -> Red, Shortest MIRI -> Green, NIRCam -> Blue
            recipe['R'] = get_chan(miri_filters[1], available_files)
            recipe['G'] = get_chan(miri_filters[0], available_files)
            recipe['B'] = get_chan(self.nircam_anchor, available_files)
            
        elif count == 3:
            recipe['R'] = get_chan(miri_filters[2], available_files)
            recipe['G'] = get_chan(miri_filters[1], available_files)
            recipe['B'] = get_chan(miri_filters[0], available_files)
        
        else:
            # Pick the spread for 4+ filters
            recipe['R'] = get_chan(miri_filters[-1], available_files)
            recipe['G'] = get_chan(miri_filters[len(miri_filters)//2], available_files)
            recipe['B'] = get_chan(miri_filters[0], available_files)
            
        return recipe
    
    def process_and_align(self, recipe, rotate_north=True, crop_size_arcsec=3.0):
        """
        Aligns all channels in the recipe to a common grid and crops to a specific size.
        """
        # Pick a reference file (Red if available, else Blue)
        ref_path = recipe['R']['path']
        if not ref_path:
            raise ValueError("Recipe must have at least one valid FITS path.")
        
        # Pull the pre-centered sky coordinates from the Reference File
        with fits.open(ref_path) as hdul:
            # load WCS from SCI extension of a reference MIRI cutout
            ref_wcs = WCS(hdul['SCI'].header, naxis=2)
            
            # This is the RA/Dec of the galaxy (anchored by the produce_cutouts function)
            sky_center = ref_wcs.wcs.crval
            
            # Define pixel scale of the final images
            pix_scale_deg = np.sqrt(np.abs(np.linalg.det(ref_wcs.pixel_scale_matrix)))
            pix_scale_arcsec = pix_scale_deg * 3600.0
            
        # Define the size of the final RGB image
        crop_pix = int(crop_size_arcsec / pix_scale_arcsec)
        
        # Define the centre for CRPIX (0-based centre of the new box)
        # Attention: FITS images start with CRPIX=1 at the first pixel
        center_f = (crop_pix - 1) / 2.0 

        # Build Target WCS from scratch and provide the necessary information
        target_wcs = WCS(naxis=2)
        target_wcs.wcs.crval = sky_center      # Pin galaxy RA/Dec to...
        target_wcs.wcs.crpix = [center_f + 1, center_f + 1] # ...the center of the box (accounting for 1-based)
        target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        # Standard North-up: CDELT1 is negative (RA increases to the left)
        target_wcs.wcs.cdelt = [-pix_scale_deg, pix_scale_deg]
        
        # Manually make the image point North-up
        # If not rotating, copy the original PC matrix to keep the same angle
        target_wcs.wcs.pc = [[1, 0], [0, 1]] if rotate_north else ref_wcs.wcs.pc

        processed_arrays = {}

        # Reproject directly into the small 3x3" box
        for chan in ['R', 'G', 'B']:
            chan_info = recipe[chan]
            if chan_info['path'] is None:
                processed_arrays[chan] = np.zeros((crop_pix, crop_pix))
                continue
            
            with fits.open(chan_info['path']) as h:
                # Passing h['SCI'] gives reproject the data + its WCS
                raw_data = h['SCI'].data
                
                # 1. Calculate the robust background
                # sigma=3.0 and maxiters=5 is standard for JWST
                _, median_bkg, std_bkg = sigma_clipped_stats(raw_data, sigma=3.0, maxiters=5)
                
                # 2. Fill NaNs with the median background
                # We use a copy to avoid modifying the original file data in memory
                clean_data = np.where(np.isnan(raw_data), median_bkg, raw_data)
                
                # 3. Inject back into a temporary HDU for reprojection
                # reproject_interp needs an object with a .header and .data
                temp_hdu = fits.ImageHDU(data=clean_data, header=h['SCI'].header)
                
                data, _ = reproject_interp(
                    temp_hdu, 
                    target_wcs, 
                    shape_out=(crop_pix, crop_pix)
                )
                processed_arrays[chan] = data
        
        return processed_arrays, target_wcs
    
    def create_rgb(self, processed_arrays, stretch=0.5):
        """
        Uses Astropy's visualisation tools to scale and combine arrays.
        """
        # 1. Select the channels
        r = processed_arrays['R']
        g = processed_arrays['G']
        b = processed_arrays['B']

        # 2. Define the interval (Scaling/Clipping)
        # PercentileInterval(99.5) automatically finds the saturation point
        # Like Trilogy's 'sat' parameter
        interval = PercentileInterval(99.5)
        
        # 3. Define the stretch (The Math)
        # Asinh is the standard for JWST to see faint arms and bright cores
        # 'a' is the stretch parameter (non-linearity)
        stretch_func = AsinhStretch(a=stretch)

        # 4. Use Astropy's make_rgb to combine and scale to 8-bit
        # This handles the normalization and stacking in one go
        rgb_image = make_rgb(r, g, b, 
                            interval=interval, 
                            stretch=stretch_func)
        
        return rgb_image

    
    def save_stamp(self, galaxy_id, rgb_array, recipe, output_name=None):
        """
        Plots the RGB image and adds a legend based on the filters used.
        """
        fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
        
        # Display the image
        ax.imshow(rgb_array, origin='lower')
        
        # Create custom legend handles
        # We use the keys from your recipe to pull the filter names
        #legend_text = f"R: {recipe['R']['name']}\nG: {recipe['G']['name']}\nB: {recipe['B']['name']}"
        legend_elements = [
            mpatches.Patch(color='red', label=f"R: {recipe['R']['name']}"),
            mpatches.Patch(color='lime', label=f"G: {recipe['G']['name']}"),
            mpatches.Patch(color='blue', label=f"B: {recipe['B']['name']}")
        ]
        
        # Add the legend
        # We use a semi-transparent black background.
        leg = ax.legend(handles=legend_elements, 
                loc='upper left', 
                fontsize=12, 
                frameon=True, 
                facecolor='black', 
                edgecolor='none', 
                labelcolor='white',
                handlelength=0.7, # Makes the patches square
                borderpad=1.2,
                handletextpad=0.5)
        
        # 5. Clean up and Save
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Use tight_layout instead of manual subplots_adjust to prevent clipping
        fig.tight_layout(pad=0)
        
        # Calculate how many pixels wide 1 arcsecond is
        ref_path = recipe['R']['path']
        if not ref_path:
            raise ValueError("Recipe must have at least one valid FITS path.")
        
        # 2. Pull the pre-centered sky coordinates from the Reference File
        with fits.open(ref_path) as hdul:
            ref_wcs = WCS(hdul['SCI'].header, naxis=2)
            pix_scale_deg = np.sqrt(np.abs(np.linalg.det(ref_wcs.pixel_scale_matrix)))
            pix_scale_arcsec = pix_scale_deg * 3600.0
        
        bar_length_arcsec = 0.5  # Changed to half-arcsec
        final_bar_length = (bar_length_arcsec / pix_scale_arcsec) / rgb_array.shape[1]

        # Draw a simple white line for 1 arcsecond
        ax.plot([0.8, 0.8 + final_bar_length], [0.05, 0.05], 
                transform=ax.transAxes, color='white', lw=2)
        ax.text(0.8, 0.07, '0.5"', transform=ax.transAxes, color='white', ha='center')
        
        # Save the result
        output_path = os.path.join(self.output_dir, output_name if output_name else f"{galaxy_id}_rgb.png")        
        
        # IMPORTANT: bbox_extra_artists ensures the legend is included in the save
        plt.savefig(output_path, 
                    bbox_inches='tight', 
                    pad_inches=0, 
                    dpi=150, 
                    facecolor='black',
                    bbox_extra_artists=(leg,))        
        
        plt.close()
        
        print(f"Saved labeled stamp to {output_path}")
