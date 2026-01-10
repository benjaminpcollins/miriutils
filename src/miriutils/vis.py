#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIRI Utils: Colour Image Processing Module
=========================================

This module provides utilities for creating colour-composite images from FITS data
obtained from JWST MIRI and NIRCam instruments. It includes functions for resampling,
normalising, and combining image data to create scientifically informative and
visually appealing colour images.

Functions
---------
    - resample_nircam: Resamples NIRCam images to a specified pixel size
    - normalise_image: Applies various stretches to normalise image data
    - preprocess_fits_image: Loads and preprocesses FITS images with background handling
    - make_stamp: Creates RGB composite images from multiple FITS files

Example usage
-------------
    from miri_utils.color_utils import resample_nircam, make_stamp

    # Resample all NIRCam files in directory
    resample_nircam("./data", 1024)

    # Create color composite
    image_dict = {
        'R': ['image_F1800W.fits[0]'],
        'G': ['image_F770W.fits[0]'],
        'B': ['image_F444W.fits[0]']
    }
    make_stamp(image_dict, 10, 0.05, 1.0, 10, 0.05, 1.0, 10, 0.05, 1.0, 
            stretch='asinh', outfile='my_color_image.pdf')

Author: Benjamin P. Collins
Date: May 15, 2025
Version: 1.0
"""

import os
import re
import glob
import numpy as np
import tempfile
from astropy.io import fits
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from collections import defaultdict
from reproject import reproject_interp
from PIL import Image
from astropy.wcs import WCS
from astropy.visualization import (AsinhStretch, LinearStretch, 
                                   ManualInterval, PercentileInterval, 
                                   make_rgb)

class RGBComposer:
    def __init__(self, cutout_dir, nircam_dir, output_dir):
        self.cutout_dir = cutout_dir
        self.nircam_dir = nircam_dir
        self.output_dir = output_dir
        
        # Define our filter "anchor"
        self.nircam_anchor = 'F444W'
        self.miri_pattern = re.compile(r'F\d{3,4}W')

    def get_filter_wavelength(self, filter_name):
        """Extracts numeric wavelength, e.g., 'F2100W' -> 2100"""
        match = re.search(r'\d+', filter_name)
        return int(match.group()) if match else 0

    def find_files_for_galaxy(self, gid):
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

    def determine_recipe(self, available_files):
        """
        Applies your logic to map available files to R, G, B channels.
        """
        # Separate MIRI filters and sort by wavelength
        miri_filters = sorted(
            [f for f in available_files.keys() if f != self.nircam_anchor],
            key=self.get_filter_wavelength
        )                
                
        nircam_path = available_files.get(self.nircam_anchor)
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
        # 1. Pick a reference file (Red if available, else Blue)
        ref_path = recipe['B']['path']
        if not ref_path:
            raise ValueError("Recipe must have at least one valid FITS path.")
        
        # 2. Pull the pre-centered sky coordinates from the Reference File
        with fits.open(ref_path) as hdul:
            ref_wcs = WCS(hdul['SCI'].header, naxis=2)
            # This is the RA/Dec of the galaxy you pinned earlier
            sky_center = ref_wcs.wcs.crval
            
            pix_scale_deg = np.sqrt(np.abs(np.linalg.det(ref_wcs.pixel_scale_matrix)))
            pix_scale_arcsec = pix_scale_deg * 3600.0
            
        # 3. Define the 3x3" Output Grid
        crop_pix = int(crop_size_arcsec / pix_scale_arcsec)
        # Precise center for CRPIX (0-based center of the new box)
        center_f = (crop_pix - 1) / 2.0 

        # 4. Build Target WCS
        target_wcs = WCS(naxis=2)
        target_wcs.wcs.crval = sky_center      # Pin galaxy RA/Dec to...
        target_wcs.wcs.crpix = [center_f + 1, center_f + 1] # ...the center of the box
        target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        if rotate_north:
            # Standard North-up: CDELT1 is negative (RA increases to the left)
            target_wcs.wcs.cdelt = [-pix_scale_deg, pix_scale_deg]
            target_wcs.wcs.pc = [[1, 0], [0, 1]]
        else:
            # If not rotating, copy the original PC matrix to keep the same angle
            target_wcs.wcs.cdelt = [-pix_scale_deg, pix_scale_deg]
            target_wcs.wcs.pc = ref_wcs.wcs.pc

        processed_arrays = {}

        # 4. Reproject directly into the small 3x3" box
        for chan in ['R', 'G', 'B']:
            chan_info = recipe[chan]
            if chan_info['path'] is None:
                processed_arrays[chan] = np.zeros((crop_pix, crop_pix))
                continue
            
            with fits.open(chan_info['path']) as h:
                # Passing h['SCI'] gives reproject the data + its WCS
                data, footprint = reproject_interp(
                    h['SCI'], 
                    target_wcs, 
                    shape_out=(crop_pix, crop_pix)
                )
                processed_arrays[chan] = np.nan_to_num(data)
        
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
        
        # Save the result
        output_path = os.path.join(self.output_dir, f"{galaxy_id}_rgb.png")
        
        # IMPORTANT: bbox_extra_artists ensures the legend is included in the save
        plt.savefig(output_path, 
                    bbox_inches='tight', 
                    pad_inches=0, 
                    dpi=150, 
                    facecolor='black',
                    bbox_extra_artists=(leg,))        
        
        plt.close()
        
        print(f"Saved labeled stamp to {output_path}")
