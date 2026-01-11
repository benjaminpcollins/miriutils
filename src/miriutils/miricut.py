#!/usr/bin/env python
# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "astropy",
#     "matplotlib",
#     "numpy",
# ]
# ///
"""
MIRI Utils: Astronomical Image Cutout Generator
==============================================

A professional, class-based utility for extracting multi-extension FITS cutouts 
from JWST mosaics. Optimised for astronomical research workflows.

Core Functionalities:
---------------------
* Class-based 'CutoutManager' for stateful survey processing.
* Dual-stage quality control: Global NaN ratio and Central Circular Mask (2" radius).
* Intelligent Directory I/O: Automatic folder nesting and selective cleanup.
* Visualisation: Orientation-aware PNG previews with N-E and X-Y compasses.
* Flexible Overwrite Logic: User-defined control over existing file handling.

Author: Benjamin P. Collins
Date: Jan 2026
Version: 4.0
"""

import os
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

# Suppress annoying WCS warnings from JWST headers that don't impact accuracy
warnings.simplefilter("ignore", category=FITSFixedWarning)

class CutoutManager:
    """A tool for generating multi-instrument astronomical cutouts."""
    
    def __init__(self, base_dir, survey_dict, catalogue, instrument="MIRI"):
        self.base_dir = base_dir
        self.instrument = instrument
        self.survey_dict = survey_dict
        
        with fits.open(catalogue) as cat_hdul:
            cat_data = cat_hdul[1].data
            self.ids = cat_data['id']
            self.ras = cat_data['ra']
            self.decs = cat_data['dec']
        
        # Easy to expand for NIRCam or HST later
        self.scales = {
            "MIRI": 0.11092,
            "NIRCAM_LW": 0.063,
            "NIRCAM_SW": 0.031
        }
        self.pixel_scale = self.scales.get(instrument, 0.1)

    def _get_paths(self, survey, filter_name):
        """Standardizes your folder structure across all projects."""
        filter_dir = os.path.join(self.base_dir, survey, filter_name.upper())
        paths = {
            'fits': os.path.join(filter_dir, 'fits'),
            'png': os.path.join(filter_dir, 'png')
        }
        for p in paths.values():
            os.makedirs(p, exist_ok=True)
        return paths

    def check_quality(self, data, global_thresh=0.4, inner_thresh=0.05, radius_arcsec=2.0):
        """
        Performs quality check using a global threshold and a circular central mask.
        """
        # 1. Global Check (Total NaNs in the whole 8x8" frame)
        if np.isnan(data).sum() / data.size > global_thresh:
            return False
        
        # 2. Circular Centre Check
        ny, nx = data.shape
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        
        # Convert radius from arcsec to pixels
        r_pix = radius_arcsec / self.pixel_scale
        
        # Create a coordinate grid
        y, x = np.ogrid[:ny, :nx]
        
        # Calculate squared distance from center (more efficient than sqrt)
        dist_sq = (x - cx)**2 + (y - cy)**2
        
        # Create the circular mask
        mask = dist_sq <= r_pix**2
        
        # Extract data within the circle
        central_pixels = data[mask]
        
        # Calculate NaN ratio within the circle
        inner_nan_ratio = np.isnan(central_pixels).sum() / central_pixels.size
        
        return inner_nan_ratio <= inner_thresh

    def is_target_in_fov(self, wcs, target_coord, shape):
        """Checks if a SkyCoord is within the pixel bounds of an image."""
        try:
            x, y = wcs.world_to_pixel(target_coord)
            return 0 <= x < shape[1] and 0 <= y < shape[0]
        except Exception:
            return False
        
    def create_galaxy_cutout(self, hdul, target_coord, cutout_size):
        """Loops through all extensions to create a matched multi-extension HDUList."""
        cutout_hdul = fits.HDUList([fits.PrimaryHDU(header=hdul[0].header)])
        
        # Loop through all extensions (skipping PrimaryHDU at 0)
        for ext in range(1, len(hdul)):
            hdu = hdul[ext]
            if hdu.data is None or hdu.data.ndim != 2:
                continue

            # Process just the SCI extension
            try:
                # 1. Grab the SCI extension (Index 1 in JWST files)
                sci_hdu = hdul['SCI']
                wcs = WCS(sci_hdu.header, naxis=2)

                # 2. Create the cutout
                cutout = Cutout2D(sci_hdu.data, target_coord, cutout_size, wcs=wcs, mode="partial")

                # 3. Create the "Centered" Header
                new_wcs = cutout.wcs
                # Center pixel (0-based)
                center_f = (cutout.data.shape[0] - 1) / 2.0 
                # Update WCS keywords
                new_wcs.wcs.crpix = [center_f + 1, center_f + 1] # +1 for FITS standard
                new_wcs.wcs.crval = [target_coord.ra.deg, target_coord.dec.deg]

                # 4. Build the final Header
                # We copy the SCI header so we keep things like 'FILTER' and 'PHOTMJSR'
                new_header = sci_hdu.header.copy()
                new_header.update(new_wcs.to_header())
                
            except Exception as e:
                print(f"Error creating cutout for {self.ids[i]}: {e}")
                return None

            # Create output HDU with original extension name preserved
            cutout_hdu = fits.ImageHDU(data=cutout.data, header=new_header)
            if 'EXTNAME' in hdu.header:
                cutout_hdu.name = hdu.header['EXTNAME']
            
            # Add to output file
            cutout_hdul.append(cutout_hdu)
                
        return cutout_hdul
    
    def save_cutout_png(self, data, angle, filter_name, output_path):
        """Generates the grayscale preview with the North/East compass."""
        plt.figure(figsize=(6, 6))      
        plt.imshow(data, origin="lower", cmap="gray")
        plt.title(filter_name)
        
        # Draw North/East compass
        ax = plt.gca()
        self.draw_compass(ax, angle_deg=angle)
        
        plt.savefig(output_path)
        plt.close()
    
    def run_survey(self, survey_label, filter_name, size_arcsec=8.0, png=True, overwrite=False):
        """The main execution method."""
        
        # Find all FITS files matching the requested filter
        filter_l = filter_name.lower()
        survey_name = survey_label.rstrip('0123456789').upper()
        indir = self.survey_dict.get(survey_label, "")
        large_mosaics = glob.glob(os.path.join(indir, f"*{filter_l}*.fits"))
            
        if len(large_mosaics) == 0:
            print(f"⚠️ No FITS files found for survey label {survey_label} with filter {filter_name}.")
            return
        
        print(f"✅ Found {len(large_mosaics)} FITS files from the {survey_name} survey with filter {filter_name}:")
        for f in large_mosaics:
            print(f"{f}")
        
        print(f"--- Processing {survey_label} | {filter_name} ---")
        
        # Create directory structure: base/survey/filter/fits
        fits_path = os.path.join(self.base_dir, survey_label, filter_name.upper(), "fits")
        png_path = os.path.join(self.base_dir, survey_label, filter_name.upper(), "png")
        os.makedirs(fits_path, exist_ok=True)
        if png: os.makedirs(png_path, exist_ok=True)
        
        # --- NEW: Overwrite/Cleanup Logic ---
        if overwrite:
            print(f"!!! Overwrite is TRUE. Cleaning old files in {survey_label}/{filter_name}...")
            # Determine which folders to check based on the 'png' flag
            folders_to_clean = [fits_path]
            if png:
                folders_to_clean.append(png_path)

            for target_dir in folders_to_clean:
                # Check if the folder exists and has elements in it
                if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
                                        
                    # Run the deletion loop
                    files = glob.glob(os.path.join(target_dir, "*"))
                    for f in files:
                        try:
                            os.remove(f)
                        except OSError as e:
                            print(f"Error deleting {f}: {e}")
        # ------------------------------------
        
        # Initialise counter for successful cutouts
        counts = 0
        total = len(self.ids)
        
        # Calculate cutout size in pixels based on MIRI instrument scale
        pixel_scale = self.pixel_scale  # arcsec/pixel
        x_pixels = int(np.round(size_arcsec/pixel_scale))
        cutout_size = (x_pixels, x_pixels)
                
        # Process each FITS file
        for mosaic_file in large_mosaics:
            with fits.open(mosaic_file) as hdul:
                # Use extension 1 as the reference for WCS and field coverage check
                ref_data = hdul['SCI'].data
                ref_header = hdul['SCI'].header
                ref_wcs = WCS(ref_header)

                # Process each galaxy from the catalogue
                for i in range(total):
                    # Create SkyCoord object for the target position
                    target_coord = SkyCoord(self.ras[i], self.decs[i], unit=(u.deg, u.deg))

                    if not self.is_target_in_fov(ref_wcs, target_coord, ref_data.shape):
                        continue
                                        
                    cutout_hdul = self.create_galaxy_cutout(hdul, target_coord, cutout_size)
                    if cutout_hdul is None:
                        continue
                    
                    # Save the cutout if it meets quality criteria (not too many NaNs and has data extensions)
                    preview_data = cutout_hdul[1].data
                    
                    if self.check_quality(preview_data) == True:
                        
                        # Calculate angle of rotation for NE cross
                        angle = self.calculate_angle(mosaic_file)  
                        #print(f"Galaxy ID {self.ids[i]}: angle = {angle:.2f} degrees")
                        
                        if png:
                            output_path = os.path.join(png_path, f"{self.ids[i]}_{filter_l}_{survey_label}.png")
                            if not os.path.exists(output_path) or overwrite:
                                self.save_cutout_png(preview_data, angle, filter_name, output_path)
                        
                        # Save multi-extension FITS cutout
                        fits_filename = os.path.join(fits_path, f"{self.ids[i]}_{filter_l}_{survey_label}.fits")
                        
                        if not os.path.exists(fits_filename) or overwrite:
                            cutout_hdul.writeto(fits_filename, overwrite=True)
                        
                        counts += 1
                        
        print(f"Files were successfully saved to {os.path.join(self.base_dir, survey_label, filter_name.upper())}.")

        # Report completion statistics
        print(f"Produced cutouts for {counts} of {total} galaxies in the catalogue.")


    @staticmethod
    def load_cutout(file_path, index=1):
        """Loads a FITS cutout file and extracts the data, header, and WCS."""
        try:
            with fits.open(file_path) as hdu:
                data = hdu[index].data
                header = hdu[index].header
                wcs = WCS(header)
            return data, wcs
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None, None


    @staticmethod
    def draw_compass(ax, angle_deg, size_pct=0.15):
        """
        Draw a North-East direction cross in the top-right corner of an image.
        
        Parameters
        ----------
        size_pct : float
            Arrow length as a fraction of the cutout width (e.g. 0.15 = 15%).
        offset_pct : float
            Padding from the top-right corner as a fraction of the cutout width.
        """
        # 1. Get current axis limits (cutout size)
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        width = abs(x_max - x_min)
        
        # 2. Scale size and offset relative to the actual cutout dimensions
        cross_size = width * size_pct
        size = cross_size / 2
        offset = 1.8 * size  # Padding from the corner

        # 3. Set Origin (Top-Right, pushed INWARD)
        # We subtract the offset so it doesn't overlap the border
        x0 = x_max - offset
        y0 = y_max - offset

        # 4. Math for Vectors
        # Note: DS9 North usually points towards higher Dec.
        # angle_deg should be the angle of North relative to the +Y axis.
        angle_rad = np.deg2rad(angle_deg)
        
        # North vector: sin/cos based on angle from Y-axis
        # We subtract the components because the origin is at the top-right; 
        # to stay inside the box, the vectors generally need to point 'down' or 'left'
        # but the math handles this if we just use the angle correctly.
        xN = x0 + size * np.sin(angle_rad)
        yN = y0 + size * np.cos(angle_rad)

        # East is typically +90 degrees from North in the coordinate system
        # but in most FITS images, East is LEFT when North is UP.
        east_angle_rad = angle_rad + np.pi/2
        xE = x0 - size * np.sin(east_angle_rad)
        yE = y0 - size * np.cos(east_angle_rad)

        # 5. Draw N/E-lines
        ax.plot([x0, xN], [y0, yN], color="yellow", lw=1.5, solid_capstyle='round')
        ax.plot([x0, xE], [y0, yE], color="yellow", lw=1.5, solid_capstyle='round')

        # 6. Labels with slight padding so they don't touch the lines
        ax.text(xN + (size*0.3 * np.sin(angle_rad)), 
                yN + (size*0.3 * np.cos(angle_rad)), 
                "N", color="yellow", fontsize=10, ha="center", va="center", fontweight='bold')
        
        ax.text(xE - (size*0.3 * np.sin(east_angle_rad)), 
                yE - (size*0.3 * np.cos(east_angle_rad)), 
                "E", color="yellow", fontsize=10, ha="center", va="center", fontweight='bold')

        # 7 Draw X/Y-lines
        xX = x0 + size
        yX = y0 # X only points to the right
        
        xY = x0 # Y only points up
        yY = y0 + size

        ax.plot([x0, xX], [y0, yX], color="cyan", lw=1.5, solid_capstyle='round')
        ax.plot([x0, xY], [y0, yY], color="cyan", lw=1.5, solid_capstyle='round')

        # 8. Labels with slight padding so they don't touch the lines
        ax.text(xX + (size*0.2), 
                yX, 
                "X", color="cyan", fontsize=10, ha="center", va="center", fontweight='bold')
        
        ax.text(xY, 
                yY + (size*0.2), 
                "Y", color="cyan", fontsize=10, ha="center", va="center", fontweight='bold')

    @staticmethod
    def calculate_angle(fits_file):
        with fits.open(fits_file) as hdul:
            # Check for SCI extension or Primary
            ext = 'SCI' if 'SCI' in hdul else 0
            header = hdul[ext].header
            w = WCS(header)

        # 1. Get the 'North' direction in pixel space
        # We look at how the sky coordinates change at the center of the image
        res = w.pixel_scale_matrix
        
        # The 'CD' or 'PC' matrix components
        # cd[1,1] is change in Dec with Y, cd[0,1] is change in RA with Y
        # This is the most robust way to find "Up" in celestial terms
        cd = res
        
        # Calculate the angle of North relative to the Y-axis (Up)
        # This automatically handles the PC matrix, scaling, and parity
        angle = np.degrees(np.arctan2(cd[0, 1], cd[1, 1]))
        
        return angle
