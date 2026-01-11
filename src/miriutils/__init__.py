# __init__.py for the miri_utils package

from .miricut import CutoutManager
from .astrometry_utils import compute_centroid, save_alignment_figure, compute_offset, get_path, generate_flag_sheet, get_survey_stats, \
    generate_master_summary, apply_wcs_shift, display_offsets
from .photometry_tools import MIRIPipeline, adjust_aperture, estimate_background, get_psf, get_aperture_params, calculate_aperture_correction, measure_flux, \
    perform_photometry, create_fits_table_from_csv, compare_aperture_statistics, write_detection_stats, plot_galaxy_filter_matrix, \
        save_vis, load_vis, create_mosaics, plot_aperture_comparison, write_aperture_summary, plot_aperture_summary, plot_appendix_figure, \
            analyse_outliers, recompute_empirical_snr, show_apertures
from .vis import RGBComposer

# Alternative dynamic approach (replaces the manual __all__ list)
__all__ = [name for name in globals() if not name.startswith('_')]