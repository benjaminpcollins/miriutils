"""
miriutils: A suite of tools for JWST/MIRI aperture photometry.
"""

__version__ = "1.0.1"
__author__ = "Benjamin P. Collins"
__email__ = "benjamin.p.collins@icloud.com"
__license__ = "BSD-3-Clause"


from .miricut import CutoutManager
from .astrometry_utils import compute_centroid, save_alignment_figure, compute_offset, get_path, generate_flag_sheet, get_survey_stats, \
    generate_master_summary, apply_wcs_shift, display_offsets
from .photometry_tools import MIRIPipeline
from .vis import RGBComposer

# Alternative dynamic approach (replaces the manual __all__ list)
__all__ = [name for name in globals() if not name.startswith('_')]