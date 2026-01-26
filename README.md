# miriutils

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18378612.svg)](https://doi.org/10.5281/zenodo.18378612)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**miriutils** is a Python library designed for high-fidelity processing, astrometric alignment, and photometric analysis of JWST/MIRI data. Originally developed for the Blue Jay survey, it provides a streamlined pipeline to move from fully calibrated product level 3 mosaics to publication-ready photometric catalogues for JWST/MIRI.

## 🚀 Key Features

* **Aperture Photometry:** Automated end-to-end pipeline (`MIRIPipeline`) with local background modeling and PSF aperture corrections.
* **Astrometric Calibration:** Specialised tools to correct systematic WCS offsets between MIRI and NIRCam.
* **Data Management:** Efficient multi-extension FITS cutout generation via `CutoutManager`.
* **Visualisation:** Publication-quality RGB composition and diagnostic plotting for mid-infrared datasets.

## 📁 Repository Structure

```text
miriutils/
├── miriutils/               # Core Package
│   ├── __init__.py          # Version & Top-level exports
│   ├── astrometry_utils.py  # WCS alignment & Offset tools
│   ├── miricut.py           # CutoutManager & Quality Control
│   ├── photometry_tools.py  # MIRIPipeline & Flux Calibration
│   └── vis.py               # RGBComposer & Plotting
├── CITATION.cff             # Citation metadata for Zenodo
├── LICENSE                  # BSD-3-Clause License
└── README.md
```

## 🛠 Installation

Currently, `miriutils` can be installed by cloning the repository:

```bash
git clone [https://github.com/benjaminpcollins/miriutils.git](https://github.com/benjaminpcollins/miriutils.git)
cd miriutils
pip install -e .
```

## 📖 Quick Start
```python
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
```

## 📫 Contact

For questions or collaborations, feel free to reach out via email or GitHub.

Email: benjamin.p.collins@icloud.com