# SOM Classifier for Lightcurves

A Python-based tool for classifying and clustering astronomical lightcurves using Self-Organizing Maps (SOM). This project provides an efficient way to analyze time-series data from astronomical observations, identify periodic variables, and group similar lightcurves together using unsupervised machine learning.

## Overview

This toolkit combines classical time-series analysis techniques (Lomb-Scargle periodogram) with modern machine learning (Self-Organizing Maps) to process and classify astronomical lightcurves. The system can:

- Extract periods from time-series photometric data
- Phase-fold lightcurves and correct for period doubling
- Generate 1D phase-binned representations
- Train SOM neural networks for unsupervised classification
- Cluster similar lightcurves in the SOM feature space
- Save and load trained models for reuse

## Features

- **Automated Period Detection**: Uses Lomb-Scargle periodogram with noise frequency filtering
- **Phase Correction**: Automatically aligns lightcurves with minima at zero phase
- **Period Doubling Detection**: Identifies and corrects period ambiguities
- **Phase Binning**: Converts lightcurves to normalized 1D arrays for neural network input
- **Self-Organizing Maps**: Unsupervised classification using minisom
- **Model Persistence**: Save and load trained SOM models
- **Visualization**: Tools for plotting lightcurves, SOM winners, and cluster distributions

## Installation

### Prerequisites

- Python 3.6 or higher
- pip package manager

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy scipy pandas matplotlib seaborn astropy minisom scikit-learn
```

## Quick Start

### 1. Processing Lightcurves

```python
from Lctools import Lctools

# Initialize lightcurve tools
lc = Lctools()

# Load your lightcurve data (HJD, magnitude, error format)
lc.set_lc('your_lightcurve.txt')

# Set noise frequencies to filter (optional)
lc.set_noise_frequencies('noise_freq')

# Compute period using Lomb-Scargle
period = lc.lomb_scargle()

# Build dataframe and correct phase
df = lc.build_df()
phase = lc.phase_correction()

# Check for period doubling
lc.check_double_period()

# Generate phase-binned representation
binned_lc = lc.normalise(lc.phase_bin())
```

### 2. Training a SOM Classifier

```python
from SOM import SOM

# Initialize SOM
som_classifier = SOM()

# Set path to your phase-binned lightcurves
som_classifier.set_files('/path/to/lightcurve/data/')

# Load data from files
som_classifier.set_data()

# Initialize and configure the SOM network
som_classifier.set_som(sigma=0.1, learning_rate=1.5)

# Train the network
som_classifier.train_som(10000)  # 10000 training iterations

# Save the trained model
som_classifier.save_model('my_som_model')
```

### 3. Using a Trained Model

```python
# Load a pre-trained model
som_classifier.load_model('som.p')

# Get SOM winner coordinates for your data
x, y = som_classifier.get_coords()

# Visualize the distribution
som_classifier.plot_winners()
```

## Project Structure

```
SOM-Classifier-for-lightcurves/
├── README.md              # This file
├── USAGE.md              # Detailed usage guide
├── BENEFITS.md           # Use cases and benefits
├── requirements.txt      # Python dependencies
├── LICENSE               # License information
├── Lctools.py           # Lightcurve analysis tools
├── SOM.py               # SOM classifier class
├── SOM_classifier.py    # Example SOM training script
├── Classifier_step1.py  # Batch processing pipeline
└── som.p                # Pre-trained SOM model (pickled)
```

## Core Components

### Lctools.py
Provides the `Lctools` class for time-series analysis:
- Period detection via Lomb-Scargle periodogram
- Phase folding and correction
- Noise frequency filtering
- Phase binning for machine learning input
- Visualization tools

### SOM.py
Provides the `SOM` class for neural network operations:
- SOM initialization and configuration
- Training on phase-binned data
- Model persistence (save/load)
- Winner coordinate extraction
- Visualization of SOM space

### Classifier_step1.py
Batch processing pipeline that:
- Processes multiple lightcurves from a directory
- Applies period analysis and phase correction
- Generates 1D phase-binned arrays
- Saves processed data for SOM training

## Input Data Format

Lightcurve files should contain three columns (space or tab-separated):
```
HJD         Magnitude   Error
2458000.5   12.345     0.01
2458001.5   12.350     0.01
2458002.5   12.340     0.01
...
```

## Documentation

- **[USAGE.md](USAGE.md)** - Detailed usage instructions and examples
- **[BENEFITS.md](BENEFITS.md)** - Use cases, benefits, and scientific applications

## Citation

If you use this tool in your research, please cite:
```
SOM Classifier for Lightcurves
Author: Jishnu P Das
URL: https://github.com/jishnupdas/SOM-Classifier-for-lightcurves
```

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

This project uses:
- **minisom**: Minimal implementation of Self-Organizing Maps
- **astropy**: Astronomical data analysis tools
- **scikit-learn**: Machine learning utilities

## Contact

For questions or support, please open an issue on the GitHub repository.
