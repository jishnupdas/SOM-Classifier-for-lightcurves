# Usage Guide

This guide provides detailed instructions for using the SOM Classifier for Lightcurves toolkit.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Preparing Your Data](#preparing-your-data)
3. [Processing Lightcurves](#processing-lightcurves)
4. [Training SOM Models](#training-som-models)
5. [Using Pre-trained Models](#using-pre-trained-models)
6. [Batch Processing](#batch-processing)
7. [Clustering Analysis](#clustering-analysis)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

First, ensure you have Python 3.6+ installed. Then install the dependencies:

```bash
pip install -r requirements.txt
```

### Verify Installation

Test that the modules load correctly:

```python
from Lctools import Lctools
from SOM import SOM
print("Installation successful!")
```

## Preparing Your Data

### Input File Format

Your lightcurve data should be in a text file with three columns:
1. **Time** (HJD or MJD)
2. **Magnitude** (or flux)
3. **Error** (measurement uncertainty)

Example (`example_lc.txt`):
```
2458000.5 12.345 0.01
2458001.5 12.350 0.01
2458002.5 12.340 0.01
2458003.5 12.355 0.01
```

### Data Quality Requirements

- No header row (pure numerical data)
- Space or tab-separated values
- Consistent time units
- Remove or flag bad data points beforehand
- Files with NaN values will be automatically rejected

### Noise Frequency File (Optional)

If your data has known instrumental frequencies to filter, create a `noise_freq` file:
```
0.5
1.0
2.0
```

These frequencies (in the same units as your periodogram) will be excluded from period detection.

## Processing Lightcurves

### Basic Workflow

#### Step 1: Initialize and Load Data

```python
from Lctools import Lctools

# Create instance
lc = Lctools()

# Load your lightcurve
lc.set_lc('path/to/your_lightcurve.txt')

# Optional: Set noise frequencies to exclude
lc.set_noise_frequencies('noise_freq')
```

#### Step 2: Period Detection

```python
# Compute period using Lomb-Scargle periodogram
period = lc.lomb_scargle()
print(f"Detected period: {period:.6f} days")
```

The Lomb-Scargle algorithm:
- Automatically searches up to maximum_frequency=8
- Filters out noise frequencies if provided
- Returns the dominant period

#### Step 3: Phase Folding

```python
# Build dataframe with phases
df = lc.build_df()

# Correct phase so minima falls at phase 0
phase = lc.phase_correction()
```

This creates a pandas DataFrame with columns: `['MJD', 'phase', 'mag', 'err']`

#### Step 4: Period Doubling Check

```python
# Check if the period is actually double or half
doubled = lc.check_double_period()

if doubled:
    # Period was corrected, rebuild dataframe
    df = lc.build_df()
    phase = lc.phase_correction()
```

This function:
- Fits a polynomial to the phased lightcurve
- Checks variance at 2× and 0.5× period
- Automatically selects the best period

#### Step 5: Phase Binning

```python
# Generate phase-binned representation (64 bins by default)
binned = lc.phase_bin()

# Normalize for neural network input
normalized_binned = lc.normalise(binned)
```

This creates a 1D array representing the lightcurve shape, suitable for SOM input.

#### Step 6: Visualization

```python
# Plot the phased lightcurve
lc.phased_plot(lc.phase, lc.mag)
```

### Complete Example

```python
from Lctools import Lctools

def process_single_lightcurve(filename):
    """Complete processing pipeline for one lightcurve."""
    
    # Initialize
    lc = Lctools()
    lc.set_lc(filename)
    lc.set_noise_frequencies('noise_freq')
    
    # Detect period
    period = lc.lomb_scargle()
    print(f"Period: {period:.6f} days")
    
    # Phase fold
    df = lc.build_df()
    phase = lc.phase_correction()
    
    # Check for period doubling
    if lc.check_double_period():
        df = lc.build_df()
        phase = lc.phase_correction()
    
    # Generate binned representation
    binned = lc.normalise(lc.phase_bin())
    
    # Plot
    lc.phased_plot(lc.phase, lc.mag)
    
    return binned, period

# Use it
binned_lc, period = process_single_lightcurve('my_star.txt')
print(f"Binned lightcurve shape: {binned_lc.shape}")
```

## Training SOM Models

### Basic SOM Training

#### Step 1: Prepare Training Data

```python
from SOM import SOM
import glob

# Initialize SOM
som = SOM()

# Set path to directory containing phase-binned lightcurves
som.set_files('/path/to/binned_lightcurves/')

# Load all valid data (rejects files with NaN)
som.set_data()
print(f"Loaded {len(som.data)} lightcurves")
```

#### Step 2: Configure the Network

```python
# Initialize SOM with parameters
# Default: 50x50 grid, 32-dimensional input
som.set_som(sigma=0.1, learning_rate=1.5)
```

Parameters:
- **sigma**: Neighborhood radius (default: 0.1)
- **learning_rate**: How quickly weights adapt (default: 1.5)
- Network size is set via `som.network_h` and `som.network_w` (default: 50×50)

#### Step 3: Train the Network

```python
# Train with random samples
num_iterations = 10000
som.train_som(num_iterations)
print("Training complete!")
```

Training tips:
- More iterations = better convergence (but slower)
- 10,000 iterations is a good starting point
- For large datasets, try 50,000+ iterations

#### Step 4: Save the Model

```python
# Save trained model for later use
som.save_model('my_trained_som')
# This creates 'my_trained_som.p'
```

### Complete Training Example

```python
from SOM import SOM

def train_som_classifier(data_path, output_model, iterations=10000):
    """Train a new SOM classifier."""
    
    # Initialize
    som = SOM()
    
    # Load data
    som.set_files(data_path)
    som.set_data()
    print(f"Training on {len(som.data)} lightcurves")
    
    # Configure network
    som.network_h = 50
    som.network_w = 50
    som.set_som(sigma=0.1, learning_rate=1.5)
    
    # Train
    print(f"Training for {iterations} iterations...")
    som.train_som(iterations)
    
    # Save
    som.save_model(output_model)
    print(f"Model saved as {output_model}.p")
    
    return som

# Use it
trained_som = train_som_classifier(
    data_path='/data/binned_lcs/',
    output_model='variable_star_som',
    iterations=20000
)
```

## Using Pre-trained Models

### Loading a Model

```python
from SOM import SOM

# Initialize
som = SOM()

# Load pre-trained model
som.load_model('som.p')
print("Model loaded successfully")
```

### Classifying New Data

```python
# Load your data
som.set_files('/path/to/new_data/')
som.set_data()

# Get SOM coordinates for each lightcurve
x, y = som.get_coords()

# Each (x, y) pair represents the "winning" neuron
# for that lightcurve
print(f"Classified {len(x)} lightcurves")
```

### Visualizing Results

```python
# Plot the distribution of lightcurves in SOM space
som.plot_winners()
```

This creates a density plot showing:
- Scatter points for each lightcurve's winner
- KDE (kernel density estimate) contours
- Clustering structure

### Accessing Classification Results

```python
# Get coordinates
x, y = som.get_coords()

# Access specific lightcurve classifications
for i, (xi, yi) in enumerate(zip(x, y)):
    filename = som.fnames[i]
    print(f"{filename}: neuron ({xi}, {yi})")
```

## Batch Processing

### Processing Multiple Files

Use `Classifier_step1.py` as a template for batch processing:

```python
import os
import pandas as pd
from Lctools import Lctools

def batch_process(file_list, output_dir):
    """Process multiple lightcurves."""
    
    for file in file_list:
        try:
            fname = file.split('/')[-1]
            
            # Skip if already processed
            if os.path.exists(output_dir + '1D_' + fname):
                continue
            
            # Initialize tools
            lc = Lctools()
            lc.set_lc(file)
            lc.set_noise_frequencies('noise_freq')
            
            # Process
            period = lc.lomb_scargle()
            df = lc.build_df()
            phase = lc.phase_correction()
            
            # Check doubling
            if lc.check_double_period():
                df = lc.build_df()
                phase = lc.phase_correction()
            
            # Save binned result
            binned = lc.normalise(lc.phase_bin())
            with open(output_dir + '1D_' + fname, 'w') as f:
                for val in binned:
                    f.write(f"{val}\n")
            
            print(f"Processed: {fname}")
            
        except Exception as e:
            print(f"Error with {file}: {e}")
            continue

# Example usage
files = glob.glob('/data/raw_lightcurves/*.txt')
batch_process(files, '/data/processed/')
```

## Clustering Analysis

After obtaining SOM coordinates, you can perform additional clustering:

### K-Means Clustering

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Get SOM coordinates
x, y = som.get_coords()
points = list(zip(x, y))

# Perform K-means
n_clusters = 15
kmeans = KMeans(n_clusters=n_clusters)
labels = kmeans.fit_predict(points)

# Visualize clusters
for i in range(n_clusters):
    cluster_x = [x[j] for j in range(len(x)) if labels[j] == i]
    cluster_y = [y[j] for j in range(len(y)) if labels[j] == i]
    plt.scatter(cluster_x, cluster_y, label=f'Cluster {i}')

plt.legend()
plt.xlabel('SOM X')
plt.ylabel('SOM Y')
plt.title('K-means Clustering of SOM Space')
plt.show()
```

### Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

points = list(zip(x, y))

# Create dendrogram
dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

# Perform clustering
hc = AgglomerativeClustering(n_clusters=15, 
                              affinity='euclidean',
                              linkage='ward')
labels = hc.fit_predict(points)
```

## Advanced Usage

### Customizing Network Size

```python
som = SOM()
som.network_h = 100  # Larger network
som.network_w = 100
som.set_files('/path/to/data/')
som.set_data()
som.set_som(sigma=0.1, learning_rate=1.5)
```

Larger networks:
- Can capture more subtle differences
- Take longer to train
- May overfit with small datasets

### Adjusting Bin Resolution

```python
lc = Lctools()
lc.binlen = 128  # More bins (default: 64)
lc.binarr = np.linspace(0, 1, lc.binlen + 1)
```

Higher resolution:
- Captures more detail in lightcurve shape
- Requires larger input_len in SOM
- Needs more training data

### Custom Training Strategy

```python
import numpy as np

# Initialize
som = SOM()
som.set_files('/data/')
som.set_data()
som.set_som(sigma=0.3, learning_rate=2.0)  # Start with larger sigma

# Multi-stage training
stages = [(0.3, 2.0, 5000),   # Broad learning
          (0.1, 1.5, 10000),  # Standard learning
          (0.05, 0.5, 5000)]  # Fine-tuning

for sigma, lr, iters in stages:
    som.som.sigma = sigma
    som.som.learning_rate = lr
    som.train_som(iters)
    print(f"Stage complete: sigma={sigma}, lr={lr}")
```

## Troubleshooting

### Common Issues

**Problem**: "ValueError: Input contains NaN"
- **Solution**: Remove files with missing data. The `set_data()` method should filter these, but double-check your input files.

**Problem**: Poor period detection
- **Solution**: 
  - Add known noise frequencies to filter
  - Check if your data spans enough cycles
  - Try adjusting the maximum_frequency parameter in `lomb_scargle()`

**Problem**: SOM doesn't show clear clustering
- **Solution**:
  - Train for more iterations
  - Try different network sizes
  - Adjust sigma and learning_rate parameters
  - Ensure your data has enough variety

**Problem**: Period doubling issues
- **Solution**: The `check_double_period()` function should handle this automatically. If issues persist, manually inspect the phased lightcurve plots.

### Performance Tips

1. **For large datasets**: Process in batches and save intermediate results
2. **For training**: Use a GPU-enabled machine if available (though minisom is CPU-based)
3. **For memory**: Process files on-demand rather than loading all at once
4. **For speed**: Use multiprocessing for batch processing

### Getting Help

- Check the [README.md](README.md) for overview
- See [BENEFITS.md](BENEFITS.md) for use cases
- Open an issue on GitHub for bugs or questions

## Example Workflows

### Workflow 1: Survey Classification

```python
# Process survey data and classify
from Lctools import Lctools
from SOM import SOM

# 1. Process all lightcurves
input_dir = '/survey/lightcurves/'
output_dir = '/survey/processed/'

# ... batch process as shown above ...

# 2. Train SOM
som = SOM()
som.set_files(output_dir)
som.set_data()
som.set_som(sigma=0.1, learning_rate=1.5)
som.train_som(20000)
som.save_model('survey_som')

# 3. Classify and cluster
x, y = som.get_coords()
# ... apply clustering as shown above ...
```

### Workflow 2: Follow-up Classification

```python
# Classify new objects using existing model
som = SOM()
som.load_model('survey_som.p')

# Process new lightcurve
lc = Lctools()
lc.set_lc('new_target.txt')
period = lc.lomb_scargle()
# ... process as usual ...

# Save and classify
np.savetxt('new_target_binned.txt', lc.normalise(lc.phase_bin()))
som.set_files('/path/with/new_target/')
som.set_data()
x, y = som.get_coords()

print(f"New target classified to neuron: ({x[-1]}, {y[-1]})")
```

## Next Steps

- Explore the [BENEFITS.md](BENEFITS.md) file to understand use cases
- Review the source code for advanced customization
- Share your results and improvements with the community!
