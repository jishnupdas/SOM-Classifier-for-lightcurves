# Project Improvements and Best Practices

This document provides a comprehensive analysis of the SOM Classifier for Lightcurves project, identifying gaps, suggesting improvements, and outlining best practices for development, testing, documentation, and deployment.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Gap Analysis](#gap-analysis)
3. [Code Quality and Structure](#code-quality-and-structure)
4. [Best Practices](#best-practices)
5. [Testing Strategy](#testing-strategy)
6. [Development Workflow](#development-workflow)
7. [Documentation Improvements](#documentation-improvements)
8. [Performance Optimization](#performance-optimization)
9. [Security Considerations](#security-considerations)
10. [Deployment and Distribution](#deployment-and-distribution)
11. [Project Structure Recommendations](#project-structure-recommendations)
12. [Roadmap](#roadmap)

---

## Executive Summary

The SOM Classifier for Lightcurves is a well-documented Python tool for astronomical data analysis. However, there are several areas for improvement:

### Strengths
✅ Comprehensive documentation (README, USAGE, BENEFITS)  
✅ Clear, focused purpose and use cases  
✅ Simple, accessible API design  
✅ Good use of scientific libraries (astropy, minisom)  

### Areas for Improvement
❌ No automated testing infrastructure  
❌ No CI/CD pipeline  
❌ Hard-coded paths in example scripts  
❌ Limited error handling and validation  
❌ No code style enforcement (linting)  
❌ No package structure (setup.py/pyproject.toml)  
❌ Limited logging capabilities  
❌ No versioning system  
❌ No contribution guidelines  

---

## Gap Analysis

### 1. Testing Infrastructure (Critical)

**Status**: ❌ Missing  
**Impact**: High  
**Priority**: Critical

**Current State**:
- No test files exist
- No testing framework configured
- No test coverage measurement
- Manual testing only

**Recommended Actions**:
- Implement pytest framework
- Create unit tests for all classes and methods
- Add integration tests for complete workflows
- Set up test coverage reporting (target: >80%)
- Add test data fixtures
- Implement property-based testing for numerical functions

**Example Test Structure**:
```
tests/
├── __init__.py
├── test_lctools.py
├── test_som.py
├── test_integration.py
├── fixtures/
│   ├── sample_lightcurve.txt
│   └── sample_binned_data.txt
└── conftest.py
```

---

### 2. Continuous Integration/Deployment (Critical)

**Status**: ❌ Missing  
**Impact**: High  
**Priority**: Critical

**Current State**:
- No GitHub Actions or other CI configured
- No automated testing on PRs
- No automated checks for code quality
- No automated releases

**Recommended Actions**:
- Set up GitHub Actions workflows
- Automate testing on all PRs and commits
- Add linting and formatting checks
- Implement automated dependency updates
- Set up automated releases with versioning
- Add build status badges to README

**Example Workflow Structure**:
```
.github/
├── workflows/
│   ├── tests.yml           # Run tests on PR/push
│   ├── lint.yml            # Code quality checks
│   ├── release.yml         # Automated releases
│   └── dependencies.yml    # Dependabot integration
└── PULL_REQUEST_TEMPLATE.md
```

---

### 3. Package Structure (High Priority)

**Status**: ❌ Missing  
**Impact**: High  
**Priority**: High

**Current State**:
- No setup.py or pyproject.toml
- Cannot install via pip
- No version management
- Modules not organized as package

**Recommended Actions**:
- Create proper Python package structure
- Add setup.py or pyproject.toml
- Implement semantic versioning
- Support pip installation
- Add package metadata
- Publish to PyPI

**Recommended Structure**:
```
som_classifier/
├── __init__.py
├── __version__.py
├── lctools.py
├── som.py
├── utils.py
└── cli.py
```

---

### 4. Code Quality Tools (High Priority)

**Status**: ❌ Missing  
**Impact**: Medium  
**Priority**: High

**Current State**:
- No linting configuration
- No code formatting standards
- No static type checking
- Inconsistent code style

**Recommended Actions**:
- Add Black for code formatting
- Add Flake8/Pylint for linting
- Add mypy for type checking
- Add isort for import sorting
- Add pre-commit hooks
- Document code style in CONTRIBUTING.md

**Configuration Files Needed**:
- `.flake8` or `setup.cfg`
- `pyproject.toml` (for Black, isort)
- `mypy.ini` or inline types
- `.pre-commit-config.yaml`

---

### 5. Error Handling and Validation (High Priority)

**Status**: ⚠️ Minimal  
**Impact**: Medium  
**Priority**: High

**Current State**:
- Basic try-except blocks in scripts
- Limited input validation
- Generic error messages
- No custom exceptions

**Recommended Actions**:
- Implement custom exception classes
- Add comprehensive input validation
- Provide informative error messages
- Add logging instead of print statements
- Validate file formats explicitly
- Add type hints and runtime type checking

**Example Custom Exceptions**:
```python
class LightcurveError(Exception):
    """Base exception for lightcurve processing"""

class InvalidDataError(LightcurveError):
    """Raised when data format is invalid"""

class PeriodDetectionError(LightcurveError):
    """Raised when period detection fails"""
```

---

### 6. Logging System (Medium Priority)

**Status**: ⚠️ Minimal (print statements)  
**Impact**: Medium  
**Priority**: Medium

**Current State**:
- Uses print() for output
- No structured logging
- No log levels
- No log file output
- Hard to debug production issues

**Recommended Actions**:
- Implement Python logging module
- Add configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Support log file output
- Add structured logging (JSON format option)
- Include timestamp, module, function in logs
- Add verbose/quiet modes

---

### 7. Configuration Management (Medium Priority)

**Status**: ❌ Missing  
**Impact**: Medium  
**Priority**: Medium

**Current State**:
- Hard-coded values (bin length, network size)
- No configuration files
- Parameters scattered across code
- Hard to modify defaults

**Recommended Actions**:
- Add configuration file support (YAML/JSON/TOML)
- Create default configuration
- Allow command-line overrides
- Document all configuration options
- Support environment variables
- Validate configuration on load

**Example Configuration**:
```yaml
# config.yaml
lightcurve:
  bin_length: 64
  max_frequency: 8
  
som:
  network_height: 50
  network_width: 50
  sigma: 0.1
  learning_rate: 1.5
  
processing:
  noise_frequencies: []
  check_doubling: true
```

---

### 8. Command-Line Interface (Medium Priority)

**Status**: ❌ Missing  
**Impact**: Medium  
**Priority**: Medium

**Current State**:
- No CLI tool
- Requires Python script modification
- Not user-friendly for non-programmers

**Recommended Actions**:
- Create CLI using Click or argparse
- Support common operations via command line
- Add progress bars for batch operations
- Provide helpful error messages
- Support piping and batch modes
- Add shell completion

**Example CLI Design**:
```bash
# Process single lightcurve
som-classifier process input.txt -o output.txt

# Train model
som-classifier train data/ -o model.p --iterations 10000

# Classify with model
som-classifier classify model.p data/ -o results.csv

# Batch processing
som-classifier batch process input_dir/ -o output_dir/
```

---

### 9. Documentation Gaps (Low-Medium Priority)

**Status**: ⚠️ Good but incomplete  
**Impact**: Low-Medium  
**Priority**: Medium

**Current State**:
- Good README, USAGE, BENEFITS docs
- No API documentation
- No docstring consistency
- No contribution guidelines
- No changelog

**Recommended Actions**:
- Add CONTRIBUTING.md
- Add CHANGELOG.md
- Generate API docs with Sphinx
- Ensure all functions have docstrings
- Add code examples in docstrings
- Create tutorials/notebooks
- Add FAQ section

---

### 10. Dependency Management (Low-Medium Priority)

**Status**: ⚠️ Basic  
**Impact**: Low-Medium  
**Priority**: Medium

**Current State**:
- Requirements.txt exists
- No version pinning
- No dependency vulnerability scanning
- No separate dev dependencies

**Recommended Actions**:
- Pin dependency versions
- Separate dev and prod dependencies
- Add dependency scanning (Dependabot/Snyk)
- Document minimum required versions
- Test with multiple dependency versions
- Consider using Poetry or pipenv

---

### 11. Performance Optimization (Low Priority)

**Status**: ⚠️ Not optimized  
**Impact**: Low-Medium  
**Priority**: Low-Medium

**Current State**:
- No performance profiling
- Sequential processing only
- No caching mechanisms
- Loading all data into memory

**Recommended Actions**:
- Add multiprocessing support for batch operations
- Implement data streaming for large datasets
- Add caching for expensive operations
- Profile and optimize hot paths
- Consider using numba for critical loops
- Add progress indicators

---

### 12. Example Data and Tutorials (Low Priority)

**Status**: ❌ Missing  
**Impact**: Low  
**Priority**: Low

**Current State**:
- No sample data included
- No Jupyter notebooks
- No step-by-step tutorials

**Recommended Actions**:
- Add sample lightcurve data
- Create Jupyter notebook tutorials
- Add end-to-end examples
- Create video tutorials (optional)
- Add troubleshooting guide with examples

---

## Code Quality and Structure

### Current Code Issues

#### 1. Hard-coded Paths
**Files**: `SOM_classifier.py`, `Classifier_step1.py`

**Issue**:
```python
dpath = '/home/jishnu/Documents/TESS/tess_data/1D_lc/'
```

**Solution**:
- Use command-line arguments
- Use configuration files
- Use environment variables
- Use relative paths when appropriate

#### 2. Magic Numbers
**Files**: `Lctools.py`, `SOM.py`

**Issue**:
```python
cf = np.polyfit(phase, mag, 30)  # Why 30?
self.binlen = 64  # Why 64?
```

**Solution**:
```python
# Define as constants with documentation
POLYFIT_DEGREE = 30  # Order for lightcurve fitting
DEFAULT_BIN_LENGTH = 64  # Phase bins for SOM input
```

#### 3. Inconsistent Error Handling

**Issue**:
```python
try:
    # code
except:
    print("error")  # Too broad, no info
```

**Solution**:
```python
try:
    # code
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise
except IOError as e:
    logger.error(f"File operation failed: {e}")
    raise
```

#### 4. Limited Type Hints

**Current**:
```python
def set_lc(self, file):
```

**Improved**:
```python
def set_lc(self, file: str) -> None:
    """Load lightcurve from file.
    
    Args:
        file: Path to lightcurve file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
```

---

## Best Practices

### Python Best Practices

#### 1. PEP 8 Compliance
- Follow PEP 8 style guide
- Use Black for auto-formatting
- Maximum line length: 88 (Black) or 79 (PEP 8)
- Use meaningful variable names

#### 2. Documentation Standards
- Follow PEP 257 for docstrings
- Use Google or NumPy docstring format
- Document all public APIs
- Include type hints (PEP 484)

**Example**:
```python
def lomb_scargle(self, maximum_frequency: float = 8.0) -> float:
    """Compute period using Lomb-Scargle periodogram.
    
    Uses the LombScargle method from astropy to detect the dominant
    period in the lightcurve data. Automatically filters noise
    frequencies if configured.
    
    Args:
        maximum_frequency: Maximum frequency to search (default: 8.0)
        
    Returns:
        Detected period in same units as input time
        
    Raises:
        ValueError: If no valid period is detected
        
    Example:
        >>> lc = Lctools()
        >>> lc.set_lc('star.txt')
        >>> period = lc.lomb_scargle()
        >>> print(f"Period: {period:.4f} days")
    """
```

#### 3. Code Organization

**Single Responsibility Principle**:
- Each class should have one purpose
- Each method should do one thing
- Split large functions into smaller ones

**Example Refactoring**:
```python
# Before: One large function
def main(file):
    # 50 lines doing everything

# After: Multiple focused functions
def load_and_validate(file: str) -> Lctools:
    """Load and validate lightcurve file."""
    
def detect_period(lc: Lctools) -> float:
    """Detect period with doubling check."""
    
def generate_binned_representation(lc: Lctools) -> np.ndarray:
    """Create phase-binned array."""
    
def save_processed_lightcurve(binned: np.ndarray, output_path: str) -> None:
    """Save binned data to file."""
```

#### 4. Dependency Injection

**Before**:
```python
class Lctools:
    def lomb_scargle(self):
        freq, power = LombScargle(t, y).autopower(maximum_frequency=8)
```

**After**:
```python
class Lctools:
    def __init__(self, max_frequency: float = 8.0):
        self.max_frequency = max_frequency
        
    def lomb_scargle(self):
        freq, power = LombScargle(t, y).autopower(
            maximum_frequency=self.max_frequency
        )
```

---

### Scientific Computing Best Practices

#### 1. Reproducibility
- Set random seeds
- Document package versions
- Provide sample data
- Include environment specs

```python
def train_som(self, iterations: int, seed: int = 42) -> None:
    """Train SOM with reproducible results."""
    np.random.seed(seed)
    self.som.train_random(self.data, iterations)
```

#### 2. Data Validation
```python
def validate_lightcurve_data(data: np.ndarray) -> None:
    """Validate lightcurve data format and quality."""
    if data.shape[1] != 3:
        raise ValueError("Expected 3 columns: time, magnitude, error")
    
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")
    
    if np.any(data[:, 2] <= 0):
        raise ValueError("Errors must be positive")
    
    if len(data) < 10:
        raise ValueError("Insufficient data points (minimum 10)")
```

#### 3. Numerical Stability
```python
# Before
array = np.array(array) / max(array)

# After
def normalize(array: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Normalize array with numerical stability."""
    max_val = np.max(array)
    if max_val < epsilon:
        raise ValueError("Cannot normalize array with zero maximum")
    return array / max_val
```

---

## Testing Strategy

### Recommended Testing Pyramid

```
                    /\
                   /  \
                  /E2E \
                 /______\
                /        \
               /Integration\
              /____________\
             /              \
            /  Unit Tests   \
           /________________\
```

### 1. Unit Tests (70%)

Test individual functions and methods in isolation.

```python
# tests/test_lctools.py
import pytest
import numpy as np
from som_classifier import Lctools

class TestLctools:
    def test_normalize_valid_array(self):
        """Test normalization with valid positive array."""
        lc = Lctools()
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = lc.normalise(arr)
        np.testing.assert_array_almost_equal(
            result,
            np.array([0.25, 0.5, 0.75, 1.0])
        )
    
    def test_normalize_zero_maximum(self):
        """Test normalization fails gracefully with zero max."""
        lc = Lctools()
        arr = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="zero maximum"):
            lc.normalise(arr)
    
    @pytest.mark.parametrize("bins,expected_len", [
        (32, 32),
        (64, 64),
        (128, 128),
    ])
    def test_phase_bin_lengths(self, bins, expected_len):
        """Test phase binning with different bin counts."""
        lc = Lctools()
        lc.binlen = bins
        lc.binarr = np.linspace(0, 1, bins + 1)
        # Setup test data
        # ...
        result = lc.phase_bin()
        assert len(result) == expected_len
```

### 2. Integration Tests (20%)

Test workflows and interactions between components.

```python
# tests/test_integration.py
def test_complete_processing_pipeline(tmp_path, sample_lightcurve):
    """Test end-to-end lightcurve processing."""
    # Create temporary input file
    input_file = tmp_path / "test_lc.txt"
    np.savetxt(input_file, sample_lightcurve)
    
    # Process
    lc = Lctools()
    lc.set_lc(str(input_file))
    period = lc.lomb_scargle()
    lc.build_df()
    lc.phase_correction()
    binned = lc.phase_bin()
    normalized = lc.normalise(binned)
    
    # Verify
    assert period > 0
    assert len(normalized) == 64
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)
    assert np.max(normalized) == pytest.approx(1.0)
```

### 3. End-to-End Tests (10%)

Test complete user workflows.

```python
def test_train_and_classify_workflow(tmp_path, sample_data_dir):
    """Test complete SOM training and classification."""
    # Train
    som = SOM()
    som.set_files(str(sample_data_dir))
    som.set_data()
    som.set_som(sigma=0.1, learning_rate=1.5)
    som.train_som(1000)
    
    # Save
    model_path = tmp_path / "test_model.p"
    som.save_model(str(model_path.with_suffix('')))
    
    # Load and classify
    som2 = SOM()
    som2.load_model(str(model_path))
    som2.set_files(str(sample_data_dir))
    som2.set_data()
    x, y = som2.get_coords()
    
    # Verify
    assert len(x) == len(som2.data)
    assert len(y) == len(som2.data)
```

### 4. Property-Based Testing

Use hypothesis for testing numerical properties.

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

@given(npst.arrays(
    dtype=np.float64,
    shape=st.integers(min_value=10, max_value=1000),
    elements=st.floats(min_value=0.1, max_value=100.0)
))
def test_normalize_properties(arr):
    """Test normalization properties hold for any valid input."""
    lc = Lctools()
    result = lc.normalise(arr)
    
    # Properties that should always hold
    assert np.max(result) <= 1.0
    assert np.min(result) >= 0.0
    assert len(result) == len(arr)
    assert np.max(result) == pytest.approx(1.0, abs=1e-10)
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
import numpy as np

@pytest.fixture
def sample_lightcurve():
    """Generate synthetic lightcurve for testing."""
    t = np.linspace(0, 100, 500)
    period = 2.5
    phase = (t % period) / period
    mag = 12.0 + 0.5 * np.sin(2 * np.pi * phase)
    err = np.random.normal(0, 0.01, len(t))
    return np.column_stack([t, mag, err])

@pytest.fixture
def sample_binned_data():
    """Generate sample phase-binned data."""
    return np.random.random(64)

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory with sample data files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    for i in range(10):
        data = np.random.random(64)
        file_path = data_dir / f"lc_{i}.txt"
        np.savetxt(file_path, data)
    
    return data_dir
```

---

## Development Workflow

### Recommended Git Workflow

#### Branch Strategy
```
main (stable, production-ready)
├── develop (integration branch)
│   ├── feature/add-cli
│   ├── feature/add-tests
│   ├── bugfix/period-detection
│   └── hotfix/critical-bug
```

#### Commit Message Convention

Follow Conventional Commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(cli): add command-line interface for batch processing

- Add Click-based CLI
- Support single file and batch modes
- Add progress bars for long operations

Closes #123

fix(lctools): handle edge case in period doubling detection

When variance is exactly 0.0001, the function incorrectly flagged
period doubling. Changed comparison to strict inequality.

Fixes #456
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.8

  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=88', '--extend-ignore=E203']

  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ['--profile=black']

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-json
      - id: check-toml
      - id: detect-private-key

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### Code Review Checklist

**For Authors**:
- [ ] Code follows project style guide
- [ ] All tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation
- [ ] No hard-coded values or paths
- [ ] Error handling is appropriate
- [ ] Commit messages are clear
- [ ] PR description explains the change

**For Reviewers**:
- [ ] Code is readable and maintainable
- [ ] Logic is correct
- [ ] Edge cases are handled
- [ ] Tests are comprehensive
- [ ] No security vulnerabilities
- [ ] Performance is acceptable
- [ ] Documentation is updated
- [ ] Breaking changes are documented

---

## Documentation Improvements

### 1. API Documentation with Sphinx

```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme

# Initialize
sphinx-quickstart docs

# Generate API docs
sphinx-apidoc -o docs/api som_classifier
```

**docs/conf.py**:
```python
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

html_theme = 'sphinx_rtd_theme'
```

### 2. CONTRIBUTING.md

Create comprehensive contribution guidelines:

```markdown
# Contributing to SOM Classifier for Lightcurves

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a virtual environment
4. Install development dependencies: `pip install -e ".[dev]"`
5. Install pre-commit hooks: `pre-commit install`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Write tests
4. Run tests: `pytest`
5. Run linters: `flake8 som_classifier tests`
6. Format code: `black som_classifier tests`
7. Commit with conventional commit message
8. Push and create PR

## Testing

- Write unit tests for all new functions
- Maintain >80% code coverage
- Use fixtures for test data
- Follow existing test structure

## Code Style

- Follow PEP 8
- Use Black for formatting
- Add type hints
- Write docstrings for all public APIs
- Keep functions small and focused

## Documentation

- Update README if adding features
- Add docstrings to new functions
- Update CHANGELOG.md
- Add examples for new functionality
```

### 3. CHANGELOG.md

Maintain a changelog following Keep a Changelog format:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Command-line interface for common operations
- Comprehensive test suite with >80% coverage
- Configuration file support (YAML)

### Changed
- Improved error handling with custom exceptions
- Refactored code for better maintainability

### Fixed
- Period doubling detection edge case
- NaN handling in phase binning

## [1.0.0] - 2024-01-15

### Added
- Initial release
- Lctools class for lightcurve analysis
- SOM class for classification
- Basic documentation
```

### 4. Enhanced README Sections

Add the following sections to README.md:

```markdown
## Installation from Source

```bash
git clone https://github.com/jishnupdas/SOM-Classifier-for-lightcurves.git
cd SOM-Classifier-for-lightcurves
pip install -e .
```

## Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run tests
pytest

# With coverage
pytest --cov=som_classifier --cov-report=html
```

## Troubleshooting

### Common Issues

**Q: Period detection fails with "No valid period found"**
A: Ensure your lightcurve has sufficient data points (>50) and spans
multiple periods. Try adjusting the maximum_frequency parameter.

**Q: NaN values in phase binning**
A: Some phase bins may be empty. Use interpolation or reduce bin count.
```

---

## Performance Optimization

### 1. Multiprocessing for Batch Operations

```python
# som_classifier/parallel.py
from multiprocessing import Pool, cpu_count
from typing import List, Callable
import os

def process_lightcurve_parallel(
    files: List[str],
    processor_func: Callable,
    n_processes: int = None
) -> List:
    """Process multiple lightcurves in parallel.
    
    Args:
        files: List of file paths to process
        processor_func: Function to apply to each file
        n_processes: Number of parallel processes (default: CPU count)
        
    Returns:
        List of processing results
    """
    if n_processes is None:
        n_processes = cpu_count()
    
    with Pool(n_processes) as pool:
        results = pool.map(processor_func, files)
    
    return results
```

### 2. Caching Expensive Operations

```python
from functools import lru_cache

class Lctools:
    @lru_cache(maxsize=128)
    def _compute_periodogram(self, data_hash: int) -> tuple:
        """Cached periodogram computation."""
        # Expensive operation
        freq, power = LombScargle(self.jd, self.mag).autopower()
        return freq, power
```

### 3. Memory-Efficient Data Loading

```python
def load_data_generator(file_list: List[str]):
    """Generator for memory-efficient data loading."""
    for file in file_list:
        data = np.loadtxt(file)
        if not np.isnan(data).any():
            yield file, data

# Usage
som = SOM()
for fname, data in load_data_generator(files):
    # Process one at a time
    som.process_single(fname, data)
```

### 4. Vectorization

```python
# Before: Loop
bn = []
for i in range(1, len(x)):
    bins = df[df.phase.between(x[i-1], x[i])].mag
    bn.append(np.mean(bins))

# After: Vectorized
bin_indices = np.digitize(df.phase, x)
bn = [df[bin_indices == i].mag.mean() for i in range(1, len(x))]
```

### 5. Progress Indicators

```python
from tqdm import tqdm

def batch_process_with_progress(files: List[str]):
    """Batch process with progress bar."""
    results = []
    for file in tqdm(files, desc="Processing lightcurves"):
        try:
            result = process_lightcurve(file)
            results.append(result)
        except Exception as e:
            tqdm.write(f"Error with {file}: {e}")
    return results
```

---

## Security Considerations

### 1. Input Validation

```python
import os
from pathlib import Path

def validate_file_path(file_path: str, allowed_extensions: tuple = ('.txt', '.dat')) -> Path:
    """Validate and sanitize file path.
    
    Args:
        file_path: Path to validate
        allowed_extensions: Tuple of allowed file extensions
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid or unsafe
    """
    path = Path(file_path).resolve()
    
    # Check file exists
    if not path.exists():
        raise ValueError(f"File not found: {file_path}")
    
    # Check it's a file, not directory
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    # Check extension
    if path.suffix not in allowed_extensions:
        raise ValueError(
            f"Invalid file extension: {path.suffix}. "
            f"Allowed: {allowed_extensions}"
        )
    
    # Prevent path traversal
    if ".." in path.parts:
        raise ValueError("Path traversal detected")
    
    return path
```

### 2. Safe Pickle Loading

```python
import pickle
from typing import Any

class SafeUnpickler(pickle.Unpickler):
    """Safe unpickler that only allows specific classes."""
    
    ALLOWED_CLASSES = {'minisom.MiniSom', 'numpy.ndarray'}
    
    def find_class(self, module: str, name: str) -> Any:
        """Only allow safe classes to be unpickled."""
        full_name = f"{module}.{name}"
        if full_name not in self.ALLOWED_CLASSES:
            raise pickle.UnpicklingError(
                f"Unpickling {full_name} is not allowed"
            )
        return super().find_class(module, name)

def load_model_safe(file_path: str) -> 'MiniSom':
    """Safely load SOM model from pickle file."""
    path = validate_file_path(file_path, ('.p', '.pkl', '.pickle'))
    
    with open(path, 'rb') as f:
        return SafeUnpickler(f).load()
```

### 3. Dependency Scanning

Add to CI workflow:

```yaml
# .github/workflows/security.yml
name: Security Scan

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run safety check
        run: |
          pip install safety
          safety check -r requirements.txt
          
      - name: Run bandit
        run: |
          pip install bandit
          bandit -r som_classifier/ -f json -o bandit-report.json
```

### 4. Secrets Management

```python
import os
from typing import Optional

def get_api_key(env_var: str = 'SOM_API_KEY') -> Optional[str]:
    """Safely retrieve API key from environment."""
    api_key = os.getenv(env_var)
    
    if not api_key:
        raise ValueError(
            f"API key not found. Set {env_var} environment variable."
        )
    
    # Don't log the actual key
    logger.info(f"API key loaded from {env_var}")
    return api_key

# Never do this:
# API_KEY = "sk-1234567890"  # Hard-coded secret!

# Do this:
# API_KEY = get_api_key()
```

---

## Deployment and Distribution

### 1. Package for PyPI

**pyproject.toml**:
```toml
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "som-lightcurve-classifier"
version = "1.0.0"
description = "Self-Organizing Map classifier for astronomical lightcurves"
readme = "README.md"
authors = [
    {name = "Jishnu P Das", email = "your.email@example.com"}
]
license = {text = "MIT"}
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Astronomy",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
keywords = ["astronomy", "lightcurves", "SOM", "classification", "machine-learning"]
dependencies = [
    "numpy>=1.16.0",
    "scipy>=1.2.0",
    "pandas>=0.24.0",
    "astropy>=3.1.0",
    "minisom>=2.2.0",
    "scikit-learn>=0.20.0",
    "matplotlib>=3.0.0",
    "seaborn>=0.9.0",
]
requires-python = ">=3.8"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "flake8>=6.0",
    "mypy>=1.0",
    "isort>=5.12",
    "pre-commit>=3.0",
]
docs = [
    "sphinx>=5.0",
    "sphinx-rtd-theme>=1.2",
]

[project.urls]
Homepage = "https://github.com/jishnupdas/SOM-Classifier-for-lightcurves"
Documentation = "https://som-classifier-lightcurves.readthedocs.io"
Repository = "https://github.com/jishnupdas/SOM-Classifier-for-lightcurves"
Issues = "https://github.com/jishnupdas/SOM-Classifier-for-lightcurves/issues"

[project.scripts]
som-classifier = "som_classifier.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["som_classifier*"]
exclude = ["tests*"]

[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --cov=som_classifier --cov-report=html --cov-report=term"

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### 2. Docker Support

**Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY som_classifier/ ./som_classifier/
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 somuser && \
    chown -R somuser:somuser /app
USER somuser

# Default command
CMD ["python", "-m", "som_classifier"]
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  som-classifier:
    build: .
    volumes:
      - ./data:/app/data
      - ./output:/app/output
    environment:
      - PYTHONUNBUFFERED=1
    command: som-classifier process /app/data -o /app/output
```

### 3. Continuous Deployment

**.github/workflows/release.yml**:
```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
          
      - name: Build package
        run: python -m build
        
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
        
      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          draft: false
          prerelease: false
```

---

## Project Structure Recommendations

### Recommended Final Structure

```
SOM-Classifier-for-lightcurves/
├── .github/
│   ├── workflows/
│   │   ├── tests.yml
│   │   ├── lint.yml
│   │   ├── release.yml
│   │   └── security.yml
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
│
├── docs/
│   ├── conf.py
│   ├── index.rst
│   ├── api/
│   ├── tutorials/
│   └── examples/
│
├── som_classifier/
│   ├── __init__.py
│   ├── __version__.py
│   ├── lctools.py
│   ├── som.py
│   ├── utils.py
│   ├── exceptions.py
│   ├── config.py
│   ├── parallel.py
│   └── cli.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_lctools.py
│   ├── test_som.py
│   ├── test_utils.py
│   ├── test_integration.py
│   └── fixtures/
│       ├── sample_lightcurve.txt
│       └── sample_binned_data.txt
│
├── examples/
│   ├── basic_usage.py
│   ├── batch_processing.py
│   ├── advanced_som.py
│   └── notebooks/
│       ├── tutorial_01_basics.ipynb
│       └── tutorial_02_advanced.ipynb
│
├── data/
│   ├── sample/
│   │   └── example_lightcurve.txt
│   └── models/
│       └── pretrained_som.p
│
├── scripts/
│   ├── benchmark.py
│   └── generate_test_data.py
│
├── .gitignore
├── .pre-commit-config.yaml
├── .flake8
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── setup.py
├── setup.cfg
├── requirements.txt
├── requirements-dev.txt
├── MANIFEST.in
├── LICENSE
├── README.md
├── USAGE.md
├── BENEFITS.md
├── IMPROVEMENTS.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
└── CITATION.cff
```

### Migration Plan

1. **Phase 1: Foundation** (Week 1)
   - Create package structure
   - Add setup.py/pyproject.toml
   - Set up Git branching strategy
   - Add .gitignore improvements

2. **Phase 2: Testing** (Week 2)
   - Create test framework
   - Write unit tests
   - Add CI for tests
   - Set up coverage reporting

3. **Phase 3: Code Quality** (Week 3)
   - Add linting configuration
   - Add type hints
   - Refactor code
   - Set up pre-commit hooks

4. **Phase 4: Documentation** (Week 4)
   - Generate API docs
   - Add CONTRIBUTING.md
   - Add CHANGELOG.md
   - Create tutorials

5. **Phase 5: Distribution** (Week 5)
   - Publish to PyPI
   - Add Docker support
   - Set up CD pipeline
   - Create release process

---

## Roadmap

### Version 1.1.0 (Next Release)
**Target: Q2 2024**

- [ ] Add comprehensive test suite
- [ ] Implement CLI interface
- [ ] Add configuration file support
- [ ] Set up CI/CD pipeline
- [ ] Publish to PyPI
- [ ] Add type hints to all functions

### Version 1.2.0
**Target: Q3 2024**

- [ ] Add multiprocessing support
- [ ] Implement caching for performance
- [ ] Add progress indicators
- [ ] Create Jupyter notebook tutorials
- [ ] Add interactive visualization tools
- [ ] Support additional input formats (FITS, CSV)

### Version 2.0.0
**Target: Q4 2024**

- [ ] Major refactoring for better architecture
- [ ] Add plugin system for custom processors
- [ ] Web interface for visualization
- [ ] Support for streaming data processing
- [ ] Integration with astronomical databases (SIMBAD, VizieR)
- [ ] Advanced clustering algorithms

### Version 2.1.0
**Target: Q1 2025**

- [ ] GPU acceleration support
- [ ] Distributed computing support
- [ ] Real-time classification pipeline
- [ ] Cloud deployment options
- [ ] REST API for remote processing

---

## Quick Wins (Immediate Improvements)

These can be implemented quickly for immediate benefit:

### 1. Add Type Hints (1 day)
```python
# Add to all function signatures
def set_lc(self, file: str) -> None:
def lomb_scargle(self) -> float:
def phase_bin(self) -> np.ndarray:
```

### 2. Improve Error Messages (1 day)
```python
# Before
except:
    print("error")

# After
except FileNotFoundError as e:
    logger.error(f"Cannot find file {file}: {e}")
    raise
except ValueError as e:
    logger.error(f"Invalid data format in {file}: {e}")
    raise
```

### 3. Add Constants (0.5 days)
```python
# At top of file
DEFAULT_BIN_LENGTH = 64
DEFAULT_POLYFIT_DEGREE = 30
DEFAULT_NETWORK_SIZE = 50
MAX_FREQUENCY_DEFAULT = 8.0
VARIANCE_THRESHOLD = 0.0001
```

### 4. Add Logging (1 day)
```python
import logging

logger = logging.getLogger(__name__)

# Replace print statements
logger.info(f"Processing file: {filename}")
logger.warning(f"Variance threshold exceeded: {variance}")
logger.error(f"Failed to process {filename}: {error}")
```

### 5. Create .flake8 Configuration (0.5 days)
```ini
# .flake8
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = 
    .git,
    __pycache__,
    .pytest_cache,
    venv,
    build,
    dist
```

---

## Conclusion

This document outlines a comprehensive plan for improving the SOM Classifier for Lightcurves project. Prioritize based on:

1. **Critical**: Testing, CI/CD, package structure
2. **High**: Code quality, error handling, logging
3. **Medium**: Configuration, CLI, documentation
4. **Low**: Performance optimization, advanced features

Focus on quick wins first to establish good practices, then systematically address larger improvements.

### Key Metrics to Track

- **Code Coverage**: Target >80%
- **Documentation Coverage**: 100% of public APIs
- **Build Success Rate**: >95%
- **Issue Resolution Time**: <7 days average
- **Code Review Turnaround**: <48 hours
- **Release Frequency**: Monthly minor releases

### Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [Semantic Versioning](https://semver.org/)

---

**Last Updated**: January 2024  
**Document Version**: 1.0  
**Maintainer**: Project Team
