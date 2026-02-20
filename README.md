# CosmicWeb

CosmicWeb is a pipeline for reconstructing the COSMOS-Web large-scale structure using weighted kernel density estimation (KDE). The workflow produces per-slice density maps, per-galaxy overdensities, and files that can be used for visualization and analysis. For a full detail of the procedure, you can read Hatamnia et al. 2025.


![Cosmic Web Visualization](Assets/cosmos_web_slice.png)

---

## Data Release

The reconstructed density maps and per-galaxy overdensity catalogs are now publicly available via the official COSMOS-Web data release portal:

https://cosmos2025.iap.fr/lss.html

A Jupyter notebook demonstrating how to load and use the released catalog is provided in:  
`Codes/6-Process_Available_Data.ipynb`

---

## Pipeline

### 1. Galaxy Weight Estimation
Photometric-redshifts are used to compute statistical weights for each galaxy.  
**File:** `1-Weight_Estimations.ipynb`

### 2. Bandwidth Estimation
Smoothing bandwidths are derived for each redshift slice.  
**File:** `2-Bandwidth.ipynb`

### 3. Density Map Construction
Weighted KDE density maps are generated for all slices across the COSMOS-Web field.  
**File:** `3-Density_Construction.py`

### 4. Per-Galaxy Density Extraction
Each galaxy’s overdensity is computed.  
**File:** `4-Galaxy_Density.ipynb`

### 5. Using the Generated Products
Instructions and examples showing how to load density maps and visualize them.  
**File:** `5-Plotting_Slices.ipynb`

### 6. Using the Public Catalog
Demonstrates how to load and work with the publicly released COSMOS-Web LSS catalog.  
**File:** `Codes/6-Process_Available_Data.ipynb`


---

## Citing This Work

If you use any part of this pipeline, please cite:

- Hatamnia et al. 2025 [(ADS)](https://ui.adsabs.harvard.edu/abs/2025arXiv251110727H/abstract)  
- Taamoli et al. 2024 [(ADS)](https://ui.adsabs.harvard.edu/abs/2024ApJ...966...18T/abstract)  
- Chartab et al. 2020 [(ADS)](https://ui.adsabs.harvard.edu/abs/2020ApJ...890....7C/abstract)  
- Darvish et al. 2015 [(ADS)](https://ui.adsabs.harvard.edu/abs/2015ApJ...805..121D/abstract)

---

## AR Demonstration

An augmented-reality visualization of COSMOS-Web has been developed in Unity.




https://github.com/user-attachments/assets/f8176ba1-4457-4a67-8840-c80d784037c3



