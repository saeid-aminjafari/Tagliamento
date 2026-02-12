# National-Scale, Satellite-Based Monitoring System for Sweden’s Surface Water Dynamics

## Research Question
**The scientific overarching question Q1: How do the variations in waterstorage and water flow correlate with climate change indicators, anthropogenic activities, and socioeconomicand ecosystem changes?**

To answer Q1, I formulate four methodological questions:
How can integrated satellite observations from ICESat-2 and SWOT improve our understanding of spatialand temporal variations in water levels and extents across Sweden's diverse aquatic ecosystems?
How will the integration of NISAR's L-band SAR data enhance the detection and quantification of waterextent in Sweden’s densely vegetated wetlands and flooded forests compared to Ka-band and existing C-band SAR data?
How will machine learning and statistical methodologies can identify the drivers and implications ofwater storage and flow changes?
What are the methodological advancements and accuracies gained from employing machine learning algorithms in processing satellite data and hydrological modelling in predicting future water storage and flow dynamics under various climate scenarios in Sweden?

## Aim
In-depth Analysis of Environmental Impact: Analyse what are the factorsof the changes in water storage and water flow and how they affect ecosystem services, and provide actionable insights for policy and decision-making

## Objectives
- Comprehensive Monitoring of Water Storage and Water Flow: Extend and refine the monitoring of waterlevels, water extent, and flow dynamics across Sweden's surface waters, utilizing cutting-edge satellitetechnology and data integration from ICESat-2, SWOT, and NISAR missions.
- Removing the spatial pattern of post-glacial rebound from the measured water level data with the help ofInSAR.
- Integration of Advanced Data Processing Techniques: Employ machine learning alongside conventionalmethods to process and analyse satellite data, enabling a more nuanced understanding of water dynamics.
- Enhanced Hydrological Modelling: Improve and validate hydrological models, such as HYPE, byincorporating satellite data and machine learning to better predict future trends and respond toenvironmental changes.

## Output
- A dataset of processed lake water levels from 2018 to the present and water extent from 2023 inNETCDF format that will be stored in the Database of the Bolin Centre for Climate Research (freelyavailable).
- Paper #1: “Data for Swedish lake water levels and their changes from 2018” - potential journal: EarthSystem Science Data (Copernicus Publications)
- A dataset of surface water extent covering all wetlands and lakes that will be stored in the Database ofthe Bolin Centre for Climate Research (freely available).
- Paper #2: “Surface Water Extent in Nordic Wetlands Using NISAR L-band Data and Deep Learning”- potential journal: Remote Sensing of Environment (Elsevier)
- Paper #3: “Spatial and Temporal Dynamics of Water Storage and Water Flow in Swedish WaterBodies” - potential journal: Water Resources Research (American Geophysical Union)
- A new update to the HYPE hydrological model that can be used by the SMHI
- Paper #4: “Integration of Satellite Altimetry and Long-Wavelength SAR data with HYPEHydrological model” - potential journal: Journal of Hydrology (Elsevier)

---

## 1. Context: ?


---

## 2. Why Does This Matter in ?


---

## 3. Literature Foundations



---

### [1] Chang et al. (2017): Nationwide Railway Monitoring Using Model Library + Hypothesis Testing


#### Why it’s relevant to our case:


#### Potential Modifications:

---

### [2] Schlögl et al. (2021): Clean Decomposition + Attribution + Clustering



#### Why it’s relevant:

#### Potential Modifications:


---

### [3] Prasanthi et al. (2025): Deep Embedded Clustering of Trend Classes



---

### [4] Kuzu et al. (2023): Building Displacement Anomalies from InSAR Time-Series (EGMS)



#### Data Preparation


#### Preprocessing


#### Model: LSTM Autoencoder

#### Loss Function: Soft-DTW

#### Anomaly Detection


#### Evaluation

#### Strengths:

#### Weaknesses / Risks:


#### In our case (railway monitoring):


---

### [5] Bai et al. (2025): KCC-LSTM — Cluster-Predict Framework for Deformation Series



#### Main Idea


#### Model Steps:


#### Relevance to our study:


---

## 4. Implementation Options



---

### Idea #1: Combine Chang et al. [1] + Schlögl et al. [2]

#### Goal

#### Step-by-Step Framework

##### Step 1: Data Preparation


##### Step 2: Time Series Modeling


##### Step 3: Attribution Analysis



##### Step 4: Comparative Analysis



##### Step 5: Interpretation



##### Output


---

### Idea #2: Modify Kuzu et al. (2023) — Ground Motion Anomaly Detection (Soft-DTW + DEM, Post-hydro Analysis)

#### Step 0 — Data Preparation



#### Step 1 — Input Preparation



#### Step 2 — Model Design & Training



#### Step 3 — Anomaly Detection & Clustering



#### Step 4 — Hydroclimatic & NDVI Correlation (Post-hoc)



#### Step 5 — Validation & Communication



---

### Idea #3: Combine Kuzu et al. [4] + Bai et al. [5] — KCC + LSTM Pipeline on EGMS


---

## 5. References

[1] Chang, L., Dollevoet, R. P. B. J., & Hanssen, R. F. (2017). Nationwide Railway Monitoring Using Satellite SAR Interferometry. *IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.*, 10(2), 596–604. https://doi.org/10.1109/JSTARS.2016.2584783

[2] Schlögl, M., Widhalm, B., & Avian, M. (2021). Comprehensive time-series analysis of bridge deformation using differential satellite radar interferometry based on Sentinel-1. *ISPRS J. Photogramm. Remote Sens.*, 172, 132–146. https://doi.org/10.1016/j.isprsjprs.2020.12.001

[3] Prasanthi, L., Krishnan, S. B., Prasad, K. V., & Chakrabarti, P. (2025). A Deep Embedded Clustering Approach for Detecting Trend Class using Time-Series Sensor Data. *Knowl.-Based Syst.*, 113609. https://doi.org/10.1016/j.knosys.2025.113609

[4] Kuzu, R. S., et al. (2023). Automatic Detection of Building Displacements Through Unsupervised Learning From InSAR Data. *IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.*, 16, 6931–6947. https://doi.org/10.1109/JSTARS.2023.3297267

[5] Bai, Z., Shen, C., Wang, Y., Lin, Y., Li, Y., & Shen, W. (2025). Bridge Deformation Prediction Using KCC-LSTM With InSAR Time Series Data. *IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.*, 18, 9582–9592. https://doi.org/10.1109/JSTARS.2025.3552665
