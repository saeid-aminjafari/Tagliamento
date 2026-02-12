#  Tagliamento River Basin: Landuse and flood hazard exposure change between 1950 and 2000

## Research Question
**The scientific question: How has landuse changed in Tagliamento River Basin and how has these changes exposed different landcovers to flood hazard?**

## Aim
In-depth Analysis of landuse and exposrue change.

## Objectives
- Investigating the landuse changed in every municipality through 1950-2000?
- Which landuse has changed more drastically? Which decade most of the change happened? Which municipality was the most affected.
- How different scenarios of flood hazard paly a role in landuse changes? How exposed are the most residential and agricultral lands to the flood?

## Output
- Two Python scripts on this GitHub repository (Tagliamento), **Tagliamento.py** and **Figures_neat.py** to calculate the areas under each landuse and municipality and create figures of the changes respectively. 
- A CSV table, containting the landuse area change in every municipality and every decade.
- Paper #1: “Title XXX”- potential journal: XXX

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
