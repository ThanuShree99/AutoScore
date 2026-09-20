# 🎯 AutoScore

**Automated Pilot Performance Assessment Using Machine Learning**

An intelligent system that converts raw 6-axis IMU (Inertial Measurement Unit) sensor data into objective, repeatable pilot performance scores using Linear Regression.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Core Concept](#core-concept)
- [Algorithm](#algorithm)
- [Why Linear Regression](#why-linear-regression)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Methodology](#methodology)
- [Future Enhancements](#future-enhancements)
- [Course Information](#course-information)
- [Author](#author)

---

## 🚀 Overview

AutoScore is an AI/ML project designed to eliminate subjectivity from pilot performance assessment. Instead of relying on instructor observation and simulator debriefs, it analyzes motion data from flight control inputs to produce a single, interpretable **10–100 performance score**.

**Key Innovation:** 6 engineered features from IMU data + Linear Regression = lightweight, deployable, interpretable pilot assessment

---

## 🎓 Problem Statement

### The Challenge
Traditional pilot-performance assessment today relies on:
- ❌ Instructor observation (subjective)
- ❌ Simulator debriefs (time-consuming)
- ❌ Hard to scale across multiple trainees

### The Solution
**AutoScore** tests whether raw motion data alone can produce an objective, repeatable score with zero human judgment calls.

### Objectives
✅ **Objective:** Turn 6-axis IMU signals into a single, interpretable 10–100 performance score  
✅ **Lightweight:** Just 6 compact engineered features computed from raw accelerometer + gyroscope data  
✅ **Deployable:** Sized to eventually run on embedded systems (ESP32-class hardware) — not a data-center model  

---

## 💡 Core Concept

### The Premise

```
Low-skilled pilots → More noise, larger corrections, oscillation
        ↓
     IMU captures this behavioral difference
        ↓
High-skilled pilots → Smoother control, better stability
```

A 6-axis IMU (accelerometer + gyroscope) captures behavioral differences between pilot skill levels. A regression model converts these patterns into one performance score.

### Key Insight
**Correction intensity is what separates skilled from unskilled pilots** — this is what AutoScore measures and scores.

---

## 🤖 Algorithm

### Linear Regression

AutoScore uses **Linear Regression** to map 6 engineered IMU features to pilot performance scores.

#### Mathematical Model
```
score = w₁·f₁ + w₂·f₂ + w₃·f₃ + w₄·f₄ + w₅·f₅ + w₆·f₆ + b

Where:
- w₁...w₆ = learned weights (one per feature)
- f₁...f₆ = standardized IMU-derived features
- b = bias term
```

#### How It Works
1. **Fit one weight** to each of the 6 IMU-derived features:
   - `rms_deviation` - Root mean square deviation of control inputs
   - `gyro_variability` - Gyroscope signal variance
   - `correction_magnitude` - Size of control corrections
   - `oscillation_freq` - Frequency of control oscillations
   - `settling_time` - Time to stabilize after disturbance
   - `energy_cost` - Total energy expended in corrections

2. **Learn weights** that minimize squared error between predicted and true scores across 3,200 training sessions

3. **Standardize features** (zero mean, unit variance) so no single feature's scale dominates

4. **Predict instantly** — one dot product operation per pilot session

---

## ⚖️ Why Linear Regression

### Linear Regression vs Random Forest Comparison

| Criterion | Linear Regression | Random Forest |
|-----------|-------------------|---------------|
| **Test MAE** | 1.667 ✓ lower | 1.896 |
| **Test R²** | 0.989 ✓ higher | 0.985 |
| **5-fold CV MAE (std)** | 1.611 (± 0.035) ✓ stabler | 1.842 (± 0.077) |
| **Model Size** | 6 weights + 1 bias | 300 trees, thousands of splits |
| **Interpretability** | Every weight has clear meaning | Feature importances only |
| **Embedded Feasibility** | Trivial — one dot product | Heavy — needs full forest |

### Verdict
**Linear Regression wins on accuracy, stability, and deployability all at once.**

Random Forest's extra complexity buys nothing here — it's both less accurate and less stable across folds.

---

## ✨ Features

### Dataset & Training
- **4,000 synthetic flight sessions** with 6 IMU-derived features
- **80:20 train/test split** for held-out evaluation
- **StandardScaler normalization** (zero mean, unit variance)
- **5-fold cross-validation** for robust performance estimation

### Model Evaluation
- **Multiple metrics:** MAE, RMSE, R² (not just MAE alone)
- **Cross-validation:** 5-fold on training data for reliable assessment
- **Residual analysis:** Checks for systematic bias in predictions
- **Feature importance:** Shows which IMU features drive predictions

### Multi-Pilot Testing
- Tests on 3 representative pilot skill levels:
  - **Beginner/Easy:** High noise, large corrections
  - **Intermediate/Medium:** Moderate control inputs
  - **Advanced/Hard:** Smooth, minimal corrections

### Code Quality
- **Modular design:** Refactored into named functions
- **CSV exports:** Results saved for report appendix
- **Reproducibility:** Fixed random seed for consistent results
- **Importable:** Can be imported as a module without re-running

---

## 📁 Project Structure

```
AutoScore/
├── AutoScore_Model_Training.ipynb    # Main Jupyter notebook with full pipeline
├── autoscore/                         # Python module (if separated)
│   ├── __init__.py
│   ├── data_generation.py            # Synthetic flight data generation
│   ├── feature_extraction.py         # IMU feature engineering
│   └── scoring.py                    # Score computation
├── results/                           # Output directory
│   ├── model_comparison.csv          # Performance metrics table
│   └── sample_predictions.csv        # Sample pilot predictions
├── README.md                          # This file
├── requirements.txt                  # Python dependencies
└── .gitignore                         # Git ignore rules
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/ThanuShree99/AutoScore.git
cd AutoScore

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
plotext>=4.0.0
```

---

## 💻 Usage

### Running the Complete Pipeline

```bash
# Option 1: Run from Jupyter Notebook
jupyter notebook AutoScore_Model_Training.ipynb

# Option 2: Run from command line (if module is set up)
python -m autoscore.train
```

### Pipeline Execution

**Step 1: Generate Data**
```
Generates 4,000 synthetic flight sessions
Extracts 6 IMU features from raw sensor data
```

**Step 2: Prepare Data**
```
80:20 train/test split
StandardScaler fit on training set
```

**Step 3: Define Models**
```
Initializes 4 models:
  - Linear Regression
  - Ridge Regression (L2 regularization)
  - SVR (Support Vector Regression)
  - Random Forest (300 trees)
```

**Step 4: Train & Evaluate**
```
5-fold cross-validation on training data
Model fitting and test set evaluation
Metric calculation (MAE, RMSE, R²)
```

**Step 5: Analyze Results**
```
Model comparison table
Feature importance visualization
Residual analysis plots
Sample pilot predictions
```

---

## 📊 Results

### Model Performance Summary

```
Model Comparison (sorted by Test MAE, lower is better):

            Model    CV_MAE_mean  Test_MAE  Test_RMSE  Test_R²
Linear Regression        1.611     1.667      2.081    0.989 ✓
Ridge Regression         1.611     1.667      2.081    0.989 ✓
              SVR        1.646     1.682      2.105    0.988
    Random Forest        1.842     1.896      2.376    0.985
```

### Key Findings

#### Accuracy
- **Linear Regression test MAE:** 1.667 (excellent precision)
- **R² Score:** 0.989 (explains 98.9% of score variance)

#### Stability
- **CV MAE mean:** 1.611
- **CV MAE std:** ± 0.035 (highly consistent across folds)

#### Feature Importance (Random Forest)
```
correction_magnitude    88.8%  ← Dominant feature
rms_deviation           6.2%
settling_time           2.8%
gyro_variability        1.4%
oscillation_freq        0.6%
energy_cost            0.2%
```

**Interpretation:** Correction intensity (correction_magnitude) is what separates skilled from unskilled pilots, confirming AutoScore's core premise.

#### Sample Predictions
```
Pilot Type      True Score  Linear Pred  Error
Beginner/Easy      50          50.00     0.00
Intermediate       65          65.01     0.01
Advanced/Hard      85          85.12     0.12
                                      avg: 0.04
```

---

## 🔬 Methodology

### Improvements Over Base Implementation

The original script had limitations suitable for academic work but not production-ready. This version adds:

#### 1. **Extended Metrics**
- Added R² and RMSE alongside MAE
- MAE alone doesn't show variance explained — R² is essential

#### 2. **Robust Validation**
- 5-fold cross-validation on training set
- Reported performance independent of lucky/unlucky splits
- Includes mean ± std for each metric

#### 3. **Feature Importance Analysis**
- Random Forest feature importances visualized
- Shows which IMU features actually drive scores
- Grounds interpretation in data

#### 4. **Residual Analysis**
- Checks if errors are randomly scattered (good) or biased (bad)
- Residuals vs. predicted score plot
- Detects systematic model failures

#### 5. **Multi-Pilot Testing**
- Tests on 3 representative pilot profiles, not just one
- Demonstrates generalization across skill levels
- Prevents overconfidence from lucky predictions

#### 6. **Reproducible Outputs**
- Results table exported to CSV (model_comparison.csv)
- Sample predictions exported (sample_predictions.csv)
- Ready for direct use in project report appendix

#### 7. **Production-Ready Code**
- Refactored into named functions
- `__main__` guard for module import
- Clear variable names and docstrings

---

## 🚀 Future Enhancements

### Planned Features

#### Phase 1: Deployment
- [ ] Embedded port to ESP32 (C implementation)
- [ ] Real-time inference from live IMU stream
- [ ] ONNX model export for cross-platform compatibility

#### Phase 2: Advanced Analytics
- [ ] Time-series decomposition of IMU signals
- [ ] Anomaly detection for unusual flight patterns
- [ ] Pilot profiling and skill progression tracking

#### Phase 3: Hardware Integration
- [ ] Direct MPU6050 / ICM20689 sensor integration
- [ ] Simulator device support (Saitek, Logitech)
- [ ] Wireless transmission to ground station

#### Phase 4: Machine Learning Improvements
- [ ] Fine-tuning with real pilot data
- [ ] Transfer learning from simulator to real aircraft
- [ ] Ensemble methods for robustness

---

## 📈 Performance Metrics Explained

### MAE (Mean Absolute Error)
Average absolute difference between predicted and true pilot scores.
- **Range:** 0–100 (score units)
- **Lower is better:** 1.667 MAE means average error of 1.67 score points
- **Interpretation:** On a 10–100 scale, this is less than 2% error

### R² (Coefficient of Determination)
Percentage of score variance explained by the model.
- **Range:** 0 to 1 (or 0–100%)
- **AutoScore R²:** 0.989 → Explains 98.9% of variation
- **Interpretation:** Almost all pilot score differences are captured

### RMSE (Root Mean Squared Error)
Emphasizes larger errors more than MAE.
- **AutoScore RMSE:** 2.081
- **Use case:** Penalizes outlier predictions more heavily
- **vs. MAE:** If all errors were equal to RMSE, total error would be same

### Cross-Validation (5-Fold)
Training data split 5 ways; each fold used once for validation, 4 times for training.
- **Purpose:** Detects overfitting, confirms stability
- **AutoScore CV MAE:** 1.611 (± 0.035) — very stable
- **vs. Single Split:** Random Forest's 5-fold std of ±0.077 shows instability

---

## 🎓 Course Information

**Course:** AI & Machine Learning (23EC402T)  
**Module:** Term Work Module  
**Institution:** [Your College/University]  
**Academic Year:** [Year]

### Learning Outcomes
- ✅ Feature engineering from raw sensor data
- ✅ Model selection & comparison using rigorous metrics
- ✅ Cross-validation & bias-variance tradeoff
- ✅ Interpretability vs. complexity in ML design
- ✅ Real-world problem formulation from abstract goals

---

## 💼 Author

**Thanu Shree N**
- 🎓 B.E Electronics & Communication Engineering, Velammal Engineering College
- 💼 Embedded Systems & AI/ML Enthusiast
- 🔗 [GitHub](https://github.com/ThanuShree99)
- 📧 [Email](mailto:thanushree.nagarajan@gmail.com)
- 💬 [LinkedIn](https://www.linkedin.com/in/thanushreen)

---

## 📝 License

This project is open source and available under the MIT License.  
See LICENSE file for details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ❓ FAQ

**Q: Why Linear Regression over deep learning?**  
A: For this problem, simple, interpretable, and deployable beats complex. Linear Regression achieves 98.9% R² with 6 weights vs. thousands of neural network parameters.

**Q: Can this run on ESP32?**  
A: Yes — that's the goal. One dot product operation is trivial for microcontrollers.

**Q: How much training data is needed?**  
A: This demo uses 3,200 sessions. Real deployment would benefit from 5,000+ with diverse pilot behaviors.

**Q: What if pilot behavior changes over time?**  
A: Model retraining on new sessions is recommended monthly or when assessment accuracy drifts.

---

## 📚 References

- Scikit-learn documentation: https://scikit-learn.org/
- Pandas user guide: https://pandas.pydata.org/docs/
- IMU sensor fusion techniques
- Machine learning best practices for embedded systems

---

## 🙏 Acknowledgments

- Velammal Engineering College for the opportunity
- Course instructor for guidance on model selection
- Scikit-learn community for excellent ML tools
- Research into pilot training assessment methodologies

---

<div align="center">

**Built with ⚡ for pilot training innovation**

If you find this project helpful, please ⭐ star the repository!

*Last updated: 2026 | Turning motion data into pilot insight*

</div>
