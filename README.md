
# Thermal Power Plant Performance Modeling with Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-yellow)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://www.kaggle.com/datasets/deepakburi062/power-plant-data-optimization-problem)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Compatible-yellow)](https://huggingface.co/spaces)

This repository explores the use of deep learning models to forecast key operational metrics of a thermal power plant. It compares Deep Neural Networks (DNN), Convolutional Neural Networks (CNN), and Long Short-Term Memory (LSTM) models for multivariate regression tasks.

---

## Overview

Thermal power plants exhibit nonlinear and time-dependent behaviors. This project applies data-driven modeling techniques to predict:

- Turbine power output
- Desuperheater feedwater flow
- Temperatures and pressures across multiple control points

The objective is to enhance predictive accuracy to support operational planning and control.

---

## Models

- **DNN**: Fully connected network with up to 3 hidden layers
- **CNN (ResNet-based)**: 1D convolutional model for time-series input
- **LSTM**: Recurrent neural network for temporal sequence modeling

---

## Dataset

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/deepakburi062/power-plant-data-optimization-problem)
- **File**: `thermal_powerplant_prediction_DL.csv`
- **Shape**: 17,280 samples × 116 features
- **Preprocessing**:
  - Feature renaming for interpretability
  - Outlier removal (z-score based)
  - Normalization using MinMaxScaler
  - 70:30 train-test split

---

## Results

| Model          | Train R²  | Test R²   | Test MSE   |
| -------------- | --------- | --------- | ---------- |
| DNN (3-layer)  | 0.89      | 0.86      | 0.1344     |
| CNN (ResNet)   | 0.985     | 0.916     | 0.0800     |
| **LSTM (RNN)** | **0.957** | **0.941** | **0.0600** |

---

## Repository Structure

```

thermal-plant-forecasting/
├── thermal_powerplant_prediction_DL.ipynb # Jupyter notebook with data preprocessing, modeling, training, and evaluation
├── thermal_powerplant_prediction_DL.csv # Dataset (17,280 × 116) used for model training and evaluation
├── thermal_powerplant_prediction_DL.pdf # Final project report summarizing methodology, experiments, and results
├── thermal_powerplant_presentation.pptx # Presentation slides for project overview and defense
├── thermal_powerplant_prediction.py # Python script version of the notebook for quick reference or automation
├── requirements.txt # Python package dependencies (TensorFlow, Pandas, Scikit-learn, etc.)
├── LICENSE # MIT License file
└── README.md # Project overview, instructions, and model summaries

```

---

## Running Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/ParthaSarathiDutta/thermal-plant-multivariate-forecasting.git
   cd thermal-plant-multivariate-forecasting
    ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook:

   ```bash
   jupyter notebook thermal_powerplant_prediction_DL.ipynb
   ```

---

## Dependencies

* Python 3.8+
* NumPy, Pandas
* Scikit-learn
* TensorFlow / Keras
* Matplotlib

---

## Contact

**Partha Sarathi Dutta**
AI Researcher, Argonne National Laboratory
📧 [pdutta3@uic.edu](mailto:pdutta3@uic.edu)
🔗 [GitHub](https://github.com/ParthaSarathiDutta)

---

## Acknowledgments

This project was developed as a collaborative effort by **Partha Sarathi Dutta**, **Safwan Mondal**, and **Subramanian Ramasamy** under the guidance of **Dr. Hadis Anahideh** at the University of Illinois Chicago.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.


