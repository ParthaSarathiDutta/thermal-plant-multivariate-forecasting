

# Thermal Power Plant Performance Modeling with Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-yellow)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://kaggle.com/your-link)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Compatible-yellow)](https://huggingface.co/spaces)


This repository presents a deep learning-based approach to model and predict the performance of a thermal power plant using historical operational data. Multiple neural architectures—including Deep Neural Networks (DNN), Convolutional Neural Networks (CNN), and Long Short-Term Memory (LSTM) networks—are evaluated to forecast key output variables.

---

## Overview

Thermal power plants operate with nonlinear and time-varying dynamics. Traditional physical modeling techniques often fall short in capturing this complexity. This project applies deep learning models to predict critical output variables such as:

* **Turbine power output**
* **Desuperheater feedwater flow**
* **Temperature and pressure at multiple control valves**

The goal is to improve accuracy in performance prediction, enabling smarter control and decision-making.

---

## Models Evaluated

* **DNN**: Fully connected feedforward networks (up to 3 hidden layers)
* **CNN (ResNet-inspired)**: 1D convolutional model adapted for time-series forecasting
* **LSTM (RNN)**: Captures sequential dependencies and temporal patterns

---

## Dataset

* **Source**: [Kaggle - Power Plant Data Optimization](https://www.kaggle.com/datasets/deepakburi062/power-plant-data-optimization-problem)
* **Included File**: `thermal_powerplant_prediction_DL.csv`
* **Observations**: 17,280 rows × 116 columns
* **Preprocessing Steps**:

  * Column renaming for domain clarity
  * Outlier handling
  * Normalization with `MinMaxScaler`
  * 70/30 train-test split

---

## Model Performance

| Model          | Train R²  | Test R²   | Test MSE   |
| -------------- | --------- | --------- | ---------- |
| DNN (3-layer)  | 0.89      | 0.86      | 0.1344     |
| CNN (ResNet)   | 0.985     | 0.916     | 0.0800     |
| **LSTM (RNN)** | **0.957** | **0.941** | **0.0600** |

> The LSTM-based model yielded the highest generalization performance.

---

## Repository Structure

```
thermal-plant-forecasting/
├── thermal_powerplant_prediction_DL.ipynb    # Main notebook with code and results
├── thermal_powerplant_prediction_DL.csv      # Dataset used in the project
├── thermal_powerplant_prediction_DL.pdf      # Final project report
├── requirements.txt                          # Python dependencies
├── LICENSE                                   # License file
└── README.md                                 # Project overview
```

---

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/ParthaSarathiDutta/thermal-plant-forecasting.git
   cd thermal-plant-forecasting
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Launch the notebook and execute step by step:

   ```bash
   jupyter notebook thermal_powerplant_prediction_DL.ipynb
   ```

---

## Tools & Libraries

* Python
* NumPy, Pandas
* Scikit-learn
* TensorFlow / Keras
* Matplotlib

---

## Author

**Partha Sarathi Dutta**
AI Researcher, Argonne National Laboratory 
Email: [pdutta3@uic.edu](mailto:pdutta3@uic.edu)
GitHub: [@ParthaSarathiDutta](https://github.com/ParthaSarathiDutta)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
