

# Thermal Power Plant Performance Modeling with Deep Learning

This project presents a comparative analysis of deep learning architectures—Deep Neural Networks (DNN), Convolutional Neural Networks (CNN), and Recurrent Neural Networks (LSTM)—for multivariate regression modeling of key output parameters in a thermal power plant based on its operational data.

---

## Project Overview

Thermal power plants exhibit complex, nonlinear, and time-dependent behavior. Conventional physics-based or linear models often fail to accurately capture these dynamics. In this work, we explore deep learning techniques to model and predict:

* **Turbine power output**
* **Desuperheater feedwater flow**
* **Temperature and pressure at multiple control points**

---

## Models Evaluated

* **DNN**: Fully connected networks with up to three hidden layers.
* **CNN (ResNet-inspired)**: 1D convolutional architecture tailored for time-series data.
* **RNN (LSTM)**: Long Short-Term Memory network for modeling temporal dependencies.

---

## Dataset

* **Source**: [Kaggle - Power Plant Data Optimization](https://www.kaggle.com/datasets/deepakburi062/power-plant-data-optimization-problem)
* **Size**: 17,280 rows × 116 columns
* **Preprocessing**:

  * Column renaming to match power plant terminology
  * Outlier handling
  * Feature normalization using `MinMaxScaler`
  * 70/30 train-test split

---

## Model Performance

| Model                 | Train R²  | Test R²   | Test MSE   |
| --------------------- | --------- | --------- | ---------- |
| DNN (3-layer sigmoid) | 0.89      | 0.86      | 0.1344     |
| CNN (ResNet)          | 0.985     | 0.916     | 0.0800     |
| **LSTM (RNN)**        | **0.957** | **0.941** | **0.0600** |

> The LSTM-based RNN demonstrated the best generalization performance on the test set.

---

## Repository Structure

```
thermal-plant-forecasting/
├── thermal_powerplant_prediction_DL.ipynb  # Main notebook
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
└── data/                                   # Data folder (dataset not included)
```

---

## Getting Started

1. Clone the repository or download the notebook.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/deepakburi062/power-plant-data-optimization-problem).
4. Run the notebook `thermal_powerplant_prediction_DL.ipynb`.

---

## Technologies Used

* Python
* NumPy, Pandas
* Scikit-learn
* TensorFlow / Keras
* Matplotlib

---

## Author

**Partha Sarathi Dutta**
PhD Researcher, University of Illinois Chicago
📧 Email: [pdutta3@uic.edu](mailto:pdutta3@uic.edu)
🔗 GitHub: [@ParthaSarathiDutta](https://github.com/ParthaSarathiDutta)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
