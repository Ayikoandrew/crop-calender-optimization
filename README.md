# Crop Calendar Optimization Using Deep Reinforcement Learning

A machine learning framework for optimizing multi-seasonal planting decisions in western Uganda under climate variability.

## Overview

This project develops a Deep Reinforcement Learning (DRL) system for optimizing crop planting calendars in Uganda's bimodal rainfall system. The framework addresses the challenge of determining optimal planting dates when traditional calendars, developed over generations based on indigenous knowledge, have become increasingly unreliable due to climate variability.

The system integrates:
- **Variational Autoencoder (VAE)** for generating synthetic rainfall scenarios
- **LSTM-based World Model** for short-term rainfall forecasting
- **Reinforcement Learning Environment** for training adaptive planting policies

## Motivation

Uganda's agricultural system is shaped by its bimodal rainfall distribution, characterized by two distinct rainy seasons: the first season (*masika*, March-June) and the second season (*vuli*, August-December). However, observational studies spanning the past three decades have documented significant alterations in these established patterns, with onset dates shifting by up to 30 days from historical norms (Nsubuga & Rautenbach, 2018).

Climate variability accounts for yield reductions of 15-20% in maize production, with losses exceeding 50% during severe drought years (Kansiime et al., 2013). This project applies DRL to learn adaptive planting strategies that can respond to evolving weather conditions, addressing limitations of both static agrometeorological advisories and traditional knowledge systems.

## Technical Approach

### Rainfall World Model

An LSTM-based probabilistic forecasting model trained on 37 years of satellite-derived precipitation data. The model predicts rainfall distributions for 14-day horizons, enabling the RL agent to anticipate near-term conditions.

### Variational Autoencoder

A convolutional VAE learns a compact latent representation of seasonal rainfall patterns. This enables:
- Generation of unlimited synthetic training scenarios
- Interpolation between climate extremes (drought to wet)
- Simulation of climate shift scenarios for robust policy training

### RL Environment

A Gymnasium-compatible environment (`CropCalendarEnv`) that simulates:
- Crop growth dynamics using a water balance model based on FAO AquaCrop principles
- Soil moisture tracking and water stress accumulation
- Multi-variety crop options (short, medium, long cycle)
- Yield computation based on cumulative stress during critical growth stages

## Project Structure

```
crop-calendar-optimization/
├── data/
│   ├── raw/                    # Original dataset (Diem, 2021)
│   │   ├── RainfallData/       # Daily rainfall CSVs (5 regions)
│   │   └── Regions/            # Shapefile boundaries
│   └── processed/              # Engineered features
├── docs/
│   └── literature_review.md    # Background and references
├── models/
│   ├── crop_calendar_env.py    # RL environment
│   ├── rainfall_world_model.py # LSTM forecaster
│   ├── rainfall_lstm_best.pt   # Trained LSTM weights
│   └── rainfall_vae.pt         # Trained VAE weights
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_rainfall_forecasting_lstm.ipynb
│   ├── 04_rl_environment.ipynb
│   └── 05_rainfall_vae.ipynb
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/crop-calendar-optimization.git
cd crop-calendar-optimization

# Create virtual environment
uv venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch
- Gymnasium 
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn, Plotly

## Usage

### 1. Data Processing

Run the exploration and feature engineering notebooks to prepare the dataset:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
```

### 2. Train World Model

Train the LSTM rainfall forecaster:

```bash
jupyter notebook notebooks/03_rainfall_forecasting_lstm.ipynb
```

### 3. Train VAE

Train the Variational Autoencoder for scenario generation:

```bash
jupyter notebook notebooks/05_rainfall_vae.ipynb
```

### 4. RL Environment

Test and train agents using the crop calendar environment:

```python
from models.crop_calendar_env import CropCalendarEnv
from models.rainfall_world_model import RainfallWorldModel

# Initialize environment
world_model = RainfallWorldModel('models/rainfall_lstm_best.pt')
env = CropCalendarEnv(weather_simulator, region='Region1')

# Run episode
obs, info = env.reset()
for _ in range(150):
    action = env.action_space.sample()  # Replace with trained policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
```

## Dataset

This project uses the "Daily rainfall totals for western Uganda, 1983-2019" dataset:

**Citation:**
> Diem, Jeremy (2021), "Daily rainfall totals for western Uganda, 1983-2019", Mendeley Data, V2, doi: [10.17632/ppxktx233y.2](https://data.mendeley.com/datasets/ppxktx233y/2)

**Associated Publication:**
> Diem, J. E., Sung, H. S., Konecky, B. L., Palace, M. W., Salerno, J., and Hartter, J. (2019). Rainfall characteristics and trends—and the role of Congo westerlies—in the western Uganda transition zone of equatorial Africa from 1983 to 2017. *Journal of Geophysical Research: Atmospheres*, 124, 10712-10729. https://doi.org/10.1029/2019JD031243

The dataset includes daily precipitation estimates from CHIRPS and TAMSAT satellite products across five rainfall regions, totaling approximately 67,525 daily observations.

## Key References

Gautron, R., Maillard, O., Preux, P., Corbeels, M., & Tittonell, P. (2022). Reinforcement learning for crop management support: Review, prospects and challenges. *Computers and Electronics in Agriculture*, 200, 107182.

Nsubuga, F. W., & Rautenbach, H. (2018). Climate change and variability: a review of what is known and ought to be known for Uganda. *International Journal of Climate Change Strategies and Management*, 10(5), 752-771.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

## License

This project is licensed under the MIT License. The rainfall dataset is available under CC BY 4.0.

## Acknowledgments

- Jeremy Diem and Georgia State University for the rainfall dataset
- CGIAR Climate Services for conceptual guidance on agricultural decision support systems