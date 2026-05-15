# neural-vol-calibrator

![Status](https://img.shields.io/badge/status-in%20progress-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

Neural network calibration engine for **Heston & SABR** stochastic volatility models.  
Replaces classical numerical optimizers with a feedforward network trained on synthetic implied volatility surfaces — achieving **~1000x speedup** over standard Levenberg-Marquardt calibration.

---

## Motivation

Calibrating stochastic volatility models to market option prices is a core task in quantitative finance. Classical methods (scipy, QuantLib) solve a non-linear least squares problem at each calibration — taking 2–5 seconds per surface. This project trains a neural network to learn the surface-to-parameter mapping directly, reducing inference to **under 1ms**.

---

## Planned Features

- [x] Project setup
- [x] Heston Monte Carlo pricer & parameter sampler
- [x] SABR Monte Carlo pricer & parameter sampler
- [x] Implied volatility surface builder
- [x] Neural calibrator (PyTorch feedforward network)
- [x] Classical benchmark (Scipy Levenberg-Marquardt)
- [x] Evaluation: RMSE/MAE per parameter
- [x] 3D Plotly vol surface visualizations (true vs predicted)
- [x] Speed benchmark: neural vs classical
- [ ] Jupyter notebook demo

---

## Project Structure

```
neural-vol-calibrator/
├── data/               # Surface generation scripts
│   ├── heston.py       # Heston MC pricer & sampler
│   ├── sabr.py         # SABR MC pricer & sampler
│   └── surface.py      # Implied vol surface builder
├── models/             # Neural network architecture
│   ├── network.py      # Feedforward calibrator
│   └── train.py        # Training loop
├── calibration/        # Classical benchmark
│   └── lm_optimizer.py # Levenberg-Marquardt baseline
├── evaluation/         # Metrics & visualizations
│   ├── metrics.py      # RMSE, MAE per parameter
│   └── plots.py        # Plotly surface plots
├── notebooks/          # Demo Jupyter notebooks
├── requirements.txt
└── README.md
```

---

## Stack

| Component | Library |
|---|---|
| Monte Carlo pricing | Python, NumPy |
| Neural calibrator | PyTorch |
| Classical baseline | SciPy |
| Visualization | Plotly |
| Notebooks | Jupyter |

---

## Author

**Abdellah Kahlaoui** — Master of Applied Mathematics, FST Settat  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/kahabdu1808)
[![GitHub](https://img.shields.io/badge/GitHub-AbdellahKah-181717?style=flat-square&logo=github)](https://github.com/AbdellahKah)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
