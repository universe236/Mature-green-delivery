# Mature green delivery

Model code for the paper *Mature green overpromises how fast new urban greening cools*.

The code follows newly greened urban sites through time and compares their cooling with matched established green
space. It covers local cooling, formation-mode classification, the mature-green benchmark, annual cooling
trajectories, late and cumulative delivery, and city-level uncertainty. Data preparation is not included; the models
start from event-level temperature and vegetation records.

## Contents

- `greendelivery/cooling.py`: local cooling of a greening event (Equation 1)
- `greendelivery/formation.py`: planted or spontaneous, and vegetation type
- `greendelivery/benchmark.py`: mature-green benchmark (Equation 2)
- `greendelivery/trajectory.py`: onset, annual delivery, year-1 share and saturating fit (Equations 3, 4, S1)
- `greendelivery/delivery.py`: late and cumulative delivery and overestimation (Equation 5)
- `greendelivery/inference.py`: city bootstrap and within-city regression
- `greendelivery/contrasts.py`: tree-grass comparison under a common sample
- `examples/quickstart.py`: runs all steps on synthetic data

## Getting started

```bash
pip install -r requirements.txt
python examples/quickstart.py
```

## Citation

Please cite the paper when using this code.
