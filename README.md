# Predictability of electric vehicle charging: explaining extensive user behavior-specific heterogeneity

**Author:** Markus Kreft, Bits to Energy Lab, ETH Zurich: <mkreft@ethz.ch>

This repository contains the Python code for the [paper](https://www.sciencedirect.com/science/article/pii/S0306261924009279):

> *Kreft, M., Brudermueller, T., Fleisch, E., & Staake, T. (2024). Predictability of electric vehicle charging: explaining extensive user behavior-specific heterogeneity. Applied Energy, 370, 123544.*

If you make use of code in this repository, **please reference the following citation**:
```bibtex
@article{KREFT2024123544,
    title = {Predictability of electric vehicle charging: Explaining extensive user behavior-specific heterogeneity},
    journal = {Applied Energy},
    volume = {370},
    pages = {123544},
    year = {2024},
    issn = {0306-2619},
    doi = {https://doi.org/10.1016/j.apenergy.2024.123544},
    url = {https://www.sciencedirect.com/science/article/pii/S0306261924009279},
    author = {Markus Kreft and Tobias Brudermueller and Elgar Fleisch and Thorsten Staake},
}
```

---

### Abstract

Smart charging systems can reduce the stress on the power grid from electric vehicles by coordinating the charging process.
To meet user requirements, such systems need input on charging demand, i.e., departure time and desired state of charge.
Deriving these parameters through predictions based on past mobility patterns allows the inference of realistic values that offer flexibility by charging vehicles until they are actually needed for departure.
While previous studies have addressed the task of charging demand predictions, there is a lack of work investigating the heterogeneity of user behavior, which affects prediction performance.
In this work we predict the duration and energy of residential charging sessions using a dataset with 59,520 real-world measurements from 267 electric vehicles.
While replicating the results put forth in related work, we additionally find substantial differences in prediction performance between individual vehicles.
An in-depth analysis shows that vehicles that on average start charging later in the day can be predicted better than others.
Furthermore, we demonstrate how knowledge that a vehicles charges over night significantly increases prediction performance, reducing the mean absolute percentage error of plugged-in duration predictions from over 200% to 15%.
Based on these insights, we propose that residential smart charging systems should focus on predictions of overnight charging to determine charging demand.
These sessions are most relevant for smart charging as they offer most flexibility and need for coordinated charging and, as we show, they are also more predictable, increasing user acceptance.

---

### Installation

The code has been tested with Python 3.12.3.
```sh
pip install -r requirements.txt
```

---

### Data

The openly accessible dataset from the Electric Nation Project is available at [https://www.nationalgrid.co.uk/electric-nation-data](https://www.nationalgrid.co.uk/electric-nation-data).
Specifically, the [Crowd Charge Transactions](https://www.nationalgrid.co.uk/downloads-view/81646) and [Charger Install](https://www.nationalgrid.co.uk/downloads-view/81655) files need to be downloaded and placed in `./data/electric_nation_data`.

---

### Usage

Adapt `config.py` for desired experiment:
```python
CONSUMPTION = True | False
DAY = "" | "-nextday" | "-sameday"
```
Run `regression.py` to train and test models.
Run `evaluation.py` to generate plots.
