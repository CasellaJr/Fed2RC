<div align="center">

# Fed2RC: Federated Rocket Kernels and Ridge Classifier for Time Series Classification
Bruno Casella, Samuele Fonio, Lorenzo Sciandra, Claudio Gallicchio, Marco Aldinucci, Mirko Polato and Roberto Esposito

[![Conference](https://img.shields.io/badge/ECAI-2025-red)](add_link)

</div>

# Overview

This repository contains the code to run and reproduce the experiments of the Fed2RC algorithm and baselines.

Please cite as:

```bibtex
@inproceedings{casella2025fed2rc,
  author  = {Casella, Bruno and Fonio, Samuele and Sciandra, Lorenzo and Gallicchio, Claudio and Aldinucci, Marco and Polato, Mirko and Esposito, Roberto},
  title   = {Fed2RC: Federated Rocket Kernels and Ridge Classifier for Time Series Classification,
  booktitle    = {28th European Conference on Artificial Intelligence, {ECAI} 2025, Bologna, Italy, October 25-30, 2025},
  year         = {2025},
  doi          = {},
  url = {}
}
```

# Abstract
Time series classification is a pivotal task in modern machine learning, with widespread applications in fields such as healthcare, finance, and cybersecurity. While deep learning methods dominate recent developments, their resource demands and privacy limitations hinder deployment on low-power and decentralized environments. To address these challenges, we introduce Fed2RC, a fully federated and gradient-free approach that integrates the efficiency of Rocket-based feature extraction with the robustness of ridge regression in a privacy-preserving setting. Fed2RC builds upon two key ideas: (i) federated selection and aggregation of high-performing random convolution kernels, and (ii) incremental and communication-efficient updates of ridge classifier parameters using closed-form solutions. Additionally, we propose a novel federated protocol for selecting the global ridge regularization parameter $\lambda$, and show how to improve the communication efficiency by matrix factorization techniques. Extensive experiments on the UCR benchmark demonstrate that Fed2RC achieves state-of-the-art results with a fraction of the computation and communication costs.

## Usage
- Clone this repo.
- Download and unzip the UCRArchive Time Series Classification dataset: [UCRArchive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/), and place it in the same directory of the repository.
- Install the requirements. 
- Run experiments (Fed2RC and baselines) with the [fluke](https://makgyver.github.io/fluke/first_run.html) framework for FL. If you want to run Fed2RC experiments without fluke, please see `code/fed2rc_no_fluke.py`.

## Results
The results directory reports the accuracy and F1-score metrics for all datasets of the UCR Archive. Results are reported for both Fed2RC and baselines and expressed in terms of accuracy and f1-score. For Fed2RC, are reported also additional results on communication costs.


## Contributors
* Bruno Casella <bruno.casella@unito.it>  

* Samuele Fonio <samuele.fonio@unito.it>

* Lorenzo Sciandra <lorenzo.sciandra@unito.it>

* Claudio Gallicchio <claudio.gallicchio@unipi.it>

* Marco Aldinucci <marco.aldinucci@unito.it>

* Mirko Polato <mirko.polato@unito.it>

* Roberto Esposito <roberto.esposito@unito.it>
