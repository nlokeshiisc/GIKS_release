# GIKS

This is the code for our paper: Continuous Treatment Effect Estimation using Gradient Interpolation and Kernel Smoothing

# Abstract

We address the Individualized continuous treatment effect (ICTE) estimation problem where we predict the effect of any continuous valued treatment on an individual using observational data. The main challenge in this estimation task is the potential confounding of treatment assignment with features of individuals in the observed data, whereas during inference ICTE requires prediction on independently sampled treatments. In contrast to prior work that relied on regularizers or unstable GAN training, we advocate the direct approach of augmenting training individuals with independently sampled treatments and inferred counterfactual outcomes. We infer counterfactual outcomes using a two-pronged strategy: a Gradient Interpolation for close-to-observed treatments, and a Gaussian Process based Kernel Smoothing which allows us to down weigh high variance inferences. We evaluate our method on five benchmarks and show that our method outperforms six state-of-the-art methods on the counterfactual estimation error. We explain the superior performance of our method by showing that (1) our inferred counterfactual responses are more accurate, and (2) adding them to the training data reduces the correlation introduced by confounders. Our proposed method is model-agnostic and we show that it improves ICTE accuracy of several existing models.

# Datasets

We experimented with $5$ datasets -- IHDP, NEWS, TCGA(0-2) with three different treatment types.

We have provided the datasets for `IHDP` and `NEWS` in this repository.
`TCGA` is a large dataset, it must be downloaded from [here](https://drive.google.com/file/d/1P-smWytRNuQFjqR403IkJb17CXU6JOM7/view) and put inside `dataset/tcga` as `tcga.p`

# Running GIKS

To run GIKS, we have provided two scripts namely `script_best_factual.py`, `script_best_giks.py`

Please run the scripts in the following order, **sequentially**.

For example, to run the experiments for `IHDP`, do

```
python script_best_factual.py --ds ihdp
python script_best_giks.py --ds ihdp
```

We run factual training for `200` epochs, and then initiate GIKS for `200` epochs. This is to ensure that the model fits on factual sample first and then starts interpolating counterfactuals. Each of these runs is moderated by our `early_stopping` script that chooses models based on the best achievable factual error on the validation dataset. All the baselines are run for `400` epochs. 

# Running Baselines

The code for the following baselines:

	1. TARNet
	2. DRNet
	3. VCNet+TR
	4. VCNet+HSIC

is available in this repository. 

The code for `SciGAN` can be found in: https://github.com/ioanabica/SCIGAN

The code for `TransTEE` can be found in: https://github.com/hlzhang109/TransTEE/tree/main/Dosage

# Evaluating CF Error

The code for evaluating `CF Error` is available in the following scripts

1. `mise_ihdp.py` for the IHDP dataset
2. `mise_news.py` for the NEWS dataset
3. `mise_tcga.py`for the TCGA dataset

# Paper plots:

We have provided the results that we obtained during our runs in the notebook: `all_results_pvalue.ipynb`