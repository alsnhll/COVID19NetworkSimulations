# COVID19NetworkSimulations
Repository for stochastic network simulations of COVID19 dynamics

## Description

This package includes custom-written code to simulate stochastic network-based compartmental models of infection dynamics for large population sizes extremely quickly, using Python's JAX package. 

The code was used to generate the simulations in "Dynamics of COVID-19 under social distancing measures are driven by transmission network structure" by Anjalika Nande, Ben Adlam, Justin Sheen, Michael Z. Levy, and Alison L. Hill, now published in [PLoS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008684). This code base was also used for a second paper, "The effect of eviction moratoria on the transmission of SARS-CoV-2", which is available on the [medrix](https://www.medrxiv.org/content/10.1101/2020.10.27.20220897v2) and is in press at Nature Communications.

We direct users to the iPython notebook Examples_UniformRandomNetwork.ipynb as a starting point. This is a tutorial of sorts where we show how to use the code, describe the different waiting time distributions and give examples of the different plotting functions and interventions that can be simulated. All of the iPython notebooks in this repository were run on Google Colab using GPUs. A stochastic simulation of ~1 million indiviuals for 1 year takes about 1 minute to run on a single GPU. 

## Contents
* networks.py : This file contains functions for the network creation code for both the two and five layer networks.
* seir.py : Contains functions to simulate stochastic SEIR COVID-19 dynamics on a weighted, directed graph. The code is implemented in JAX, a framework for generating high-performance code optimized to run on GPUs. This file also contains different plotting functions and functions that calculate times to peak infection levels and individual risks of infection.
* Wellmixed_sim.ipynb : We used this iPython notebook to analyse COVID-19 dynamics pre and post intervention in well-mixed populations. This was used to generate Figure 2 in the main text.
* Two_layer_network_sim.ipynb : This iPython notebook was used to analyse the effects of household transmission using the two layer network. Figures 3,4 and 6A in the main text were generated from this file.
* Five_layer_network_sim.ipynb : This iPython notebook was used to analyse the effects of clustered adoption of social distancing measures using the five layer network. Figures 5 and 6B in the main text were generated using this file.
* Relaxing_intervention.ipynb : We used this iPython notebook to analyse the effects of relaxing intervention by merging households. This uses the two layer network and was used to generate Figure 7 in the main text.
* CitiesData.xlsx : Contains the COVID-19 count data used to generate Figure 1 in the main text.
