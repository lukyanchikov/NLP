# Sentiment Analysis
# About

This repository contains a sentiment analysis solution template. `NLP.py` implements training/testing and (batch) predicting. `app.py` implements a Flask endpoint definition for sending sentiment prediction requests corresponding to single reviews. `curl.txt` contains examples of sentiment predictions requests.

# Acknowledgements

The following repository has been an accelerator: [https://github.com/haldersourav/imdb-nlp-sentiment](https://github.com/haldersourav/imdb-nlp-sentiment)

# Prerequisites

The following components are recommended prerequisites:

- [Python 3.6.7](https://www.python.org/downloads/release/python-367/)

# Installation

(1) Clone/download/unzip this repository in the working folder of your choice (you must have read, write and execute rights in it).

(2) Using the OS terminal, navigate to the working folder and type: `pip install -r requirements.txt`

(3) Unzip into the working folder the `IMDB-trainvalidate.zip` archive

(4) Unzip into the `bert-base-cased` subfolder of the working folder the `pytorch_model.zip` archive (represented by multiple volumes)

(5) Unzip into the `bert-base-uncased` subfolder of the working folder the `pytorch_model.zip` archive (represented by multiple volumes)

(6) Edit the `path=r'C:\Users\slukyanc\Desktop\Machine Learning with Python Labs'` line by substituting the OS path to your working folder

# Running

(1) Edit the run parameter values in the `#Run by defining the parameter values:` section (scroll down or use Find to reach it)

(2) Use the `Run All` command in the `Cell` menu in Jupyter Notebook

(3) Review the outputs produced under the `#Run by defining the parameter values:` section (NB: the very last output - simulation of accuracy - may take 1-2 minutes to appear, the simulation progress is indicated by the `... out of 10 simulation runs completed.` messages displayed in the `#Simulate accuracy:` section)