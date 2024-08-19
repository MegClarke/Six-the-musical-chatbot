# GDPLabs AI/Machine Learning Projects

Welcome to the GDPLabs AI/Machine Learning Projects repository! This repository stores all my AI and machine learning projects for my internship at GDPLabs. Below is a brief overview of the current contents and future plans for the repository.

## Table of Contents

- Overview
- Installation
- Kaggle Titanic Competition
   - Environment Set-up
   - Running
   - Testing
- Six the Musical Chatbot


## Overview

This repository is a collection of my AI and machine learning projects developed during my internship at GDPLabs. The projects are intended to showcase various skills and techniques in the field of AI/ML, including data exploration, model training, and deployment of models for practical applications.


## Installation

To set up the repository and install the necessary dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/GDP-ADMIN/ai-internship-meagan.git


## Kaggle Titanic Competition

This project involves exploring and attempting to solve the Kaggle Titanic competition. The goal is to predict the survival of passengers aboard the Titanic based on various features such as age, gender, class, etc. The project includes data preprocessing, feature engineering, model training, and evaluation using different machine learning algorithms.

### Environment Set-up
```bash
   cd titanic-competition
   conda env create -f environment.yml
   conda activate glairenv
```

### Running
```bash
   cd titanic-competition
   python final_model.py input/train.csv --model decision_trees --threshold 0.55
```

### Testing
```bash
   cd titanic-competition
   pytest test_suite.py
```


### Six the Musical Chatbot

A RAG-based chatbot project focused on "Six the Musical". The chatbot is designed to answer questions and provide information about the musical. This project involves large language models (LLMs), vector stores, text embedding, and other RAG techniques. See chatbot-project/README.md for more detailed set-up and build intructions.
