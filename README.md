# Privacy-Preserving Federated Learning System 
# for Distributed Healthcare Data

A federated learning system that enables multiple hospital institutions 
to collaboratively train machine learning models on distributed 
healthcare data without transferring raw patient records, integrating 
differential privacy and secure aggregation for patient confidentiality.

## Overview
Healthcare data breaches cost an average of 10.9 million USD per 
incident. Centralised machine learning exposes patient records to 
breaches and regulatory violations. This study designs and implements 
a privacy-preserving federated learning system using the FedAvg 
algorithm across distributed hospital datasets, with explicit attention 
to real-world messy data preprocessing, privacy integration, and 
rigorous performance evaluation against centralised approaches.

## System Architecture
- Federated Averaging (FedAvg) algorithm for decentralised training
- Local model training at each hospital node without raw data transfer
- Differential Privacy (DP-SGD) with gradient clipping and Gaussian
  noise injection to protect against membership inference attacks
- Secure aggregation of model updates across distributed institutions
- Horizontal federated learning across heterogeneous hospital datasets
  with shared feature space but distinct patient cohorts

## Privacy Mechanisms
- Differential Privacy with configurable epsilon privacy budget
- Gradient clipping and noise calibration per training round
- Secure model parameter aggregation
- Zero raw data transfer between participating institutions

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Communication efficiency across federated rounds
- Comparative evaluation against centralised baseline models
- Privacy-utility trade-off analysis across epsilon values

## Key Features
- End-to-end pipeline from messy hospital data to deployed model
- Robust preprocessing of missing values, inconsistencies and outliers
- Simulated multi-hospital distributed training environment
- Performance comparison between federated and centralised approaches
- Streamlit web interface for system demonstration

## Tech Stack
- Python
- TensorFlow Federated / PySyft
- Scikit-learn
- Streamlit
- Pandas, NumPy, Matplotlib

## How to Run
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Run the dashboard: streamlit run app.py

## Author
Caleb Markus
Data Scientist & Scholar
Taraba State University, Jalingo, Nigeria
