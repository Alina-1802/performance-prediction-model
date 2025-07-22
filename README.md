# Performance Prediction Model

A predictive model for estimating garment production team efficiency, built as part of a university course in mathematical modeling for business analytics. 

<img width="400" height="350" alt="act_pre_model1" src="https://github.com/user-attachments/assets/ef4610e8-0bff-49e9-bd05-a84a2ed744dc" />

## 📋 Description

This project aims to create a regression-based predictive model to estimate the productivity of garment factory teams.

The final solution helps identify factors affecting productivity, making it possible to implement targeted improvements in scheduling, motivation systems, and operational planning.

## 🏭 Business Context

Companies in the garment industry often struggle with workforce efficiency due to changing workloads, product styles, and unpredictable delays. 
This model can be used to:

- Predict future productivity and plan workforce accordingly
- Detect inefficiencies like frequent style changes or long idle times
- Design better incentive systems
- Support production scheduling and delivery planning

## 📊 Dataset

- Name: **Productivity Prediction of Garment Employees**  
- Source: [Kaggle Dataset](https://www.kaggle.com/datasets/ishadss/productivity-prediction-of-garment-employees/discussion?sort=undefined)  
- Records: 1197 (691 complete rows after cleaning)  
- Features: 15 (date, day, department, style changes, team size, overtime, idle time, incentives, etc.)

## 🔧 Data Preparation Workflow

- Data loading & inspection  
- Handling missing values using **KNN imputation**  
- Encoding categorical variables  
- Removing outliers  
- Feature scaling / normalization  
- Correlation analysis to select predictors

## 🤖 Models Used

Three **linear regression models** were trained and compared:

1. **Model 1**: all variables included  
2. **Model 2**: variables with correlation ≥ 0.1  
3. **Model 3**: variables with correlation ≥ 0.05 → *best results*

> 💡 **Note:** Despite being linear models, results were influenced by weak/non-linear correlations in the data.

## ✅ Key Results

- The relatively poor performance indicates that linear regression may not be well suited for this non-linear problem. Implementing more advanced models could lead to a substantial improvement in results.
- Best performance achieved by Model 3  
- Negative influence on productivity:  
  - `no_of_style_change`, `idle_men`, `smv`
- Positive influence on productivity:  
  - `incentive`, `targeted_productivity`
- Final model provides actionable insights for workforce management

## 🧠 Tools & Technologies

- Python
- pandas, scikit-learn, matplotlib, numpy, seaborn

## 🖼️ Sample Visuals

<img width="300" height="250" alt="act_pre_model3" src="https://github.com/user-attachments/assets/9f58890e-b527-481d-b6cd-f54901a65491" />
<img width="300" height="250" alt="act_pre_model2" src="https://github.com/user-attachments/assets/4a177eba-62c7-4401-88e0-bdf215ecab2e" />
<img width="300" height="250" alt="act_pre_model1" src="https://github.com/user-attachments/assets/28bac033-d818-41ef-802b-b1081c43ce0c" />

<img width="300" height="250" alt="res_model3" src="https://github.com/user-attachments/assets/a5880608-0b82-4d47-a759-f320c6ed4610" />
<img width="300" height="250" alt="res_model2" src="https://github.com/user-attachments/assets/b6cf5b34-08a4-4a79-b28d-a68fde2fa2e8" />
<img width="300" height="250" alt="res_model1" src="https://github.com/user-attachments/assets/a80879d6-0ab3-490e-a9af-9b0c006fd61b" />

