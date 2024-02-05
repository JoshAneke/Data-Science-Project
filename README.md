[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fBs3woei)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13054227&assignment_repo_type=AssignmentRepo)
# COSC 3337 - Data Science I 
## Data Analysis and Discovery ##

### Due Date: December 8, 2023 11:59 PM ###

## Overview
This assignment is an open-ended exploration of the COVID-19 dataset. 
The dataset and some initial ideas to get started are provided, but the 
analysis and discovery to be undertaken is up to each student/group. 
You may pick and choose which features/attributes to use in conducting 
your analyses. You may also derive more features/attributes from preexisting 
features/attributes.

## Learning Objectives
1. Data Literacy: Understand and interpret complex datasets related to public health.
2. Analytical Skills: Develop the ability to conduct thorough data analysis, including statistical and trend analysis.
3. Critical Thinking: Engage in hypothesis generation and testing, encouraging a deeper understanding of the data and its implications.
4. Technical Proficiency: Gain hands-on experience with data manipulation, visualization, and modeling techniques.
5. Communication Skills: Enhance the ability to present findings clearly and effectively.

# COVID-19 Dataset Subset - United States - Our World in Data
This README file describes the subset of the COVID-19 dataset that 
focuses on the United States. The dataset is provided by Our World 
in Data and includes both original and derived features. The dataset
has been curated by Bryan Tuck for in-depth analysis of the 
pandemic's impact within the United States.

## Dataset Overview:
This subset of the COVID-19 dataset reflects the situation in the United States, providing detailed data on cases, deaths, hospitalizations, and other metrics relevant to the pandemic. Our World in Data has compiled this information, drawing from official sources and making it accessible for public analysis. The dataset is part of a larger collection that tracks the global impact of the COVID-19 pandemic, with daily updates throughout its duration.

## Original Variables:
- `date`: The date when the data was recorded.
- `total_cases`: Total confirmed cases of COVID-19.
- `new_cases`: New confirmed cases of COVID-19 on the given date.
- `total_deaths`: Total deaths attributed to COVID-19.
- `new_deaths`: New deaths attributed to COVID-19 on the given date.
- `total_cases_per_million`: Total confirmed cases of COVID-19 per 1,000,000 people.
- `total_deaths_per_million`: Total deaths attributed to COVID-19 per 1,000,000 people.
- `icu_patients`: Number of COVID-19 patients in intensive care units (ICUs) on the given date.
- `hosp_patients`: Number of COVID-19 patients in the hospital on the given date.
- `weekly_hosp_admissions`: Number of COVID-19 patients newly admitted to hospitals in the given week.

## Derived Features:
- `daily_case_change_rate`: The percentage change in new cases compared to the total cases on the previous day.
- `daily_death_change_rate`: The percentage change in new deaths compared to the total deaths on the previous day.
- `hospitalization_rate`: The percentage of total COVID-19 cases that resulted in hospitalization on the given date.
- `icu_rate`: The percentage of total COVID-19 cases that required intensive care on the given date.
- `case_fatality_rate`: The percentage of total COVID-19 cases that resulted in death.
- `7day_avg_new_cases`: The 7-day rolling average of new COVID-19 cases.
- `7day_avg_new_deaths`: The 7-day rolling average of new COVID-19 deaths.
- `hospitalization_need`: Categorical assessment of hospitalization rates as 'Low', 'Medium', or 'High' based on quantile distribution.
- `icu_requirement`: Categorical assessment of ICU rates as 'Low', 'Medium', or 'High' based on quantile distribution.

The derived categorical assessments ('hospitalization_need' and 'icu_requirement') are relative to the dataset and are based on the distribution of the data within the United States. These labels are intended to facilitate the analysis and may not represent absolute thresholds for public health action.

The subset presented here includes only the most pertinent variables for a focused analysis on the United States, allowing for a clear understanding of the trends and patterns specific to the U.S. during the COVID-19 pandemic.

## Data Preprocessing:
- The 'date' column has been converted to a datetime object to facilitate time series analysis.
- Quantile-based discretization function (`pd.cut`) has been used to convert continuous variables into categorical variables for 'hospitalization_need' and 'icu_requirement'.
- Rolling window functions have been used to calculate 7-day averages for new cases and deaths.

This README is intended to provide a comprehensive understanding of the subset and its variables for anyone using it for analysis, modeling, or educational purposes.

## Example Ideas

You are not limited to these areas and are encouraged to explore with creativity and critical thinking. 

### Regression 
**Objective:** Utilize regression models to predict future healthcare requirements based on current trends.

**Task:** Predict the number of ICU admissions using variables like total_cases and total_deaths. Analyze how well your model performs and discuss its potential real-world applicability.

### Classification 
**Objective:** Classify days or periods into different risk categories based on COVID-19 data.

**Task:** Use classification models to categorize days into varying levels of hospitalization_need. Explore which features are most predictive and discuss the implications of your findings.

### Outlier Detection
**Objective:** Identify and analyze outliers to uncover unusual patterns or data errors.

**Task:** Use statistical methods to detect outliers in key variables such as new_cases, new_deaths, icu_patients, and hosp_patients. Techniques like the Interquartile Range (IQR), Z-scores, or visual methods (box plots) can be employed.

### Exploratory Data Analysis (EDA)
**Objective:** Conduct a thorough exploratory analysis to uncover underlying patterns and insights.

**Task:** Visualize trends in total_cases, total_deaths, and hospitalization_rate. Perform clustering and analysis to investigate any correlations or surprising patterns in the data and hypothesize their causes.


## Deliverables
1. A Python file with your code and analysis.  Make sure that plots/figures are generated for analyses to support your findings/conclusions.
2. A separate pdf of your report (1-2 pages) of your findings and the conclusions.
Please make sure that Github username, name, and PSID of all team members are included
in the report.

## Submission Instructions
Please push and commit your code and report to the github repository. 
Please ensure that your Github username, name, and PSID of all team members are included in both the code and the report.