# Personalized Casual Recourse

## Author
Denise Tampieri (denise.tampieri@studenti.unitn.it)

## Overview
This project focuses on developing a personalized casual recourse framework that comprises two main stages: **Estimation of the user SCM** and **Recourse using the Estimated SCM**. The estimation phase represents our innovative contribution to the field and is based on the Causal structures of [ijmbarr/causalgraphicalmodels](https://github.com/ijmbarr/causalgraphicalmodels), while the recourse phase heavily relies on the implementation from [RicardoDominguez/AdversariallyRobustRecourse](https://github.com/RicardoDominguez/AdversariallyRobustRecourse).

## Prerequisites
Before running the code, please ensure you have Python installed on your system. It is recommended to create a virtual environment for this project.

### Setting Up the Environment
1. Create a virtual environment:
   ```bash
   python -m venv your_env_name
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     your_env_name\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source your_env_name/bin/activate
     ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code
To run the code, you can use the following files that start with **main**:

- **For Estimation:**
  - `main_estimation_SCM.py`: Executes the estimation process.
  
- **For Recourse:**
  - `main_find_recourse.py`: Executes the recourse process.

- **For Plotting Estimation Results:**
  - `main_plot_results_estimation_SCM.py`: Generates plots for the estimation phase (e.g., Figures 4.2 and 4.3).
  
- **For Plotting Recourse Results:**
  - `main_recourse_plots_functions.py`: Generates plots for recourse, focusing on cost and effectiveness (e.g., Figures 4.6 and 4.7)

## Conclusion
Feel free to explore the code and adapt it for your own use. For any questions or further information, please contact me.
```
