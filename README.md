# MATH5472-Final Project
The code here is for review of paper False discovery rates: a new deal
## fdr_bayes.py
This is the package for method developed according to the paper. The required parameter at class initialization is beta value and it's corresponding measurement precision. There is another optional parameter lam0 which is the lambda value for $\pi_0$ applied in penalty score caluclation. Running the fit method will calibrate the model. lfdr, lfsr can be calculated by calling the corresponding function. 
## fdr_emp.py
This is the package developed for emprical FDR method. The required parameter at class initialization is z value. Running the fit method will calibrate the model. lfdr can be calculated by calling the corresponding function. 
## fdr_project.py
This script runs the analysis for the report, including generating each figure. It utilizes classes from dr_bayes.py and fdr_emp.py for the necessary calculations.
