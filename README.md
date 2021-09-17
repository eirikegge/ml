# ml
Data Mining, Machine Learning &amp; Deep Learning (CBS Project)

Note: 
 - Code (py) & Dataset (csv)
 - GridSearchCV runned on UCloud: 
 	- 8 vCPU
	- 47 RAM (GB)

Co-authors:
 - Helena van Eek
 - Magnus Beck Eliassen
 - Sabrina Breunig

--Abstract Complementary Research Paper--

Due to stroke being the second leading cause of preventable death worldwide, it is vital to understand the medical conditions and lifestyle factors impacting the risk of the disease to conduct medical examinations promptly. Consequently, data-driven solutions for accurately identifying a patient’s stroke risk are valuable for heightening chances of prevention and early treatment. For this purpose, this paper presents supervised machine learning techniques leveraging medical data for the prediction of stroke risk. The problem investigated revolves around the vast imbalance of the examined medical dataset, causing the algorithms’ failure to detect a single stroke risk in the baseline models. To master this challenge, this paper first applies the oversampling techniques Synthetic Minority Oversampling Technique (SMOTE) and Adaptive Synthetic Sampling (ADASYN) to then finally tune the classifiers Support Vector Machines (SVM), Random Forest (RF), Multi-Layer Perceptron (MLP), and the majority vote on the synthetically balanced dataset to boost the recall score of the minority class as the prioritized performance heuristic. The results highlight SVM combined with SMOTE as the favored classifier due to its recall score of 1.0 for correctly predicting “stroke” with a false positive rate of 60%. The presentation of the variables heart disease, hypertension, age, residence type, and average glucose level as the most important features for the proposed classifer fnalizes the conducted analysis.


--



