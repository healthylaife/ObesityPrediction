# ObesityPrediction
This repository contains files to build, train and test a deep-learning prediction model.


##### Data Availability
Interested scholars can access the data by contacting Nemours Biomedical Research Informatics Center and signing a data use agreement.

### Repository Structure

- **cohort_selection_2.ipynb**
	is the file used to select final cohort for descriptive analysis and prediction model training
- **cohort_prelim**
	is the file used to run descriptive analysis on the selected cohort.
- **dl_train_sig_obs.py**
  consists of code to create batches of data according to batch_size and create, train and test the model.
- **dl_train_sig_obs_geo.py**
  consists of code to create batches of data for temporal and geographic validation of the model.
- **mimic_model_sig_obs.py**
  consist of different model architectures.
- **evaluation.py**
  consists of class to perform evaluation of results obtained from models.
- **fairness.py**
  consists of code to perform fairness evaluation.
- **parameters.py**
  consists of list of hyperparameters to be defined for model training.
- **./saved_models**
	consists of models saved during training.


