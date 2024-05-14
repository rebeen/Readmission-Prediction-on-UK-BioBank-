import tensorflow 
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


# Prepare data for input into NN_improved function

Read_table=pd.read_csv("/home/wasim/wasim_data/customers/02_08_23_Nick_Reynolds_group/16_08_23_Nick_Reynolds_Readmission/including_10_100_patients/readmission_table_include_10_100_21022024.tsv", header=0, sep='\t')
Read_table['eid'] =Read_table ['eid'].astype('int')
Read_table.set_index('eid', inplace=True)

demographics=pd.read_csv("/home/wasim/wasim_data/customers/02_08_23_Nick_Reynolds_group/16_08_23_Nick_Reynolds_Readmission/including_10_100_patients/demographics.tsv", header=0, sep='\t')
demographics['eid'] =demographics['patient_id'].astype('int')
demographics.set_index('eid', inplace=True)

diagnosis =pd.read_csv("/home/wasim/wasim_data/customers/02_08_23_Nick_Reynolds_group/16_08_23_Nick_Reynolds_Readmission/including_10_100_patients/Top_codes_ICD10_binary_22022024.tsv", header=0, sep='\t')
diagnosis['eid'] =diagnosis ['eid'].astype('int')
diagnosis.set_index('eid', inplace=True)

prescription =pd.read_csv("/home/wasim/wasim_data/customers/02_08_23_Nick_Reynolds_group/16_08_23_Nick_Reynolds_Readmission/including_10_100_patients/BNF_binary_22022024.tsv", header=0, sep='\t')
prescription['eid'] =prescription ['eid'].astype('int')
prescription.set_index('eid', inplace=True)

Lon_term_condition =pd.read_csv("/home/wasim/wasim_data/customers/02_08_23_Nick_Reynolds_group/16_08_23_Nick_Reynolds_Readmission/including_10_100_patients/LTC_binary_22022024.tsv", header=0, sep='\t')
Lon_term_condition['eid'] =Lon_term_condition ['eid'].astype('int')
Lon_term_condition.set_index('eid', inplace=True)


df1 = pd.merge(Read_table,demographics, on="eid",  how="left")
df2=  pd.merge(df1,diagnosis, on="eid",  how="left")
df3 = pd.merge(df2, prescription, on="eid",  how="left")
df4 = pd.merge(df3, Lon_term_condition, on="eid",  how="left")
readmission_dataset=df4

readmission_dataset=readmission_dataset.fillna(0) # remove NAs due to merging


# Only maintain balanced classes on training dataset
# Obtain an initial size of the readmission balance class dataset for the test set
readmission_dataset_majority_class = readmission_dataset[readmission_dataset['readmission_status'] == 0].reset_index(drop=True) # Subset majority class
readmission_dataset_minority_class = readmission_dataset[readmission_dataset['readmission_status'] == 1].reset_index(drop=True) # Subset minority class
readmission_dataset_majority_sample = readmission_dataset_majority_class.sample(n=len(readmission_dataset_minority_class), random_state=None).reset_index(drop=True) # Sample the same number of majority class pati>
readmission_data_balanced = pd.concat([readmission_dataset_majority_sample, readmission_dataset_minority_class], axis=0).reset_index(drop=True) # Final balanced dataframe
test_fold_patients = len(readmission_data_balanced) * 0.2
majority_ratio = (len(readmission_dataset_majority_class) / len(readmission_dataset)) # Calculate original ratio of majority class but for a 20-fold size dataset
minority_ratio = (len(readmission_dataset_minority_class) / len(readmission_dataset)) # Calculate original ratio of minority class but for a 20-fold size dataset
majority_test_n = int(test_fold_patients * (majority_ratio)) # Obtain test size majority class based on original class ratio/proportion
minority_test_n = int(test_fold_patients * (minority_ratio))  # Obtain test size minority class based on original class ratio/proportion
readmission_dataset_majority_test = readmission_dataset[readmission_dataset['readmission_status'] == 0].sample(n=majority_test_n, random_state=None).reset_index(drop=True) # Sample 20-fold dataset but w>
readmission_dataset_minority_test = readmission_dataset[readmission_dataset['readmission_status'] == 1].sample(n=minority_test_n, random_state=None).reset_index(drop=True)
readmission_data_test = pd.concat([readmission_dataset_majority_test, readmission_dataset_minority_test], axis=0).reset_index(drop=True) # Final 20 fold test set with original class proportions

# Rebalance training set again without testing patients
readmission_dataset=readmission_dataset[-readmission_dataset['patient_id'].isin(readmission_data_test['patient_id'])].reset_index(drop=True)
readmission_dataset_majority_class = readmission_dataset[readmission_dataset['readmission_status'] == 0].reset_index(drop=True) # Subset majority class
readmission_dataset_minority_class = readmission_dataset[readmission_dataset['readmission_status'] == 1].reset_index(drop=True) # Subset minority class
readmission_dataset_majority_sample = readmission_dataset_majority_class.sample(n=len(readmission_dataset_minority_class), random_state=None).reset_index(drop=True) # Sample the same number of majority class pati>
readmission_data_balanced = pd.concat([readmission_dataset_majority_sample, readmission_dataset_minority_class], axis=0).reset_index(drop=True) # Final balanced dataframe

# Final testing and train splits shuffled
readmission_train = readmission_data_balanced.sample(frac=1, random_state=None).reset_index(drop=True) # Make sure shuffle order
readmission_test = readmission_data_test.sample(frac=1, random_state=None).reset_index(drop=True) # Make sure shuffle order


# Splitting the test dataset into the different feature groups
demo_train=readmission_train.loc[:,['Sex', 'Year_of_birth', 'Townsend_deprivation_index_at_recruitment', 'Alcohol_intake_frequency', 'Smoking_status', 'Alcohol_drinker_status', 'Ethnic_background','Body_mass_index', 'Age_at_recruitment', 'Body_fat_percentage']].values
LTC_train=readmission_train.loc[:,"cystic_renal":"downs"].values
diag_train=readmission_train.loc[:,"M201":"C944"].values
scripts_train=readmission_train.loc[:,"010101":"050301"].values
label_train=readmission_train['readmission_status'].values

demo_test=readmission_test.loc[:,['Sex', 'Year_of_birth', 'Townsend_deprivation_index_at_recruitment', 'Alcohol_intake_frequency', 'Smoking_status', 'Alcohol_drinker_status', 'Ethnic_background','Body_mass_index', 'Age_at_recruitment', 'Body_fat_percentage']].values
LTC_test=readmission_test.loc[:,"cystic_renal":"downs"].values
diag_test=readmission_test.loc[:,"M201":"C944"].values
scripts_test=readmission_test.loc[:,"010101":"050301"].values
label_test=readmission_test['readmission_status'].values

model = load_model('Final_NN_improved_best_settings_complex.h5') # load trained model that had the best epoch training and validation loss

# Get model predictions
train_predictions = model.predict([demo_train,diag_train,scripts_train,LTC_train])
test_predictions = model.predict([demo_test,diag_test,scripts_test,LTC_test])
print("predicting")

# Binarize the predictions using a threshold of 0.5
train_predictions_binary = np.where(train_predictions >= 0.5, 1, 0)
test_predictions_binary = np.where(test_predictions >= 0.5, 1, 0)

# Calculate metrics
metrics = {
    "Train AUROC": roc_auc_score(label_train, train_predictions),
    "Train AUPRC": average_precision_score(label_train, train_predictions),
    "Test AUROC": roc_auc_score(label_test, test_predictions),
    "Test AUPRC": average_precision_score(label_test, test_predictions),
    "Train Precision": precision_score(label_train, train_predictions_binary,average=None),
    "Train Recall": recall_score(label_train, train_predictions_binary,average=None),
    "Train F1 Score": f1_score(label_train, train_predictions_binary,average=None),
    "Test Precision": precision_score(label_test, test_predictions_binary,average=None),
    "Test Recall": recall_score(label_test, test_predictions_binary,average=None),
    "Test F1 Score": f1_score(label_test, test_predictions_binary,average=None)
}

print(f"{metrics}")
