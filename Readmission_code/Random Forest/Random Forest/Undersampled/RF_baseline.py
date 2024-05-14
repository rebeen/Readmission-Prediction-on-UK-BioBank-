import tensorflow
from tensorflow.keras.layers import Flatten, Input, Dense, LayerNormalization, Dropout, Concatenate, Embedding
from keras.models import Model
from tensorflow.keras.metrics import TrueNegatives,TruePositives,FalseNegatives,FalsePositives,AUC,BinaryCrossentropy
import pandas as pd
import numpy as np
import tqdm
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle

NUM_THREADS=27
tensorflow.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
# tensorflow.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)

# Prepare data for input into NN_improved function


import pandas as pd
from sklearn.utils import shuffle

        
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score


Read_table=pd.read_csv("/home/wasim/wasim_data/14_08_23_Readdmision/final_readmission_dataset/Corrected_readmission_data_rerun_models/including_10_100_patients/readmission_table_include_10_100_21022024.tsv", header=0, sep='\t')
Read_table['eid'] =Read_table ['eid'].astype('int')
Read_table.set_index('eid', inplace=True)

demographics=pd.read_csv("/home/wasim/wasim_data/14_08_23_Readdmision/final_readmission_dataset/Corrected_readmission_data_rerun_models/including_10_100_patients/demographics.tsv", header=0, sep='\t')
demographics['eid'] =demographics['patient_id'].astype('int')
demographics.set_index('eid', inplace=True)

diagnosis =pd.read_csv("/home/wasim/wasim_data/14_08_23_Readdmision/final_readmission_dataset/Corrected_readmission_data_rerun_models/including_10_100_patients/Top_codes_ICD10_binary_22022024.tsv", header=0, sep='\t')
diagnosis['eid'] =diagnosis ['eid'].astype('int')
diagnosis.set_index('eid', inplace=True)

prescription =pd.read_csv("/home/wasim/wasim_data/14_08_23_Readdmision/final_readmission_dataset/Corrected_readmission_data_rerun_models/including_10_100_patients/BNF_binary_22022024.tsv", header=0, sep='\t')
prescription['eid'] =prescription ['eid'].astype('int')
prescription.set_index('eid', inplace=True)

Lon_term_condition =pd.read_csv("/home/wasim/wasim_data/14_08_23_Readdmision/final_readmission_dataset/Corrected_readmission_data_rerun_models/including_10_100_patients/LTC_binary_22022024.tsv", header=0, sep='\t')
Lon_term_condition['eid'] =Lon_term_condition ['eid'].astype('int')
Lon_term_condition.set_index('eid', inplace=True)


Age_at_event = pd.read_csv("/home/wasim/wasim_data/14_08_23_Readdmision/final_readmission_dataset/Corrected_readmission_data_rerun_models/including_10_100_patients/Age_at_LTC_event_padded_zeros.tsv", header=0, sep='\t')
Age_at_event['eid'] =Age_at_event ['eid'].astype('int')
Age_at_event.set_index('eid', inplace=True)


df1 = pd.merge(Read_table,demographics, on="eid",  how="left")
df2 =  pd.merge(df1,diagnosis, on="eid",  how="left")
df3 = pd.merge(df2, prescription, on="eid",  how="left")
df4 = pd.merge(df3, Lon_term_condition, on="eid",  how="left")
df5 = pd.merge(df4, Age_at_event, on="eid",  how="left")
readmission_dataset=df5

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


# Combine all features for Random Forest
X_train = np.concatenate([demo_train, diag_train, scripts_train, LTC_train], axis=1)
X_test = np.concatenate([demo_test, diag_test, scripts_test, LTC_test], axis=1)

# It's generally a good idea to standardize the dataset when using models that rely on the distance between data points.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train= label_train
y_test = label_test

# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=200,n_jobs=60)  # You can fine-tune these parameters

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]  # Probability of the positive class
y_pred = rf_classifier.predict(X_test)

# Calculate performance metrics
auc_roc = roc_auc_score(y_test, y_pred_proba)
auc_prc = average_precision_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred,average=None)
recall = recall_score(y_test, y_pred,average=None)
f1 = f1_score(y_test, y_pred,average=None)

# Print metrics
metrics_string = f"AUROC: {auc_roc}, AUPRC: {auc_prc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}"
print(metrics_string)
