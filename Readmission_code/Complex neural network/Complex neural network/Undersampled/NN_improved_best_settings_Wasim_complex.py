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

# NUM_THREADS=40
# tensorflow.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
# tensorflow.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)

# Prepare data for input into NN_improved function
# Prepare data for input into NN_improved function

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
readmission_dataset_majority_sample = readmission_dataset_majority_class.sample(n=len(readmission_dataset_minority_class), random_state=200).reset_index(drop=True) # Sample the same number of majority class pati>
readmission_data_balanced = pd.concat([readmission_dataset_majority_sample, readmission_dataset_minority_class], axis=0).reset_index(drop=True) # Final balanced dataframe
test_fold_patients = len(readmission_data_balanced) * 0.2
majority_ratio = (len(readmission_dataset_majority_class) / len(readmission_dataset)) # Calculate original ratio of majority class but for a 20-fold size dataset
minority_ratio = (len(readmission_dataset_minority_class) / len(readmission_dataset)) # Calculate original ratio of minority class but for a 20-fold size dataset
majority_test_n = int(test_fold_patients * (majority_ratio)) # Obtain test size majority class based on original class ratio/proportion
minority_test_n = int(test_fold_patients * (minority_ratio))  # Obtain test size minority class based on original class ratio/proportion
readmission_dataset_majority_test = readmission_dataset[readmission_dataset['readmission_status'] == 0].sample(n=majority_test_n, random_state=200).reset_index(drop=True) # Sample 20-fold dataset but w>
readmission_dataset_minority_test = readmission_dataset[readmission_dataset['readmission_status'] == 1].sample(n=minority_test_n, random_state=200).reset_index(drop=True)
readmission_data_test = pd.concat([readmission_dataset_majority_test, readmission_dataset_minority_test], axis=0).reset_index(drop=True) # Final 20 fold test set with original class proportions

# Rebalance training set again without testing patients
readmission_dataset=readmission_dataset[-readmission_dataset['patient_id'].isin(readmission_data_test['patient_id'])].reset_index(drop=True)
readmission_dataset_majority_class = readmission_dataset[readmission_dataset['readmission_status'] == 0].reset_index(drop=True) # Subset majority class
readmission_dataset_minority_class = readmission_dataset[readmission_dataset['readmission_status'] == 1].reset_index(drop=True) # Subset minority class
readmission_dataset_majority_sample = readmission_dataset_majority_class.sample(n=len(readmission_dataset_minority_class), random_state=200).reset_index(drop=True) # Sample the same number of majority class pati>
readmission_data_balanced = pd.concat([readmission_dataset_majority_sample, readmission_dataset_minority_class], axis=0).reset_index(drop=True) # Final balanced dataframe

# Final testing and train splits shuffled
readmission_train = readmission_data_balanced.sample(frac=1, random_state=200).reset_index(drop=True) # Make sure shuffle order
readmission_test = readmission_data_test.sample(frac=1, random_state=200).reset_index(drop=True) # Make sure shuffle order


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




# Final dimensions of input data. demographics= 11 dim, diagnosis =13 dim , scripts= 357 dim,  LTC= 201 dim 

def NN_improved(demographics,diagnosis,scripts,LTC,params={},**kwarg):

    # Reduce dropout rate will improve training time significantly. 
    # No need for high dropout for a simple NN as done in Chris model
    # Adding layer normalization will improve training time as well
    # Implement early stopping during training as Chris has already done
    # Chris did not overwrite (i.e. delete previous) layer transformations. This may also be a reason for slow training time

    # Simple multi-perceptron fully connected neural network

        demographics_input = Input(shape=(demographics,))
        diagnosis_input = Input(shape=(diagnosis,))
        scripts_input = Input(shape=(scripts,))
        LTC_input = Input(shape=(LTC,))

        demographics = demographics_input
        diagnosis = diagnosis_input
        scripts = scripts_input
        LTC = LTC_input

        for a in range(params.get('branching_num_layers_demographics',0)):    
            # demographics=Embedding(demographics.shape[1],params.get(),name='demographics_embed') No embedding layer for demographics as already represented in raw continous format
            demographics = Dense(params.get('neurons_demo'), activation='relu')(demographics)
            demographics=LayerNormalization(axis=-1)(demographics)
            demographics = Dropout(0.2)(demographics) # reduce drop out rate to 0.2

        # diagnosis = Embedding(2,params.get('embedding_dim_diagnosis'),name='diagnosis_embed')(diagnosis)
        for b in range(params.get('branching_num_layers_diagnosis',0)):
            diagnosis = Dense(params.get('neurons_diag'), activation='relu')(diagnosis)
            diagnosis=LayerNormalization(axis=-1)(diagnosis)
            diagnosis = Dropout(0.2)(diagnosis) # reduce drop out rate to 0.2
            #diagnosis = Flatten()(diagnosis)

        # LTC = Embedding(2, params.get('embedding_dim_LTC'), name='ltc_embed')(LTC)
        for c in range(params.get('branching_num_layers_LTC',0)):
            LTC = Dense(params.get('neurons_LTC'), activation='relu')(LTC)
            LTC = LayerNormalization(axis=-1)(LTC)
            LTC = Dropout(0.2)(LTC) # reduce drop out rate to 0.2
            #LTC = Flatten()(LTC)

        #scripts = Embedding(2, params.get('embedding_dim_SCR'), name='preadmission_scripts_embed')(scripts)
        for d in range(params.get('branching_num_layers_scripts',0)):
            scripts = Dense(params.get('neurons_SCR'), activation='relu')(scripts)
            scripts = LayerNormalization(axis=-1)(scripts)
            scripts = Dropout(0.2)(scripts) # reduce drop out rate to 0.2
            #scripts = Flatten()(scripts)

        X = Concatenate(axis=-1)([demographics, diagnosis, LTC, scripts])

        for e in range(params.get('num_layers_',0)):
            X = Dense(params.get('neurons_output'), activation='relu')(X)
            X = LayerNormalization(axis=-1)(X)
            X = Dropout(0.2)(X)


        output = Dense(1, activation='sigmoid', name='output')(X)

        model = Model(inputs=[demographics_input,diagnosis_input,scripts_input,LTC_input], outputs=output)

        return model



############## TRAINING THE IMPROVED neural network

from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from tensorflow.keras.optimizers import SGD

early_stopping=EarlyStopping(monitor='val_loss',restore_best_weights=True,patience=5)

params = {'branching_num_layers_demographics': 1, # Neural network settings with the best training and validation loss
'branching_num_layers_diagnosis': 1,
    'branching_num_layers_LTC': 1,
    'branching_num_layers_scripts': 1, 
    'num_layers_': 1,
    'neurons_demo': 60,
    'neurons_diag': 1000,
    'neurons_LTC': 400,
    'neurons_SCR': 600,
    'neurons_output': 1000}

# Build and compile model
model_=NN_improved(demo_train.shape[1],diag_train.shape[1],scripts_train.shape[1],LTC_train.shape[1],params=params)
print('initialized model')
lr = 0.00001 # use default Adam learning rate

adam_optimizer = Adam(learning_rate=lr)
#sgd=SGD(learning_rate=lr)
model_.compile(optimizer=adam_optimizer,loss='binary_crossentropy')

# Train model
print('training model')
history=model_.fit([demo_train,diag_train,scripts_train,LTC_train],label_train, validation_data=([demo_test, diag_test, scripts_test, LTC_test], label_test), epochs=30, batch_size=64, callbacks=[early_stopping], verbose=1)


model_.save("Final_NN_improved_best_settings_complex.h5")

