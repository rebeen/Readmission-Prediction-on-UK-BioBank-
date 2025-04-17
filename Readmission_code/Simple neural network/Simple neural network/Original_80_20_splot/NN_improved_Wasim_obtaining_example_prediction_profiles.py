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
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
# NUM_THREADS=27
# tensorflow.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
# tensorflow.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)

# Prepare data for input into NN_improved function

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





df1 = pd.merge(Read_table,demographics, on="eid",  how="left")
df2 =  pd.merge(df1,diagnosis, on="eid",  how="left")
df3 = pd.merge(df2, prescription, on="eid",  how="left")
df4 = pd.merge(df3, Lon_term_condition, on="eid",  how="left")

readmission_dataset=df4

readmission_dataset=readmission_dataset.fillna(0) # remove NAs due to merging

readmission_train,readmission_test=train_test_split(readmission_dataset,test_size=0.2,stratify=readmission_dataset['readmission_status'],random_state=200)

readmission_train=readmission_train.sample(frac=1,random_state=200).reset_index(drop=True) # make sure shuffle
readmission_test=readmission_test.sample(frac=1,random_state=200).reset_index(drop=True)

# Splitting the test dataset into the different feature groups
demo_train=readmission_train.loc[:,['Sex', 'Year_of_birth', 'Townsend_deprivation_index_at_recruitment', 'Alcohol_intake_frequency', 'Smoking_status', 'Alcohol_drinker_status', 'Ethnic_background','Body_mass_index', 'Age_at_recruitment', 'Body_fat_percentage']].values
LTC_train=readmission_train.loc[:,"cystic_renal":"downs"].values
diag_train=readmission_train.loc[:,"M201":"C944"].values # M201:C944 starting code of all_top_codes dataset. K810:C07 starting code of 2 year window + all_top_codes dataset
scripts_train=readmission_train.loc[:,"010101":"050301"].values
label_train=readmission_train['readmission_status'].values

demo_test=readmission_test.loc[:,['Sex', 'Year_of_birth', 'Townsend_deprivation_index_at_recruitment', 'Alcohol_intake_frequency', 'Smoking_status', 'Alcohol_drinker_status', 'Ethnic_background','Body_mass_index', 'Age_at_recruitment', 'Body_fat_percentage']].values
LTC_test=readmission_test.loc[:,"cystic_renal":"downs"].values
diag_test=readmission_test.loc[:,"M201":"C944"].values # M201:C944 starting code of all_top_codes dataset. K810:C07 starting code of 2 year window + all_top_codes dataset
scripts_test=readmission_test.loc[:,"010101":"050301"].values
label_test=readmission_test['readmission_status'].values


# Final dimensions of input data. demographics= 11 dim, diagnosis =13 dim , scripts= 357 dim,  LTC= 201 dim 

def NN_improved(demographics,diagnosis,scripts,LTC,params={},**kwarg):

    # Reduce dropout rate will improve training time significantly. 
    # No need for high dropout for a simple NN as done in Chris model
    # Adding layer normalization will improve training time as well
    # Implement early stopping during training as Chris has already done
    # Chris did not overwrite (i.e. delete previous) layer transformations. This may also be a reason for slow training time

    # Simple neural network with only 1 fully connected layer 

    demographics_input = Input(shape=(demographics,))
    diagnosis_input = Input(shape=(diagnosis,))
    scripts_input = Input(shape=(scripts,))
    LTC_input = Input(shape=(LTC,))

    demographics = demographics_input
    diagnosis = diagnosis_input
    scripts = scripts_input
    LTC = LTC_input

    # GOT RID OF EMBEDDING LAYERS. Use Binary Encodings. There are not sematic meanings between script or LTCs or diagnosis indices for embeddings to capture. Maybe be reducing performance. Embedding layers generally used to capture sematic meanings e.g. between words
    # BUT If the integers are purely categorical and have no meaningful order or relationship, better to use binary encodings 
    #diagnosis = Embedding(2, params.get('embedding_dim_diagnosis'), name='diagnosis_embed')(diagnosis) # embedding dim is 2 i.e. 2 for 0 or 1 this is the correct way for binary encoding inputs
    diagnosis = Flatten()(diagnosis)
    
    # LTC = Embedding(2, params.get('embedding_dim_LTC'), name='ltc_embed')(LTC)
    LTC = Flatten()(LTC)
    
    # scripts = Embedding(2, params.get('embedding_dim_SCR'), name='preadmission_scripts_embed')(scripts)
    scripts =  Flatten()(scripts)

    #Age_at_event = Flatten()(Age_at_event)
    
    X = Concatenate(axis=-1)([demographics, diagnosis, LTC, scripts])

    for e in range(params.get('num_layers_',0)): # single fully connected layer(s)
        X = Dense(params.get('neurons_output'), activation='relu')(X)
        X = LayerNormalization(axis=-1)(X)
        X = Dropout(0.2)(X)


    output = Dense(1, activation='sigmoid', name='output')(X) # output layer

    model = Model(inputs=[demographics_input,diagnosis_input,scripts_input,LTC_input], outputs=output)

    return model



############## TRAINING THE IMPROVED neural network

from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


# Define range of layers, neurons and embedding dimensions to test
 # neurons  decided based on dim of original input data i.e. go up to original input dim dimension in powers of 2
 # embedding dim first decided based on square root number of unique categorie variables and then values chosen around that value in powers of 2
 
# best metrics
out_layers=[2]   ; out_neurons=[800] # output layer neurons approximates decided on the possible concatenate dimensions of the previous combinations and taken as average been final output (1) and previous Concatenate dim size

from tensorflow.keras.optimizers import SGD

early_stopping=EarlyStopping(monitor='val_loss',restore_best_weights=True,patience=5)

Model_settings=[]

for i in tqdm(out_layers): # testing all combinations of chosen hyperparameter search space

    for j in tqdm(out_neurons):

        params = {
            'num_layers_': i,
            'neurons_output': j}

        # Build and compile model
        model_=NN_improved(demo_train.shape[1],diag_train.shape[1],scripts_train.shape[1],LTC_train.shape[1],params=params)
        print('initialized model')
        lr = 0.01 # set Chris original learning rate to Adam optimizer default 

        adam_optimizer = Adam(learning_rate=lr)
        #sgd=SGD(learning_rate=lr)
        model_.compile(optimizer=adam_optimizer,loss='binary_crossentropy')

        # Train model
        print('training model')
        history=model_.fit([demo_train,diag_train,scripts_train,LTC_train],label_train
                            ,epochs=10,validation_data=([demo_test,diag_test,scripts_test,LTC_test], label_test),batch_size=64,callbacks=[early_stopping],verbose=1)
        print('training complete')
        loss_=history.history['loss'][-1]

        Model_loss_=f'Model_loss:{loss_},output_layers:{i},neurons_output{j}'

        Model_settings.append(Model_loss_)
        print(Model_loss_)
        
        # Get model predictions
        test_predictions = model_.predict([demo_test,diag_test,scripts_test,LTC_test])
        print("predicting")
        # Binarize the predictions using a threshold of 0.5
        test_predictions_binary = np.where(test_predictions >= 0.5, 1, 0)
        # Calculate metrics
        metrics = {
        "Test AUROC": roc_auc_score(label_test, test_predictions),
        "Test AUPRC": average_precision_score(label_test, test_predictions),
        "Test Precision": precision_score(label_test, test_predictions_binary,average=None),
        "Test Recall": recall_score(label_test, test_predictions_binary,average=None),
        "Test F1 Score": f1_score(label_test, test_predictions_binary,average=None)
         }
        print(metrics)

df_test_predictions_simple=pd.concat([readmission_test.reset_index(drop=True),pd.DataFrame([test_predictions_binary],columns=['Predicted_label']).reset_index(drop=True)],axis=1)

df_test_predictions_simple.to_csv("df_test_predictions_simple.csv")
