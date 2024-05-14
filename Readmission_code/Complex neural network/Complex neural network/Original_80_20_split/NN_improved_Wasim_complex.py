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

#NUM_THREADS=60
#tensorflow.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
# tensorflow.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)

# Prepare data for input into NN_improved function

from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score


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
diag_train=readmission_train.loc[:,"M201":"C944"].values
scripts_train=readmission_train.loc[:,"010101":"050301"].values
label_train=readmission_train['readmission_status'].values

demo_test=readmission_test.loc[:,['Sex', 'Year_of_birth', 'Townsend_deprivation_index_at_recruitment', 'Alcohol_intake_frequency', 'Smoking_status', 'Alcohol_drinker_status', 'Ethnic_background','Body_mass_index', 'Age_at_recruitment', 'Body_fat_percentage']].values
LTC_test=readmission_test.loc[:,"cystic_renal":"downs"].values
diag_test=readmission_test.loc[:,"M201":"C944"].values
scripts_test=readmission_test.loc[:,"010101":"050301"].values
label_test=readmission_test['readmission_status'].values


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
        
        # GOT RID OF EMBEDDING LAYERS. Use Binary Encodings. There are not sematic meanings between script or LTCs or diagnosis indices for embeddings to capture. Maybe be reducing performance. Embedding layers g>    # BUT If the integers are purely categorical and have no meaningful order or relationship, better to use binary encodings
        for a in range(params.get('branching_num_layers_demographics',0)):    
            # demographics=Embedding(demographics.shape[1],params.get(),name='demographics_embed') No embedding layer for demographics as already represented in raw continous format
            demographics = Dense(params.get('neurons_demo'), activation='relu')(demographics)
            demographics=LayerNormalization(axis=-1)(demographics)
            demographics = Dropout(0.2)(demographics) # reduce drop out rate to 0.2

        # diagnosis = Embedding(2,params.get('embedding_dim_diagnosis'),name='diagnosis_embed')(diagnosis) # IMPORTANT: Embedding input dim is 2 because vocab consists of 1 or 0 i.e. binary notation 
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

        # scripts = Embedding(2, params.get('embedding_dim_SCR'), name='preadmission_scripts_embed')(scripts)
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


# Define range of layers, neurons and embedding dimensions to test
demographic_layers=[1,2,3] ; demo_neurons=[15,30,60] 
diagnosis_layers=[1,2,3] ; diag_neurons=[250,500,1000]   ;  # neurons  decided based on dim of original input data i.e. go up to original input dim dimension in powers of 2
LongTC_layers=[1,2,3] ; LongTC_neurons=[100,200,400]    ;  # embedding dim first decided based on square root number of unique categorie variables and then values chosen around that value in powers of 2
SCR_layers=[1,2,3] ; SCR_neurons=[150,300,600] ; out_layers=[1,2,3]   ; out_neurons=[250,500,1000] # output layer neurons approximates decided on the possible concatenate dimensions of the previous combinations and taken as average been final output (1) and previous Concatenate dim size

from tensorflow.keras.optimizers import SGD

early_stopping=EarlyStopping(monitor='val_loss',restore_best_weights=True,patience=5)

Model_settings=[]

for i in tqdm(range(0,3)): # testing all combinations of chosen hyperparameter search space

    for j in tqdm(range(0,3)):

        params = {'branching_num_layers_demographics': demographic_layers[i],
        'branching_num_layers_diagnosis': diagnosis_layers[i],
            'branching_num_layers_LTC': LongTC_layers[i],
            'branching_num_layers_scripts': SCR_layers[i],
            'num_layers_': out_layers[i],
            'neurons_demo': demo_neurons[j],
            'neurons_diag': diag_neurons[j],
            'neurons_LTC': LongTC_neurons[j],
            'neurons_SCR': SCR_neurons[j],
            'neurons_output': out_neurons[j]}

        # Build and compile model
        model_=NN_improved(demo_train.shape[1],diag_train.shape[1],scripts_train.shape[1],LTC_train.shape[1],params=params)
        print('initialized model')
        lr = 0.0001 # using default Adam learning rate 

        adam_optimizer = Adam(learning_rate=lr)
        #sgd=SGD(learning_rate=lr)
        model_.compile(optimizer=adam_optimizer,loss='binary_crossentropy')

        # Train model
        print('training model')
        history=model_.fit([demo_train,diag_train,scripts_train,LTC_train],label_train
                            ,epochs=5,validation_data=([demo_test,diag_test,scripts_test,LTC_test],label_test),batch_size=64,callbacks=[early_stopping],verbose=1)
        print('training complete')
        loss_=history.history['loss'][-1]

        Model_loss_=f'Model_loss:{loss_}, demographic_layers:{demographic_layers[i]}, diagnosis_layers:{diagnosis_layers[i]}, LTC_layers:{LongTC_layers[i]},script_layers:{SCR_layers[i]},output_layers:{out_layers[i]} \
                    ,neurons_demo:{demo_neurons[j]}, neurons_diag:{diag_neurons[j]}, neurons_ltc:{LongTC_neurons[j]}, neurons_script:{SCR_neurons[j]}, neurons_output{out_neurons[j]} '

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
