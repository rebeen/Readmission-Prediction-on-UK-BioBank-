import pandas as pd
import copy
import  numpy as np
import os
import itertools
from imblearn.over_sampling import ADASYN, SMOTE, SVMSMOTE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import recall_score, precision_score, f1_score
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Add, Input, Embedding, concatenate, Flatten, Dropout, BatchNormalization,LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn import metrics
from sklearn.metrics import auc
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import NMF

from sklearn.preprocessing import MinMaxScaler
#!pip install ebcdic
import tqdm
from tqdm import tqdm
from sklearn.metrics import f1_score

print("reading files ")
Read_table=pd.read_csv("readmission_table_include_10_100_21022024.tsv", header=0, sep='\t')
Read_table['eid'] =Read_table ['eid'].astype('int')
Read_table.set_index('eid', inplace=True)

demographics = pd.read_csv("demographics.tsv",header=0, sep='\t')
demographics['eid'] = demographics['patient_id'].astype('int')
demographics.set_index('eid', inplace=True)



diagnosis =pd.read_csv("Top_codes_ICD10_binary_22022024.tsv", header=0, sep='\t')
diagnosis['eid'] =diagnosis ['eid'].astype('int')
diagnosis.set_index('eid', inplace=True)

prescription =pd.read_csv("BNF_binary_22022024.tsv", header=0, sep='\t')
prescription['eid'] =prescription ['eid'].astype('int')
prescription.set_index('eid', inplace=True)

Lon_term_condition =pd.read_csv("LTC_binary_22022024.tsv", header=0, sep='\t')
Lon_term_condition['eid'] =Lon_term_condition ['eid'].astype('int')
Lon_term_condition.set_index('eid', inplace=True)

print(" data is read")



print(Lon_term_condition.shape)
print(prescription.shape)
print(diagnosis.shape)
print(demographics.shape)
print(Read_table.shape)


scaler = MinMaxScaler()


df1 = pd.merge(Read_table, demographics, on="eid", how="left")
df2 = pd.merge(df1, diagnosis, on="eid", how="left")
df3 = pd.merge(df2, prescription, on="eid", how="left")
df4 = pd.merge(df3, Lon_term_condition, on="eid", how="left")
readmission_dataset = df4




readmission_dataset=readmission_dataset.drop(['admidate', 'admimeth', 'disdate', 'classpat_uni','patient_id'],axis=1)
print("readmission_dataset")
print(readmission_dataset.shape)

readmission_dataset=readmission_dataset.fillna(0)


readmission_dataset_majority_class = readmission_dataset[readmission_dataset['readmission_status'] == 0].reset_index(drop=True) # Subset majority class
readmission_dataset_minority_class = readmission_dataset[readmission_dataset['readmission_status'] == 1].reset_index(drop=True) # Subset minority class
readmission_dataset_majority_sample = readmission_dataset_majority_class.sample(n=len(readmission_dataset_minority_class), random_state=200).reset_index(drop=True) # Sample the same number of majority class pati>
readmission_data_balanced = pd.concat([readmission_dataset_majority_sample, readmission_dataset_minority_class], axis=0).reset_index(drop=True) # Final balanced dataframe
test_fold_patients = len(readmission_data_balanced) * 0.2

majority_ratio = (len(readmission_dataset_majority_class) / len(readmission_dataset)) # Calculate original ratio of majority class but for a 20-fold size dataset
minority_ratio = (len(readmission_dataset_minority_class) / len(readmission_dataset)) # Calculate original ratio of minority class but for a 20-fold size dataset

print("majority_ratio" ,majority_ratio)
print("majority_ratio" ,minority_ratio)

majority_test_n = int(test_fold_patients * (majority_ratio)) # Obtain test size majority class based on original class ratio/proportion
minority_test_n = int(test_fold_patients * (minority_ratio))  # Obtain test size minority class based on original class ratio/proportion

print("majority_test_n" ,majority_test_n)
print("minority_test_n" ,minority_test_n)

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

print("readmission_train",readmission_train.shape)
print("readmission_test",readmission_test.shape)

print("ends w code")
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

def SSL_method(demo,diag,scripts,LTC):

    static_input = Input(shape=(demo,), name='demographics_input')
    static_output = Dense(10, activation='relu', name='static_dense_1')(static_input)
    static_output = Dropout(0.4)(static_output)
    #static_skip   = Dense(64, activation='relu', name='static_dense_1')(static_output)
    #static_output = Dense(512, activation='relu', name='static_dense_2')(static_output)
    static_output = Dropout(0.2)(static_output)
    static_output= Flatten()(static_output)

    diag_input = Input(shape=(diag,), name='diag_input')
    diag_embed = Embedding(1306, 100, input_length=10)(diag_input)
    diag_output = Dense(1600, activation='relu')(diag_input)
    diag_output = Dropout(0.4)(diag_output)
    diag_output = BatchNormalization()(diag_output)
    diag_output = LayerNormalization()(diag_output)
    #diag_output= Dense(800, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),name='diag_dense_1')(diag_output)
    #diag_output = BatchNormalization()(diag_output)
    #diag_output = LayerNormalization()(diag_output)
    #diag_output = Dropout(0.8)(diag_output)
    diag_output = Dense(512, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),name='diag_dense_2') (diag_output)
    diag_output = Flatten()(diag_output)

    ltc_input = Input(shape=(LTC,), name='ltc_input')
    ltc_embed = Embedding(202, 100, input_length=15, name='ltc_embed')(ltc_input)
    ltc_output= Dense(204, activation='relu')(ltc_input)
    ltc_output = Dropout(0.4)(ltc_output)
    ltc_output = BatchNormalization()(ltc_output)
    ltc_output = LayerNormalization()(ltc_output)
    #ltc_output = Dense(800, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), activation='relu',name='ltc_dense_1')(ltc_output)
    #ltc_output = BatchNormalization()(ltc_output)
    #ltc_output = LayerNormalization()(ltc_output)
    #ltc_output = Dropout(0.7)(ltc_output)
    ltc_output = Dense(512, activation='relu', name='ltc_dense_2')(ltc_output)
    ltc_output = Flatten()(ltc_output)

    scripts_input = Input(shape=(scripts,), name='scripts_input')
    scripts_embed = Embedding(478, 100, input_length=66, name='scripts_embed')(scripts_input)
    scripts_out = Dense(250, activation='relu',name='scripts_dense_1')(scripts_input )
    scripts_out = Dropout(0.4)(scripts_out)
    scripts_out = BatchNormalization()(scripts_out)
    scripts_out = LayerNormalization()(scripts_out)
    #scripts_out = Dense(800, activation='relu',name='scripts_dense_2')(scripts_input)
    #scripts_out = BatchNormalization()(scripts_out)
    #scripts_out = LayerNormalization()(scripts_out)
    #scripts_out = Dropout(0.8)(scripts_out)
    # scripts_out = Dense(, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),name='scripts_dense_2')(scripts_out)
    scripts_out = Dense(512, activation='relu', name='scripts_dense_3')(scripts_out)
    scripts_out = Flatten()(scripts_out)

    concat = concatenate([static_output,diag_output, scripts_out,ltc_output],name='concatenate')
    # decoder
    decoder_static = Dense(demo)(concat)
    decoder_diag = Dense(diag)(concat)
    decoder_ltc = Dense(LTC)(concat)
    decoder_scripts = Dense(scripts)(concat)


    model = Model(inputs=[static_input, diag_input, scripts_input,ltc_input],
                  outputs=[decoder_static, decoder_diag, decoder_scripts,decoder_ltc])

    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0001))
    model.summary()
    return model
    # plot_model(model, to_file='modelnn_plot.png', show_shapes=True, show_layer_names=True)

def standardize(train, test):


    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)+0.000001

    X_train = (train - mean) / std
    X_test = (test - mean) /std
    return X_train, X_test

demo_train,demo_test = standardize( demo_train,demo_test)
LTC_train,LTC_test =standardize( LTC_train,LTC_test)
diag_train,diag_test = standardize( diag_train,diag_test)
scripts_train,scripts_test =standardize(  scripts_train,scripts_test)

print(demo_train.shape,demo_test.shape)
print(LTC_train.shape,LTC_test.shape)
print(diag_train.shape,diag_test.shape)
print(scripts_train.shape,scripts_test.shape)

def DA_Jitter(X, sigma=0.005):
    randata = np.random.normal(loc=0, scale=sigma, size=X.shape)
    # print("jitter",randata)
    return X+randata

original_train = copy.deepcopy ([demo_train,diag_train,scripts_train,LTC_train,])
corrupt_train= [DA_Jitter(x) for x in [demo_train,diag_train,scripts_train,LTC_train]]
print('initialized model')

print(demo_train.shape)
print(LTC_train.shape)
print(diag_train.shape)
print(scripts_train.shape)



model_=SSL_method(demo_train.shape[1],diag_train.shape[1],scripts_train.shape[1],LTC_train.shape[1])

# Train model
print('training model')
early_stopping = EarlyStopping(monitor='loss',restore_best_weights=True,patience=5)

history = model_.fit(corrupt_train,original_train,epochs=200,batch_size=64,callbacks=[early_stopping],verbose=2)
print('training complete')
loss_=history.history['loss'][-1]



nnet_concat_layer= Model(inputs=model_.input,outputs=model_.get_layer('concatenate').output)
#nnet_concat_layer.trainable = True

for layer in nnet_concat_layer.layers:
    layer.trainable = True

demo_input = Input(shape=(demo_train.shape[1]))
ltc_input = Input(shape=(LTC_train.shape[1]))
diagnosis_input = Input(shape=(diag_train.shape[1]))
script_input = Input(shape=(scripts_train.shape[1]))

x=nnet_concat_layer([demo_input,diagnosis_input,script_input,ltc_input], training=True)

layer_1=Dense(512, activation='elu')(x)
layer_1 = BatchNormalization()(layer_1)
layer_1 = LayerNormalization()(layer_1)
layer_1=Dropout(0.2)(layer_1)

layer_11=Dense(256, activation='elu')(layer_1)

layer_11 = BatchNormalization()(layer_11)
layer_11 = LayerNormalization()(layer_11)
layer_2=Dropout(0.2)(layer_11)
layer_22=Dense(64, activation='elu')(layer_2)
layer_22 = BatchNormalization()(layer_22)
layer_22 = LayerNormalization()(layer_22)

outputs =Dense(1,activation='sigmoid')(layer_22)

linear_model = Model([demo_input,diagnosis_input,script_input,ltc_input], outputs, name="linear_model")
linear_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])


early_stop = EarlyStopping(monitor='loss', patience=5)

history = linear_model.fit(x=original_train, y=label_train, batch_size=64,validation_split=0.10,  epochs=200, callbacks=early_stop)
test_predictions = linear_model.predict([demo_test,diag_test,scripts_test,LTC_test])
test_predictions_binary = (test_predictions > 0.5).astype(int)


metrics_={
"Test Recall": recall_score(label_test, test_predictions_binary,average=None),
"Test F1 Score": f1_score(label_test, test_predictions_binary,average=None),
"Test Precision": precision_score(label_test, test_predictions_binary,average=None)
# "AUROC": roc_auc_score(label_test, test_predictions_binary)
}
print(metrics_)



print("metrics ;)")
preds = linear_model.predict([demo_test,diag_test,scripts_test,LTC_test])
# pred_proba = linear_model.predict_classes(test_inputs)

#y_hat = np.argmax(preds,axis=1)
#test_labels= np.argmax(np.asarray(test_labels),axis=1)
y_hat= list(itertools.chain(*preds))
#print(preds)


y_hat = [round(x) for x in y_hat]
#print(y_hat)
unique, counts = np.unique(label_test, return_counts=True)
print(np.asarray((unique, counts)).T)
#print(y_hat)
precision, recall, fscore, support = score(label_test, y_hat)
scor = pd.DataFrame({'Actual_label':sorted(set(label_test)),'precision':precision,'recall':recall,'fscore':fscore})
total=[np.mean(recall),np.mean(precision),np.mean(fscore)]
print(scor)
print('\ntotal',total)
#y_hat = linear_model.predict_classes(test_inputs,varbose=0)
#accura= accuracy_score(test_labels,y_hat)
#precision= precision_score(test_labels,y_hat)
#f1_sco = f1_score(test_labels,y_hat)
#auc = roc_auc_score(test_labels,y_hat)
#print("accuracy: ", accura,"\n","precision: ", precision,"\n","f1_sco: ",f1_sco, "\nauc: ",auc)

probs = []
train_probs = []
final_preds = []
train_final_preds = []
for pred in preds:
    probs.append(pred[0])

    if pred[0] < 0.5:
        final_preds.append(0)
    else:
        final_preds.append(1)


# print(f'Test AUPRC: {precision_recall_curve(label_test, probs)}')
print(f'Test AUROC: {roc_auc_score(label_test, probs)}')

# print(f'Test AUPRC: {average_precision_score(label_test.values, probs)}')
# print(precision_recall_fscore_support(test_labels, final_preds, average='binary', pos_label=1))
#skplt.metrics.plot_precision_recall_curve(test_labels, preds)
#skplt.metrics.plot_roc(test_labels, preds)
#plt.show()
precis, reca, thresholds = precision_recall_curve(label_test, probs)
#Use AUC function to calculate the area under the curve of precision recall curve
auc_precision_recall = auc(reca, precis)

tp = 0.0
tn = 0.0
fp = 0.0
fn = 0.0
precision = 0
recall = 0

for pred, label in zip(final_preds, label_test):
    if pred >= 0.5 and label == 1.0:
        tp += 1.0
    elif pred >= 0.5 and label == 0.0:
        fp += 1.0
    elif pred <= 0.5 and label == 1.0:
        fn += 1.0
    else:
        tn += 1.0

if (tp + fp) > 0:
    precision = tp / (tp + fp)
else:
    precision = 0.0
if (tp + fn) > 0:
    recall = tp / (tp + fn)
else:
    recall = 0.0
if (precision + recall) > 0:
    f1_score = 2 * ((precision * recall) / (precision + recall))
else:
    f1_score = 0.0

# print(tp, tn, fp, fn)

# print(f"Test Precision: {precision}\nTest Recall: {recall}\nTest F1 Score: {f1_score}")


plt.figure(0)

print("AUPRC test result",auc_precision_recall)
plt.plot(reca, precis,label='AUPRC = '+str(round(auc_precision_recall,3)),color='red')

plt.xlabel("Recall",fontsize=14)
plt.ylabel("Precision",fontsize=14)
plt.legend(fontsize=15)
plt.savefig("AUPRC")
plt.show()
plt.figure(1)

fpr, tpr, _ = metrics.roc_curve(label_test, probs)
aucroc_ = metrics.roc_auc_score(label_test, probs)
plt.plot(fpr,tpr,label="AUC ROC="+str(round(aucroc_,3)),color='orange')
plt.legend(loc=4,fontsize=15)

plt.xlabel("True Positive Rate",fontsize=15)
plt.ylabel("False Positive Rate",fontsize=15)
plt.savefig("AUC ROC")
plt.show()