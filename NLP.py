import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from datasets import Dataset
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
import matplotlib.pyplot as plt
import seaborn as sn

# Common steps and functions

model1_nm = 'bert-base-uncased' # pre-trained model name
tokz1 = AutoTokenizer.from_pretrained(model1_nm) # tokenizer based on pre-trained model
tokz1.model_max_length = 512 # max length (in tokens) of model inputs

model2_nm = 'bert-base-cased'
tokz2 = AutoTokenizer.from_pretrained(model2_nm)
tokz2.model_max_length = 512

# - functions to apply tokenizers to inputs (truncation to max length)

def tok1_func(x):
    return tokz1(x["input"],truncation=True)

def tok2_func(x):
    return tokz2(x["input"],truncation=True)

# - evaluation metrics

def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)
    acc = balanced_accuracy_score(labels,predictions) # computes the balanced accuracy score
    f1 = f1_score(labels, predictions, average = 'micro') # computes F1 score
    return {'accuracy': acc, 'f1_score': f1}

# Training/testing

# - prepare data

df = pd.read_csv(r"C:\Users\serge\eclipse-workspace\NLP\IMDB-trainvalidate.csv",sep=",",encoding='latin-1')
df.head()

# - cleaning the data

df['input'] = df['review']
df.loc[df['sentiment']=='positive','labels'] = 1
df.loc[df['sentiment']=='negative','labels'] = 0
df['labels'] = df['labels'].astype('int')
df.drop(columns=['sentiment','review'],inplace=True)
df.dropna(inplace=True)
df = df.sample(frac=0.1)
df.head()

# - visualizing the data distribution

df_pie = df['labels'].value_counts()
df_pie.head()
labels = ['Positive','Negative']
plt.figure()
plt.pie(df_pie, labels=labels, autopct='%1.1f%%')
plt.title('IMDB reviews (sentiment category distribution)')
plt.show()

# - tokenization

ds = Dataset.from_pandas(df)

tok1_ds = ds.map(tok1_func, batched=True)
dds1 = tok1_ds.train_test_split(0.2, seed=77)

tok2_ds = ds.map(tok2_func, batched=True)
dds2 = tok2_ds.train_test_split(0.2, seed=77)

# - train

torch.cuda.is_available()

bs = 16 # Batch size
epochs = 5 # number of epochs
lr = 2e-5 # Learning rate
num_class = 2 # Number of classes of sentiments (positive and negative)

args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs,
    num_train_epochs=epochs, weight_decay=0.01, report_to='none')

model1 = AutoModelForSequenceClassification.from_pretrained(model1_nm, num_labels=num_class)

model2 = AutoModelForSequenceClassification.from_pretrained(model2_nm, num_labels=num_class)

trainer1 = Trainer(model1, args, train_dataset=dds1['train'], eval_dataset=dds1['test'],
                  tokenizer=tokz1, compute_metrics=compute_metrics)

trainer2 = Trainer(model2, args, train_dataset=dds2['train'], eval_dataset=dds2['test'],
                  tokenizer=tokz2, compute_metrics=compute_metrics)

trainer1.train()

trainer2.train()

# - test

y1 = trainer1.predict(dds1['test'])
y1_lblz = y1.predictions.argmax(-1)

y2 = trainer2.predict(dds2['test'])
y2_lblz = y2.predictions.argmax(-1)

# - save the models

trainer1.save_model(r"C:\Users\serge\eclipse-workspace\NLP\bert-base-uncased")

trainer2.save_model(r"C:\Users\serge\eclipse-workspace\NLP\bert-base-cased")

# Predicting

df_pred = pd.read_csv(r"C:\Users\serge\eclipse-workspace\NLP\IMDB-test.csv",sep=";",encoding='latin-1')
df_pred.head()

# - cleaning the data

df_pred['input'] = df_pred['review']
df_pred.loc[df_pred['sentiment']=='positive','labels'] = 1
df_pred.loc[df_pred['sentiment']=='negative','labels'] = 0
df_pred['labels'] = df_pred['labels'].astype('int')

# - visualizing the data distribution

df_pred_pie = df_pred['labels'].value_counts()
df_pred_pie.head()
labels = ['Positive','Negative']
plt.figure()
plt.pie(df_pred_pie, labels=labels, autopct='%1.1f%%')
plt.title('IMDB reviews (sentiment category distribution)')
plt.show()

# - tokenization

df_labels=np.array(df_pred['labels'])

df_pred.drop(columns=['sentiment','review','labels'],inplace=True)
df_pred.dropna(inplace=True)
df_pred.head()

ds_pred = Dataset.from_pandas(df_pred)

tok1_pred_ds = ds_pred.map(tok1_func, batched=True)
dds1_pred = tok1_pred_ds

tok2_pred_ds = ds_pred.map(tok2_func, batched=True)
dds2_pred = tok2_pred_ds

# - loading a trained model

model1_pred = AutoModelForSequenceClassification.from_pretrained(r"C:\Users\serge\eclipse-workspace\NLP\bert-base-uncased")
trainer1_pred = Trainer(model1_pred,tokenizer=tokz1)

model2_pred = AutoModelForSequenceClassification.from_pretrained(r"C:\Users\serge\eclipse-workspace\NLP\bert-base-cased")
trainer2_pred = Trainer(model2_pred,tokenizer=tokz2)

# - predict

y1_pred = trainer1_pred.predict(dds1_pred)
y1_pred_lblz = y1_pred.predictions.argmax(-1)

y2_pred = trainer2_pred.predict(dds2_pred)
y2_pred_lblz = y2_pred.predictions.argmax(-1)

# - results

df_res=df_pred
df_res['bertbaseuncased']=y1_pred_lblz
df_res['bertbasecased']=y2_pred_lblz
df_res.to_csv(r"C:\Users\serge\eclipse-workspace\NLP\IMDB-results.csv", sep=";") # a .csv file with results

cm1 = confusion_matrix(df_labels, y1_pred_lblz)

Confusion1=pd.DataFrame(np.array((cm1[1][1],cm1[0][1],cm1[0][0],cm1[1][0]))).T
Confusion1.columns=['TP','FN','TN','FP']
Confusion1['ACC']=(Confusion1['TP']+Confusion1['TN'])/(Confusion1['TP']+Confusion1['TN']+Confusion1['FP']+Confusion1['FN']) # accuracy
Confusion1['TPR']=(Confusion1['TP'])/(Confusion1['TP']+Confusion1['FN']) # true positive rate
Confusion1['TNR']=(Confusion1['TN'])/(Confusion1['TN']+Confusion1['FP']) # true negative rate
Confusion1['PPV']=(Confusion1['TP'])/(Confusion1['TP']+Confusion1['FP']) # positive predictive value
Confusion1['NPV']=(Confusion1['TN'])/(Confusion1['TN']+Confusion1['FN']) # negative predictive value

plt.figure()
sn.heatmap(cm1, annot=True)
plt.xlabel('Predicted (0 - non-target, 1 - target)')
plt.ylabel('Actual (1 - target, 0 - non-target)')
plt.title('CM: bertbaseuncased ACC='+str(round(Confusion1['ACC'][0],2))+', TPR='+str(round(Confusion1['TPR'][0],2))+', TNR='+str(round(Confusion1['TNR'][0],2)))
plt.show()

cm2 = confusion_matrix(df_labels, y2_pred_lblz)

Confusion2=pd.DataFrame(np.array((cm2[1][1],cm2[0][1],cm2[0][0],cm2[1][0]))).T
Confusion2.columns=['TP','FN','TN','FP']
Confusion2['ACC']=(Confusion2['TP']+Confusion2['TN'])/(Confusion2['TP']+Confusion2['TN']+Confusion2['FP']+Confusion2['FN']) # accuracy
Confusion2['TPR']=(Confusion2['TP'])/(Confusion2['TP']+Confusion2['FN']) # true positive rate
Confusion2['TNR']=(Confusion2['TN'])/(Confusion2['TN']+Confusion2['FP']) # true negative rate
Confusion2['PPV']=(Confusion2['TP'])/(Confusion2['TP']+Confusion2['FP']) # positive predictive value
Confusion2['NPV']=(Confusion2['TN'])/(Confusion2['TN']+Confusion2['FN']) # negative predictive value

plt.figure()
sn.heatmap(cm2, annot=True)
plt.xlabel('Predicted (0 - non-target, 1 - target)')
plt.ylabel('Actual (1 - target, 0 - non-target)')
plt.title('CM: bertbasecased ACC='+str(round(Confusion2['ACC'][0],2))+', TPR='+str(round(Confusion2['TPR'][0],2))+', TNR='+str(round(Confusion2['TNR'][0],2)))
plt.show()