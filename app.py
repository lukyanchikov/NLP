from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer

app = Flask(__name__)

model1_nm = 'bert-base-uncased'
tokz1 = AutoTokenizer.from_pretrained(model1_nm)
tokz1.model_max_length = 512

model2_nm = 'bert-base-cased'
tokz2 = AutoTokenizer.from_pretrained(model2_nm)
tokz2.model_max_length = 512

def tok1_func(x):
    return tokz1(x["input"],truncation=True)
    
def tok2_func(x):
    return tokz2(x["input"],truncation=True)

def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)
    acc = balanced_accuracy_score(labels,predictions)
    f1 = f1_score(labels, predictions, average = 'micro')
    return {'accuracy': acc, 'f1_score': f1}

@app.route('/predict', methods=['POST'])
def predict():
    review = request.get_json()

    review = pd.DataFrame.from_dict(review, orient='columns')
    
    review['input'] = review['review']
    review['sentiment'] = 'unknowns'
    review['labels'] = 0.5
    review.drop(columns=['sentiment','review','labels'],inplace=True)
    
    review_pred = Dataset.from_pandas(review)

    tok1_pred_ds = review_pred.map(tok1_func, batched=True)
    dds1_pred = tok1_pred_ds

    tok2_pred_ds = review_pred.map(tok2_func, batched=True)
    dds2_pred = tok2_pred_ds
    
    model1_pred = AutoModelForSequenceClassification.from_pretrained(r"C:\Users\serge\eclipse-workspace\NLP\bert-base-uncased")
    trainer1_pred = Trainer(model1_pred,tokenizer=tokz1)

    model2_pred = AutoModelForSequenceClassification.from_pretrained(r"C:\Users\serge\eclipse-workspace\NLP\bert-base-cased")
    trainer2_pred = Trainer(model2_pred,tokenizer=tokz2)
    
    sentiment1 = trainer1_pred.predict(dds1_pred)
    sentiment1_lbl = sentiment1.predictions.argmax(-1)

    sentiment2 = trainer2_pred.predict(dds2_pred)
    sentiment2_lbl = sentiment2.predictions.argmax(-1)
    
    response = [{"bertbaseuncased": int(sentiment1_lbl[0]), "bertbasecased": int(sentiment2_lbl[0])}]

    return jsonify(data=response)

if __name__ == '__main__':
    app.run(debug=True)