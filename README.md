# 📰 News Headline Classification using BERT

## 📌 Objective
The goal of this task is to **classify news headlines** into predefined categories using **state-of-the-art BERT transformer models**.  
The task demonstrates how **pre-trained language models** can be fine-tuned on domain-specific data to achieve high accuracy and F1-score for text classification tasks.



## 🧠 Purpose & Advantages
Text classification is one of the most common **Natural Language Processing (NLP)** tasks.  
Using BERT offers several advantages:

1. **Contextual Understanding**: BERT captures word meanings based on context, unlike traditional bag-of-words models.
2. **High Accuracy**: Pre-trained models accelerate training and improve predictive performance.
3. **Flexibility**: Can classify headlines into multiple categories with minimal additional training.
4. **Integration**: Works seamlessly with frameworks like Hugging Face Transformers and Gradio for web demos.



## 📂 Dataset
- Source: Preprocessed news headlines dataset (train/test split generated)  
- Input: `headline` text  
- Target: `category` label  
- Training Examples: 120,000  
- Testing Examples: 7,600  


## ⚙️ Methodology / Approach

1. **Preprocessing**
   - Load dataset and split into train/test sets
   - Tokenize headlines using `BertTokenizer`
   - Prepare datasets for Hugging Face `Trainer`

```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

tokenized_train = train_dataset.map(tokenize, batched=True)
```
## Model Setup

Model: BertForSequenceClassification

Fine-tune on the news dataset

Some weights initialized randomly for the classifier head
```
from transformers import BertForSequenceClassification
from transformers import Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer
)
```
## Training

Fine-tune for 2 epochs

Monitor Training Loss, Validation Loss, Accuracy, and F1 Score
```
Epoch    Training Loss    Validation Loss    Accuracy    F1
1        0.1918           0.1762             0.9455     0.9455
2        0.1238           0.1920             0.9480     0.9480
```
## Evaluation

Metrics: Accuracy and F1 Score

Achieved high accuracy: 94–95%

Achieved high F1-score: 0.945–0.948

## 🚀 Gradio Web App Demo

Deploy the model as an interactive web demo using Gradio

Users can enter headlines and instantly classify them
```
import gradio as gr

def classify_headline(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_label = outputs.logits.argmax().item()
    return category_names[predicted_label]

gr.Interface(
    fn=classify_headline,
    inputs="text",
    outputs="label",
    title="📰 News Topic Classifier (BERT)",
    description="Enter a news headline to classify it"
).launch()
```
<img width="872" height="436" alt="image" src="https://github.com/user-attachments/assets/95e9a78e-a89d-423f-80ab-b497c7605bad" />

## 💡 Key Insights

Fine-tuning pre-trained transformers yields highly accurate text classification.

BERT can understand contextual nuances in headlines better than traditional ML models.

Integration with Gradio allows instant deployment and testing.

The pipeline is extendable to any text classification task (tweets, reviews, articles).

## ⚙️ How to Use

1.Clone the Repository
```
git clone https://github.com/your-username/bert-news-classification.git
cd bert-news-classification
```

2.Install Dependencies
```
pip install -r requirements.txt
```

3.Run Notebook
```
jupyter notebook notebooks/bert_news_classification.py
```

OR Run Python Script
```
python bert_news_classification.py
```
