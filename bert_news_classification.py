# ============================================================
# TASK 1: News Topic Classifier using BERT (FIXED + SINGLE CELL)
# ============================================================

# -----------------------------
# 2️⃣ IMPORT LIBRARIES
# -----------------------------
import torch
import numpy as np

from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import accuracy_score, f1_score
import gradio as gr

# -----------------------------
# 3️⃣ LOAD DATASET
# -----------------------------
dataset = load_dataset("ag_news")

# -----------------------------
# 4️⃣ TOKENIZER
# -----------------------------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize_data(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

encoded_dataset = dataset.map(tokenize_data, batched=True)
encoded_dataset = encoded_dataset.remove_columns(["text"])
encoded_dataset.set_format("torch")

# -----------------------------
# 5️⃣ MODEL
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
)

# -----------------------------
# 6️⃣ METRICS
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# -----------------------------
# 7️⃣ TRAINING ARGUMENTS (FIXED)
# -----------------------------
training_args = TrainingArguments(
    output_dir="./bert_news",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"   # 🔥 THIS STOPS W&B PROMPT
)


# -----------------------------
# 8️⃣ TRAINER
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# -----------------------------
# 9️⃣ TRAIN
# -----------------------------
trainer.train()

# -----------------------------
# 🔟 EVALUATE
# -----------------------------
trainer.evaluate()

# -----------------------------
# 1️⃣1️⃣ SAVE MODEL
# -----------------------------
model.save_pretrained("news_bert_model")
tokenizer.save_pretrained("news_bert_model")

# -----------------------------
# 1️⃣2️⃣ GRADIO APP
# -----------------------------
label_map = {
    0: "World 🌍",
    1: "Sports 🏀",
    2: "Business 💼",
    3: "Sci/Tech 🔬"
}

def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return label_map[torch.argmax(outputs.logits).item()]

gr.Interface(
    fn=predict_news,
    inputs="text",
    outputs="label",
    title="📰 News Topic Classifier (BERT)",
    description="Enter a news headline and classify it"
).launch()
