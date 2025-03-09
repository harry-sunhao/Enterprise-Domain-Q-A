import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback, AdamW
)
class QADataset(Dataset):
    def __init__(self, questions, tokenizer, max_length=512):
        self.questions = questions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]["QUESTION_TITLE"]
        answer_text = self.questions[idx]["ANSWER"]
        context = self.questions[idx]["QUESTION_TEXT"]
        if answer_text == '-':
            answer_start = -1
            answer_end = -1
        else:
            answer_start = int(self.questions[idx]["START_OFFSET"])
            answer_end = int(self.questions[idx]["END_OFFSET"])
        # Tokenize question and context
        inputs = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Prepare data
        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "start_positions": torch.tensor(answer_start, dtype=torch.long),
            "end_positions": torch.tensor(answer_end, dtype=torch.long)
        }
        return item

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)

DATA_PATH = os.path.join(parent_directory, "data")

with open(os.path.join(DATA_PATH, "training_Q_A.json"), "r", encoding="utf-8") as f:
    questions = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

train_dataset = QADataset(questions, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

model = AutoModelForQuestionAnswering.from_pretrained("microsoft/deberta-v3-large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

def train_model(model, train_loader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        loop = tqdm(train_loader, leave=True)

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            valid_indices = (start_positions != -1) & (end_positions != -1)
            if valid_indices.sum() == 0:
                continue
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item())

train_model(model, train_loader, optimizer, epochs=3)

model.save_pretrained("./fine_tuned_deberta")
tokenizer.save_pretrained("./fine_tuned_deberta")


