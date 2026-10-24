import json
import re
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

torch.manual_seed(42)

# ============================================================
# Same topology names used in the other multitopology scripts.
# ============================================================
TOPOLOGY_PATHS = {
    "us_ieee118": "data/processed/us_ieee118",
    "uk_ieee30": "data/processed/uk_ieee30",
    "fr_pegase89": "data/processed/fr_pegase89",
    "gb_full": "data/processed/gb_full",
    "eu_case300": "data/processed/eu_case300",
    "india_synthetic": "data/processed/india_synthetic",
}

MAX_LEN = 30
EPOCHS = 20


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    return tokens


def text_to_sequence(text, vocab, max_len=MAX_LEN):
    tokens = tokenize(text)
    seq = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    seq = seq[:max_len]
    seq = seq + [vocab["<PAD>"]] * (max_len - len(seq))
    return seq


def extract_structured_features(scenario):
    weather = scenario["weather"]
    event = scenario["event"]
    total_wind = sum(v["wind_mw"] for v in scenario["renewable_output"].values())
    total_solar = sum(v["solar_mw"] for v in scenario["renewable_output"].values())
    return [
        weather["wind_speed_kmh"] / 90.0,
        weather["solar_irradiance_wm2"] / 1000.0,
        int(weather["storm_flag"]),
        (scenario["load_multiplier"] - 0.8) / 1.4,
        total_wind / 240.0,
        total_solar / 160.0,
        1 if event["type"] == "line_outage" else 0,
        1 if event.get("cause") == "storm" else 0,
        1 if event.get("cause") == "equipment_failure" else 0,
    ]


class ScenarioDataset(Dataset):
    def __init__(self, data, vocab):
        self.structured = [extract_structured_features(s) for s in data]
        self.text_seqs = [text_to_sequence(s["text_report"], vocab) for s in data]
        self.labels = [int(s["congested"]) for s in data]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.structured[idx], dtype=torch.float32),
            torch.tensor(self.text_seqs[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class MultiModalCongestionModel(nn.Module):
    def __init__(self, vocab_size, struct_dim=9, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.struct_branch = nn.Sequential(
            nn.Linear(struct_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, structured, text_seq):
        struct_out = self.struct_branch(structured)
        embedded = self.embedding(text_seq)
        _, (hidden, _) = self.text_lstm(embedded)
        text_out = hidden.squeeze(0)
        combined = torch.cat([struct_out, text_out], dim=1)
        return self.fusion(combined).squeeze(1)


def train_multimodal_for_topology(topology_name):
    base_dir = TOPOLOGY_PATHS[topology_name]
    train_path = f"{base_dir}/train_multimodal.jsonl"
    val_path = f"{base_dir}/val_multimodal.jsonl"
    test_path = f"{base_dir}/test_multimodal.jsonl"
    vocab_out_path = f"{base_dir}/vocab.json"
    model_out_path = f"app/model2_multimodal_{topology_name}.pt"
    model_vocab_out_path = f"app/vocab_{topology_name}.json"
    results_out_path = f"experiments/model2_multimodal_results_{topology_name}.txt"

    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path)
    test_data = load_jsonl(test_path)

    print(f"[{topology_name}] Train: {len(train_data)} scenarios")
    print(f"[{topology_name}] Val:   {len(val_data)} scenarios")
    print(f"[{topology_name}] Test:  {len(test_data)} scenarios")

    word_counts = Counter()
    for scenario in train_data:
        tokens = tokenize(scenario["text_report"])
        word_counts.update(tokens)

    print(f"[{topology_name}] Total unique words found: {len(word_counts)}")

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in word_counts.most_common():
        vocab[word] = len(vocab)

    print(f"[{topology_name}] Vocabulary size: {len(vocab)}")

    with open(vocab_out_path, "w") as f:
        json.dump(vocab, f)

    train_dataset = ScenarioDataset(train_data, vocab)
    val_dataset = ScenarioDataset(val_data, vocab)
    test_dataset = ScenarioDataset(test_data, vocab)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"[{topology_name}] DataLoaders ready. Train batches: {len(train_loader)}")

    model = MultiModalCongestionModel(vocab_size=len(vocab))
    print(f"\n[{topology_name}] Model architecture:\n{model}")

    pos_weight = torch.tensor([len(train_data) / (2 * sum(1 for s in train_data if s["congested"]))])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\n[{topology_name}] --- Training for {EPOCHS} epochs ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for structured, text_seq, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(structured, text_seq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_preds_list, val_labels_list = [], []
        with torch.no_grad():
            for structured, text_seq, labels in val_loader:
                outputs = model(structured, text_seq)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_preds_list.extend(preds.tolist())
                val_labels_list.extend(labels.tolist())
        val_acc = accuracy_score(val_labels_list, val_preds_list)
        print(f"[{topology_name}] Epoch {epoch+1}/{EPOCHS} - Train Loss: {total_loss/len(train_loader):.4f} - Val Accuracy: {val_acc:.3f}")

    torch.save(model.state_dict(), model_out_path)
    with open(model_vocab_out_path, "w") as f:
        json.dump(vocab, f)
    print(f"[{topology_name}] Model 2 saved to {model_out_path}")

    model.eval()
    test_preds_list, test_labels_list = [], []
    with torch.no_grad():
        for structured, text_seq, labels in test_loader:
            outputs = model(structured, text_seq)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            test_preds_list.extend(preds.tolist())
            test_labels_list.extend(labels.tolist())

    acc = accuracy_score(test_labels_list, test_preds_list)
    prec = precision_score(test_labels_list, test_preds_list)
    rec = recall_score(test_labels_list, test_preds_list)
    f1 = f1_score(test_labels_list, test_preds_list)
    cm = confusion_matrix(test_labels_list, test_preds_list)

    print(f"\n[{topology_name}] --- Test Set Performance (FINAL) ---")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"\n[{topology_name}] Confusion Matrix (Test Set):")
    print(f"                 Predicted: No   Predicted: Yes")
    print(f"Actual: No       {cm[0][0]:>13}   {cm[0][1]:>14}")
    print(f"Actual: Yes      {cm[1][0]:>13}   {cm[1][1]:>14}")

    results_text = f"""
MODEL 2: MULTI-MODAL (Structured + Text) — LSTM + Fusion
Topology: {topology_name}
Generated from simulation, {len(train_data)+len(val_data)+len(test_data)} total scenarios

--- Test Set Performance (FINAL) ---
Accuracy:  {acc:.3f}
Precision: {prec:.3f}
Recall:    {rec:.3f}
F1 Score:  {f1:.3f}

Confusion Matrix (Test Set):
                 Predicted: No   Predicted: Yes
Actual: No       {cm[0][0]:>13}   {cm[0][1]:>14}
Actual: Yes      {cm[1][0]:>13}   {cm[1][1]:>14}
"""

    with open(results_out_path, "w") as f:
        f.write(results_text)

    print(f"\n[{topology_name}] Results saved to {results_out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        topology = sys.argv[1]
    else:
        topology = "uk_ieee30"
    train_multimodal_for_topology(topology)