import copy
import math
import random
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

def get_ffn_layers(model):
    if hasattr(model, "distilbert"):
        layer0 = model.distilbert.transformer.layer[0]
        lin1 = layer0.ffn.lin1
        lin2 = layer0.ffn.lin2
    elif hasattr(model, "bert"):
        layer0 = model.bert.encoder.layer[0]
        lin1 = layer0.intermediate.dense
        lin2 = layer0.output.dense
    else:
        raise AttributeError("Unsupported model architecture for FFN extraction.")
    return lin1, lin2
def get_ffn1_weight(model):
    lin1, _ = get_ffn_layers(model)
    return lin1.weight.detach().cpu()

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_dataset_sst2(tokenizer, max_length: int = 128, n_train: int = 2000, n_test: int = 1000):
    try:
        raw = load_dataset("glue", "sst2")
    except Exception:
        def synthetic_split(n):
            texts, labels = [], []
            pos_tokens = ["great", "good", "excellent", "wonderful", "amazing"]
            neg_tokens = ["bad", "awful", "terrible", "poor", "boring"]
            for _ in range(n):
                y = np.random.randint(0, 2)
                token = random.choice(pos_tokens if y == 1 else neg_tokens)
                texts.append(f"This movie is {token}.")
                labels.append(int(y))
            return {"sentence": texts, "label": labels}

        raw = {"train": synthetic_split(n_train),"validation": synthetic_split(n_test),}

    def tokenize(example):
        return tokenizer(
            example["sentence"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )
    train_ds = raw["train"].select(range(min(n_train, len(raw["train"]))))
    test_ds = raw["validation"].select(range(min(n_test, len(raw["validation"]))))

    tokenized_train = train_ds.map(tokenize, batched=True)
    tokenized_test = test_ds.map(tokenize, batched=True)

    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(tokenized_train, batch_size=16, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(tokenized_test, batch_size=32, shuffle=False, collate_fn=collator)

    return (tokenized_train, tokenized_test), (train_loader, test_loader)

def build_model(model_name: str = "google/bert_uncased_L-2_H-128_A-2",):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return tokenizer, model


def evaluate_model(model, dataloader, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if "labels" in batch:
                labels = batch.pop("labels")
            elif "label" in batch:
                labels = batch.pop("label")
            else:
                raise KeyError("Expected 'label' or 'labels' in batch")
            outputs = model(**batch)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return {"loss": total_loss / total, "accuracy": correct / total}

def fine_tune_model(model,train_loader,test_loader,device,epochs: int = 2,lr: float = 2e-5,optimizer_name: str = "adamw",
                    momentum: float = 0.9,freeze_encoder: bool = False,weight_decay: float = 0.0,):
    for p in model.parameters():
        p.requires_grad = True
    if freeze_encoder:
        if hasattr(model, "distilbert"):
            for p in model.distilbert.parameters():
                p.requires_grad = False
        elif hasattr(model, "bert"):
            for p in model.bert.parameters():
                p.requires_grad = False
    encoder_param_name = None
    if hasattr(model, "distilbert"):
        encoder_param_name = "distilbert.transformer.layer.0.ffn.lin1.weight"
    elif hasattr(model, "bert"):
        encoder_param_name = "bert.encoder.layer.0.intermediate.dense.weight"
    named_params = dict(model.named_parameters())

    assert (encoder_param_name in named_params), f"Could not find encoder parameter {encoder_param_name}"

    encoder_param = named_params[encoder_param_name]
    print(f"{encoder_param_name} requires_grad={encoder_param.requires_grad} (freeze={freeze_encoder})")

    model.to(device)
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    def _param_in_optimizer(opt, target_param):
        for group in opt.param_groups:
            for p in group["params"]:
                if p is target_param:
                    return True
        return False
    assert _param_in_optimizer(optimizer, encoder_param), f"{encoder_param_name} not found in optimizer param groups!"
    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, num_training_steps // 10), num_training_steps=num_training_steps)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if "labels" in batch:
                labels = batch.pop("labels")
            elif "label" in batch:
                labels = batch.pop("label")
            else:
                raise KeyError("Expected 'label' or 'labels' in batch")
            outputs = model(**batch)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        eval_stats = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs} - val_loss={eval_stats['loss']:.4f}, val_acc={eval_stats['accuracy']:.4f}")
    return model


# %%
def extract_embeddings(model, dataloader, device):
    model.to(device)
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            lbl = batch["labels"] if "labels" in batch else batch["label"]
            labels.append(lbl)
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k not in ("label", "labels")
            }
            outputs = model(**inputs, output_hidden_states=True)
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                hidden = outputs.last_hidden_state
            else:
                hidden = outputs.hidden_states[-1]
            cls_emb = hidden[:, 0, :].cpu()
            embeddings.append(cls_emb)
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    return embeddings, labels

class LinearLogReg(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.linear = nn.Linear(d_in, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def train_logreg(X_train, y_train, X_test, y_test, epochs: int = 50, lr: float = 1e-2, batch_size: int = 64):

    device = X_train.device
    model = LinearLogReg(X_train.size(1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    def run_epoch(X, y, train: bool):
        model.train() if train else model.eval()
        total_loss, correct, total = 0.0, 0, 0
        perm = torch.randperm(X.size(0))
        for i in range(0, X.size(0), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X[idx], y[idx]
            logits = model(xb)
            loss = loss_fn(logits, yb.float())
            if train:
                loss.backward()
                opt.step()
                opt.zero_grad()
            total_loss += loss.item() * yb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        return total_loss / total, correct / total

    for _ in range(epochs):
        run_epoch(X_train, y_train, train=True)
    test_loss, test_acc = run_epoch(X_test, y_test, train=False)
    return test_loss, test_acc


def random_projection_experiment(X_train, y_train, X_test, y_test, m_list: List[int]):
    results = {}
    d = X_train.size(1)
    X_train = X_train.to(torch.float32)
    X_test = X_test.to(torch.float32)
    y_train = y_train.to(torch.long)
    y_test = y_test.to(torch.long)
    device = X_train.device
    for m in m_list:
        m_eff = min(m, d)
        P = torch.randn(m_eff, d, device=device) / math.sqrt(m_eff)
        Xtr = X_train @ P.t()
        Xte = X_test @ P.t()
        loss, acc = train_logreg(Xtr, y_train, Xte, y_test, epochs=80, lr=5e-3)
        results[m_eff] = {"loss": loss, "accuracy": acc}
        print(f"Projection m={m_eff}: test_loss={loss:.4f}, test_acc={acc:.4f}")
    return results



def compute_min_norm_interpolant(X_train, y_train, X_test, y_test):
    Xtr = X_train.double()
    Xte = X_test.double()
    y_pm = (2 * y_train - 1).double()  
    y_pm_test = (2 * y_test - 1).double()
    pinv = torch.linalg.pinv(Xtr)
    w = pinv @ y_pm  
    margins_train = (Xtr @ w) * y_pm
    margins_test = (Xte @ w) * y_pm_test
    return w, margins_train.cpu().numpy(), margins_test.cpu().numpy()

def compute_spectral_norms(model) -> Dict[str, float]:
    with torch.no_grad():
        norms = {}
        W_cls = model.classifier.weight.detach().cpu()
        norms["classifier"] = torch.linalg.svdvals(W_cls).max().item()
        if hasattr(model, "distilbert"):
            layer0 = model.distilbert.transformer.layer[0]
            norms["ffn_lin1"] = torch.linalg.svdvals(layer0.ffn.lin1.weight.detach().cpu()).max().item()
            norms["ffn_lin2"] = torch.linalg.svdvals(layer0.ffn.lin2.weight.detach().cpu()).max().item()
        elif hasattr(model, "bert"):
            layer0 = model.bert.encoder.layer[0]
            norms["ffn_lin1"] = torch.linalg.svdvals(layer0.intermediate.dense.weight.detach().cpu()).max().item()
            norms["ffn_lin2"] = torch.linalg.svdvals(layer0.output.dense.weight.detach().cpu()).max().item()
    return norms


def compute_singular_values(model, top_k: int = 100) -> Dict[str, np.ndarray]:
    spectra = {}
    with torch.no_grad():
        w1, w2 = get_ffn_layers(model)
        svals_lin1 = torch.linalg.svdvals(w1.weight.detach().cpu()).numpy()
        svals_lin2 = torch.linalg.svdvals(w2.weight.detach().cpu()).numpy()
        spectra["ffn_lin1"] = svals_lin1[: min(top_k, len(svals_lin1))]
        spectra["ffn_lin2"] = svals_lin2[: min(top_k, len(svals_lin2))]
    return spectra


def perturbation_experiment(model, dataloader_test, device, sigma_list: List[float]):
    base_weight = model.classifier.weight.data.clone()
    results = {}
    for sigma in sigma_list:
        with torch.no_grad():
            model.classifier.weight.data = base_weight + sigma * torch.randn_like(base_weight)
        stats = evaluate_model(model, dataloader_test, device)
        results[sigma] = stats["accuracy"]
        print(f"Noise sigma={sigma:.3f}: test_acc={stats['accuracy']:.4f}")
    with torch.no_grad():
        model.classifier.weight.data = base_weight  # restore
    return results

def plot_projection_results(results: Dict[int, Dict[str, float]], fname: str = "fig_projection_accuracy.png"):
    dims = sorted(results.keys())
    accs = [results[m]["accuracy"] for m in dims]
    losses = [results[m]["loss"] for m in dims]
    plt.figure(figsize=(6, 4))
    plt.plot(dims, accs, marker="o", label="Test accuracy")
    plt.plot(dims, losses, marker="s", label="Test loss")
    plt.xlabel("Projection dimension m")
    plt.ylabel("Metric")
    plt.title("Random projection performance")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_margin_hist(margins_train, margins_test, fname: str = "fig_margin_hist.png"):
    plt.figure(figsize=(6, 4))
    plt.hist(margins_train, bins=40, alpha=0.6, label="Train margins")
    plt.hist(margins_test, bins=40, alpha=0.6, label="Test margins")
    plt.xlabel("Margin")
    plt.ylabel("Count")
    plt.title("Minimum-norm interpolant margins")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_spectral_norms(pre: Dict[str, float], post: Dict[str, float], fname: str = "fig_spectral_norms.png"):
    labels = list(pre.keys())
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, [pre[k] for k in labels], width, label="Pre")
    plt.bar(x + width / 2, [post[k] for k in labels], width, label="Post")
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Spectral norm")
    plt.title("Spectral norms before/after fine-tuning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_noise_robustness(results: Dict[float, float], fname: str = "fig_noise_robustness.png"):
    sigmas = sorted(results.keys())
    accs = [results[s] for s in sigmas]
    plt.figure(figsize=(6, 4))
    plt.plot(sigmas, accs, marker="o")
    plt.xlabel("Noise level sigma")
    plt.ylabel("Test accuracy")
    plt.title("Classifier weight noise robustness")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_spectral_norms_triplet(base: Dict[str, float],adam: Dict[str, float],sgd: Dict[str, float],fname: str = "fig_spectral_norms_opt.png",):
    labels = list(base.keys())
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(7, 4))
    plt.bar(x - width, [base[k] for k in labels], width, label="Pretrain")
    plt.bar(x, [adam[k] for k in labels], width, label="Adam")
    plt.bar(x + width, [sgd[k] for k in labels], width, label="SGD")
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Spectral norm")
    plt.title("Spectral norms: optimizer comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def plot_singular_value_spectra(
    base_spec: Dict[str, np.ndarray],
    adam_spec: Dict[str, np.ndarray],
    sgd_spec: Dict[str, np.ndarray],
    fname: str = "fig_spectral_spectra.png",
):
    plt.figure(figsize=(7, 4))
    max_len = min(len(base_spec["ffn_lin1"]), len(adam_spec["ffn_lin1"]), len(sgd_spec["ffn_lin1"]))
    idx = np.arange(max_len)
    plt.plot(idx, base_spec["ffn_lin1"][:max_len], label="Pretrain FFN1")
    plt.plot(idx, adam_spec["ffn_lin1"][:max_len], label="Adam FFN1")
    plt.plot(idx, sgd_spec["ffn_lin1"][:max_len], label="SGD FFN1")
    print("Top singular values (first 5):")
    print("Pre:", base_spec["ffn_lin1"][:5])
    print("Adam:", adam_spec["ffn_lin1"][:5])
    print("SGD:", sgd_spec["ffn_lin1"][:5])
    plt.xlabel("Index (sorted singular values)")
    plt.ylabel("Singular value")
    plt.title("FFN lin1 singular spectra (top 100)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_top5_singular_values(base_spec: Dict[str, np.ndarray],adam_spec: Dict[str, np.ndarray],sgd_spec: Dict[str, np.ndarray],fname: str = "fig_spectral_top5.png",):
    max_len = min(5, len(base_spec["ffn_lin1"]), len(adam_spec["ffn_lin1"]), len(sgd_spec["ffn_lin1"]))
    labels = [f"k={i+1}" for i in range(max_len)]
    x = np.arange(max_len)
    width = 0.25
    plt.figure(figsize=(7, 4))
    plt.bar(x - width, base_spec["ffn_lin1"][:max_len], width, label="Pretrain")
    plt.bar(x, adam_spec["ffn_lin1"][:max_len], width, label="Adam")
    plt.bar(x + width, sgd_spec["ffn_lin1"][:max_len], width, label="SGD")
    plt.xlabel("Singular value index")
    plt.ylabel("Value (log scale)")
    plt.yscale("log")
    plt.title("Top-5 FFN lin1 singular values")
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def analyze_dataset(raw_train, raw_val, tokenizer, fname_prefix: str = "fig_data"):
    train_py = raw_train.with_format("python")
    val_py = raw_val.with_format("python")
    # Label distribution
    labels_train = [int(x) for x in train_py["label"]]
    labels_val = [int(x) for x in val_py["label"]]
    plt.figure(figsize=(5, 4))
    plt.hist(labels_train, bins=[-0.5, 0.5, 1.5], alpha=0.7, label="Train")
    plt.hist(labels_val, bins=[-0.5, 0.5, 1.5], alpha=0.7, label="Validation")
    plt.xticks([0, 1], ["neg", "pos"])
    plt.ylabel("Count")
    plt.title("Label distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fname_prefix}_labels.png")
    plt.close()

    def lengths(split):
        if "sentence" in split.column_names:
            return [len(tokenizer(s, truncation=True)["input_ids"]) for s in split["sentence"]]
        elif "input_ids" in split.column_names:
            ids = split["input_ids"]
            return [len(seq) if not torch.is_tensor(seq) else seq.numel() for seq in ids]
        else:
            return []

    len_train = lengths(train_py)
    len_val = lengths(val_py)
    plt.figure(figsize=(6, 4))
    plt.hist(len_train, bins=40, alpha=0.7, label="Train")
    plt.hist(len_val, bins=40, alpha=0.7, label="Validation")
    plt.xlabel("Tokenized length")
    plt.ylabel("Count")
    plt.title("Sequence length distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fname_prefix}_lengths.png")
    plt.close()

    print("Dataset EDA:")
    print(f"Train size: {len(labels_train)}, Val size: {len(labels_val)}")
    print(f"Train label mean: {np.mean(labels_train):.3f}")
    print(f"Val label mean: {np.mean(labels_val):.3f}")
    print(f"Avg tokenized length (train): {np.mean(len_train):.2f}, (val): {np.mean(len_val):.2f}")

if __name__ == "__main__":
    
    RUN_EPOCH_SWEEP = False  
    RUN_DATASIZE_SWEEP = False  
    epoch_grid = [5, 10, 50]
    data_size_grid = [10000, 50000, 500000]

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, base_model = build_model()
    (train_ds, test_ds), (train_loader, test_loader) = load_dataset_sst2(
        tokenizer, max_length=64, n_train=2000, n_test=800
    )
    analyze_dataset(train_ds, test_ds, tokenizer, fname_prefix="fig_data")
    ft_model = copy.deepcopy(base_model)

    print("Evaluating non-fine-tuned baseline...")
    baseline_eval = evaluate_model(base_model.to(device), test_loader, device)
    baseline_spectral = compute_spectral_norms(base_model)
    X_train_base, y_train_base = extract_embeddings(base_model, train_loader, device)
    X_test_base, y_test_base = extract_embeddings(base_model, test_loader, device)
    m_list = [X_train_base.size(1), 300, 100, 50]
    proj_results_base = random_projection_experiment(X_train_base, y_train_base, X_test_base, y_test_base, m_list)
    w_base, margins_train_base, margins_test_base = compute_min_norm_interpolant(
        X_train_base, y_train_base, X_test_base, y_test_base
    )
    sigma_list = [0.0, 0.01, 0.05, 0.1]
    robustness_base = perturbation_experiment(base_model, test_loader, device, sigma_list)
    print("Fine-tuning model with AdamW (15 epochs, unfrozen, wd=0.01)...")
    ft_model = fine_tune_model(
        ft_model,
        train_loader,
        test_loader,
        device,
        epochs=15,
        lr=3e-5,
        optimizer_name="adamw",
        freeze_encoder=False,
        weight_decay=0.01,
    )
    spectral_ft = compute_spectral_norms(ft_model)
    spectra_ft = compute_singular_values(ft_model, top_k=100)
    ft_eval = evaluate_model(ft_model, test_loader, device)
    X_train_ft, y_train_ft = extract_embeddings(ft_model, train_loader, device)
    X_test_ft, y_test_ft = extract_embeddings(ft_model, test_loader, device)
    proj_results_ft = random_projection_experiment(X_train_ft, y_train_ft, X_test_ft, y_test_ft, m_list)
    w_ft, margins_train_ft, margins_test_ft = compute_min_norm_interpolant(
        X_train_ft, y_train_ft, X_test_ft, y_test_ft
    )
    robustness_ft = perturbation_experiment(ft_model, test_loader, device, sigma_list)
    print("Fine-tuning model with SGD+momentum (20 epochs, unfrozen, wd=0.0)...")
    sgd_model = copy.deepcopy(base_model)
    sgd_model = fine_tune_model(sgd_model,train_loader,test_loader,device,
        epochs=20,lr=2e-3,optimizer_name="sgd",momentum=0.9,freeze_encoder=False,weight_decay=0.0,)

    torch.save(base_model.state_dict(), "model_pretrained.pt")
    torch.save(ft_model.state_dict(), "model_adam.pt")
    torch.save(sgd_model.state_dict(), "model_sgd.pt")

    with torch.no_grad():
        W_pre = get_ffn1_weight(base_model)
        W_adam = get_ffn1_weight(ft_model)
        W_sgd = get_ffn1_weight(sgd_model)
        diff_adam_pre = (W_adam - W_pre).norm().item()
        diff_sgd_pre = (W_sgd - W_pre).norm().item()
        diff_sgd_adam = (W_sgd - W_adam).norm().item()
        print("FFN lin1 Frobenius norms:")
        print(f"||W_adam - W_pre||_F = {diff_adam_pre:.6f}")
        print(f"||W_sgd  - W_pre||_F = {diff_sgd_pre:.6f}")
        print(f"||W_sgd  - W_adam||_F = {diff_sgd_adam:.6f}")
    spectral_sgd = compute_spectral_norms(sgd_model)
    spectra_sgd = compute_singular_values(sgd_model, top_k=100)
    sgd_eval = evaluate_model(sgd_model, test_loader, device)
    robustness_sgd = perturbation_experiment(sgd_model, test_loader, device, sigma_list)

    # Plots (fine-tuned by default) plus baseline margin comparison
    plot_projection_results(proj_results_ft, fname="fig_projection_accuracy_ft.png")
    plot_projection_results(proj_results_ft, fname="fig_projection_accuracy.png")
    plot_margin_hist(margins_train_ft, margins_test_ft, fname="fig_margin_hist_ft.png")
    plot_margin_hist(margins_train_ft, margins_test_ft, fname="fig_margin_hist.png")
    plot_spectral_norms(baseline_spectral, spectral_ft, fname="fig_spectral_norms_compare.png")
    plot_spectral_norms(baseline_spectral, spectral_ft, fname="fig_spectral_norms.png")
    plot_noise_robustness(robustness_ft, fname="fig_noise_robustness_ft.png")
    plot_noise_robustness(robustness_ft, fname="fig_noise_robustness.png")

    # Baseline plots
    plot_projection_results(proj_results_base, fname="fig_projection_accuracy_base.png")
    plot_margin_hist(margins_train_base, margins_test_base, fname="fig_margin_hist_base.png")
    plot_noise_robustness(robustness_base, fname="fig_noise_robustness_base.png")

    # Optimizer comparison plots
    plot_spectral_norms_triplet(baseline_spectral, spectral_ft, spectral_sgd, fname="fig_spectral_norms_opt.png")
    plot_singular_value_spectra(
        compute_singular_values(base_model, top_k=100),
        spectra_ft,
        spectra_sgd,
        fname="fig_spectral_spectra.png",
    )
    plot_top5_singular_values(
        compute_singular_values(base_model, top_k=5),
        spectra_ft,
        spectra_sgd,
        fname="fig_spectral_top5.png",
    )

    # Summary print
    print("\n=== Baseline (non-fine-tuned) ===")
    print(f"Eval: acc={baseline_eval['accuracy']:.4f}, loss={baseline_eval['loss']:.4f}")
    for m, stats in proj_results_base.items():
        print(f"Baseline m={m}: acc={stats['accuracy']:.4f}, loss={stats['loss']:.4f}")
    print("Spectral norms (baseline):")
    for k, v in baseline_spectral.items():
        print(f"{k}: {v:.4f}")
    print("Noise robustness (baseline):")
    for sigma, acc in robustness_base.items():
        print(f"sigma={sigma}: acc={acc:.4f}")

    print("\n=== Fine-tuned ===")
    print(f"Eval: acc={ft_eval['accuracy']:.4f}, loss={ft_eval['loss']:.4f}")
    for m, stats in proj_results_ft.items():
        print(f"Fine-tuned m={m}: acc={stats['accuracy']:.4f}, loss={stats['loss']:.4f}")
    print("Spectral norms (fine-tuned):")
    for k, v in spectral_ft.items():
        print(f"{k}: {v:.4f}")
    print("Noise robustness (fine-tuned):")
    for sigma, acc in robustness_ft.items():
        print(f"sigma={sigma}: acc={acc:.4f}")

    print("\n=== SGD fine-tuned ===")
    print(f"Eval: acc={sgd_eval['accuracy']:.4f}, loss={sgd_eval['loss']:.4f}")
    print("Spectral norms (sgd):")
    for k, v in spectral_sgd.items():
        print(f"{k}: {v:.4f}")
    print("Noise robustness (sgd):")
    for sigma, acc in robustness_sgd.items():
        print(f"sigma={sigma}: acc={acc:.4f}")

    #epoch sweep on fine-tuned embeddings 
    if RUN_EPOCH_SWEEP:
        epoch_sweep_results = []
        for ep in epoch_grid:
            print(f"\n[Epoch sweep] logistic regression on embeddings for {ep} epochs...")
            loss_ep, acc_ep = train_logreg(X_train_ft, y_train_ft, X_test_ft, y_test_ft, epochs=ep, lr=5e-3)
            epoch_sweep_results.append((ep, acc_ep, loss_ep))
        eps, accs, losses = zip(*epoch_sweep_results)
        plt.figure(figsize=(6, 4))
        plt.plot(eps, accs, marker="o", label="Accuracy")
        plt.plot(eps, losses, marker="s", label="Loss")
        plt.xlabel("Fine-tuning epochs")
        plt.ylabel("Metric")
        plt.title("Epoch sweep (default dataset)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("fig_epoch_sweep.png")
        plt.close()
        print("\nEpoch sweep results (epochs, acc, loss):", epoch_sweep_results)

    #data-size sweep 
    if RUN_DATASIZE_SWEEP:
        datasize_results = []
        for n in data_size_grid:
            print(f"\n[Data-size sweep] synthetic embeddings with n={n} ...")
            d_syn = 50
            # Generate a synthetic linear problem
            w_true = torch.randn(d_syn)
            Xtr = torch.randn(n, d_syn)
            logits = Xtr @ w_true + 0.5 * torch.randn(n)
            ytr = (logits > 0).long()
            n_test_syn = min(5000, max(1000, n // 10))
            Xte = torch.randn(n_test_syn, d_syn)
            logits_te = Xte @ w_true + 0.5 * torch.randn(n_test_syn)
            yte = (logits_te > 0).long()
            loss_n, acc_n = train_logreg(Xtr, ytr, Xte, yte, epochs=20, lr=5e-3, batch_size=512)
            datasize_results.append((n, acc_n, loss_n))
        ns, accs_n, losses_n = zip(*datasize_results)
        plt.figure(figsize=(6, 4))
        plt.plot(ns, accs_n, marker="o", label="Accuracy")
        plt.plot(ns, losses_n, marker="s", label="Loss")
        plt.xlabel("Training set size n")
        plt.xscale("log")
        plt.ylabel("Metric")
        plt.title("Data-size sweep (2 epochs)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("fig_datasize_sweep.png")
        plt.close()
        print("\nData-size sweep results (n, acc, loss):", datasize_results)
