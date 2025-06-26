import quantus
import numpy as np
import torch
import torch.nn as nn


class PredictorWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    @property
    def reconstruct_attention(self):
        return self.model.reconstruct_attention
        
    @reconstruct_attention.setter
    def reconstruct_attention(self, value):
        self.model.reconstruct_attention = value

    def forward(self, x):
        """
        Quantus-compatible forward pass (returns only prediction logits).
        Handles cases when model returns either:
        - a tuple (e.g., logits, attention)
        - just logits
        """
        output = self.model(x)

        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        # Ensure 2D shape: (batch_size, 1) or (batch_size,)
        logits = logits.squeeze(-1) if logits.ndim == 2 and logits.shape[1] == 1 else logits
        return logits  # shape: (batch_size,)

    def forward_original(self, x):
        """Original forward pass (returns both outputs)"""
        return self.model(x)

def explain_func(inputs: np.ndarray, 
                 model: PredictorWrapper,
                 targets=None,
                 **kwargs) -> np.ndarray:
        
    wrapper = model
    wrapper.reconstruct_attention = True
    with torch.no_grad():
        _, attention_map = wrapper.forward_original(inputs)
    wrapper.reconstruct_attention = False
    if wrapper.model.__class__.__name__ == "MultiHeadGatedAttentionMIL":
        attention_map = attention_map[:, 1, :, :, :]  # Select positive head
    return attention_map.squeeze(0).cpu().numpy()

def evaluate_selectivity(model, test_loader, softmax=True):
    wrapped_model = PredictorWrapper(model).eval()
    selectivity = quantus.Selectivity(
        patch_size=56, perturb_baseline="gaussian_noise",)

    positive_scores = []
    negative_scores = []
    for batch in test_loader:
        image = batch['image']
        label = batch['target']['label']
        print("selectivity...")
        scores = selectivity(
            model=wrapped_model,
            x_batch=image.cpu().numpy(),
            y_batch=label.cpu().numpy().astype(int),
            explain_func=explain_func,
            softmax=softmax
        )
        print(f"Scores: {scores}")
        if label.item() == 1:
            positive_scores.extend(scores)
        else:
            negative_scores.extend(scores)
    return {
        "positive": np.mean(positive_scores) if positive_scores else None,
        "negative": np.mean(negative_scores) if negative_scores else None,
    }

def evaluate_sparseness(model, test_loader):
    wrapped_model = PredictorWrapper(model).eval()
    sparseness = quantus.Sparseness()

    positive_scores = []
    negative_scores = []
    for batch in test_loader:
        image = batch['image']
        label = batch['target']['label']
        scores = sparseness(
            model=wrapped_model,
            x_batch=image.cpu().numpy(),
            y_batch=label.cpu().numpy(),
            a_batch=explain_func(image, wrapped_model),
        )
        print(f"Scores: {scores}")
        if label.item() == 1:
            positive_scores.extend(scores)
        else:
            negative_scores.extend(scores)
    return {
        "positive": np.mean(positive_scores) if positive_scores else None,
        "negative": np.mean(negative_scores) if negative_scores else None,
    }

def evaluate_relevance_rank_accuracy(model, test_loader):
    wrapped_model = PredictorWrapper(model).eval()
    relevance_rank_accuracy = quantus.RelevanceRankAccuracy()

    positive_scores = []
    for batch in test_loader:
        label = batch['target']['label']
        if label.item() == 0:
            continue
        image = batch['image']
        mask = batch['target'].get('mask', None)
        if mask is not None:
            mask = mask.unsqueeze(0)
        if label.item() == 1:
            scores = relevance_rank_accuracy(
                model=wrapped_model,
                x_batch=image.cpu().numpy(),
                y_batch=label.cpu().numpy(),
                a_batch=explain_func(image, wrapped_model),
                s_batch=mask.cpu().numpy() if mask is not None else None,
            )
            print(f"Scores: {scores}")
            positive_scores.extend(scores)

    return {
        "positive": np.mean(positive_scores) if positive_scores else None,
    }


def evaluate_faithfulness_correlation(model, test_loader):
    wrapped_model = PredictorWrapper(model).eval()
    faithfulness_correlation = quantus.FaithfulnessCorrelation()

    positive_scores = []
    negative_scores = []
    for batch in test_loader:
        image = batch['image']
        label = batch['target']['label']
        mask = batch['target'].get('mask', None)
        attention=explain_func(image, wrapped_model)
        scores = faithfulness_correlation(
            model=wrapped_model,
            x_batch=image.cpu().numpy(),
            y_batch=label.cpu().numpy().astype(int),
            a_batch=explain_func(image, wrapped_model),
            s_batch=mask.cpu().numpy() if mask is not None else None,
        )
        print(f"Scores: {scores}")
        if label.item() == 1:
            positive_scores.extend(scores)
        else:
            negative_scores.extend(scores)
    return {
        "positive": np.mean(positive_scores) if positive_scores else None,
        "negative": np.mean(negative_scores) if negative_scores else None,
    }

def evaluate_local_lipschitz_estimate(model, test_loader):
    wrapped_model = PredictorWrapper(model).eval()
    local_lipschitz_estimate = quantus.LocalLipschitzEstimate()

    positive_scores = []
    negative_scores = []
    for batch in test_loader:
        image = batch['image']
        label = batch['target']['label']
        scores = local_lipschitz_estimate(
            model=wrapped_model,
            x_batch=image.cpu().numpy(),
            y_batch=label.cpu().numpy().astype(int),
            # a_batch=explain_func(image, wrapped_model),
            explain_func=explain_func,
        )
        print(f"Scores: {scores}")
        if label.item() == 1:
            positive_scores.extend(scores)
        else:
            negative_scores.extend(scores)
    return {
        "positive": np.mean(positive_scores) if positive_scores else None,
        "negative": np.mean(negative_scores) if negative_scores else None,
    }