import quantus
import numpy as np
import torch
import torch.nn as nn
import logging
logging.basicConfig(level=logging.ERROR)

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
        output = self.model(x)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        # Handle binary models (single logit output)
        # if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(-1)  # shape: (B,)
        
        # Case 1: binary classification (sigmoid output)
        if logits.ndim == 1:
            probs = torch.sigmoid(logits)  # shape: (B,)
            probs = torch.stack([1 - probs, probs], dim=1)  # shape: (B, 2)

        # Case 2: multiclass (or binary) with softmax
        elif logits.ndim == 2:
            probs = torch.softmax(logits, dim=1)  # shape: (B, C)

        else:
            raise ValueError(f"Unexpected output shape from model {logits.shape}. ")

        return probs  # always shape: (B, 2)

    def forward_original(self, x):
        return self.model(x)


class PredictorWrapperToSoftmax(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    @property
    def reconstruct_attention(self):
        return self.model.reconstruct_attention
        
    @reconstruct_attention.setter
    def reconstruct_attention(self, value):
        self.model.reconstruct_attention = value
    
    @staticmethod
    def binary_to_softmax(probs):
        probs = probs.item() if probs.ndim == 0 else probs.squeeze(-1)
        probs = torch.stack([1 - probs, probs], dim=1)  # convert to 2-class softmax equivalent
        return probs
    
    def forward(self, x):
        output = self.model(x)

        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        logits = logits.squeeze(-1) if logits.ndim == 2 and logits.shape[1] == 1 else logits
        softmax_probs = self.binary_to_softmax(torch.sigmoid(logits))
        return softmax_probs#.reshape(-1, 1) # shape: (batch_size,)
        # return softmax_probs.squeeze(0)   # shape: (batch_size,)
        # return softmax_probs              # shape: (batch_size, 2)

    def forward_original(self, x):
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

def evaluate_selectivity(model, test_loader, use_wrapper):
    return evaluate_metric_by_class(
        model, test_loader,
        metric=quantus.Selectivity(patch_size=56, perturb_baseline="gaussian_noise"),
        use_wrapper=use_wrapper,
        explain_func=explain_func,
    )

def evaluate_sparseness(model, test_loader):
    return evaluate_metric_by_class(
        model, test_loader,
        metric=quantus.Sparseness(),
        needs_a_batch=True,
        explain_func=explain_func,
    )

def evaluate_relevance_rank_accuracy(model, test_loader):
    return evaluate_metric_by_class(
        model, test_loader,
        metric=quantus.RelevanceRankAccuracy(),
        needs_a_batch=True,
        needs_s_batch=True,
        explain_func=explain_func,
        skip_label=0, # there are no masks for negative class in CMMD
    )

def evaluate_topk_intersection(model, test_loader):
    return evaluate_metric_by_class(
        model, test_loader,
        metric=quantus.TopKIntersection(k=40_000),
        needs_a_batch=True,
        needs_s_batch=True,
        explain_func=explain_func,
        skip_label=0, # there are no masks for negative class in CMMD
    )

def evaluate_faithfulness_correlation(model, test_loader, use_wrapper):
    return evaluate_metric_by_class(
        model, test_loader,
        metric=quantus.FaithfulnessCorrelation(perturb_baseline="uniform"),
        needs_a_batch=True,
        needs_s_batch=True,
        explain_func=explain_func,
        use_wrapper=use_wrapper,
    )

def evaluate_mprt(model, test_loader):
    return evaluate_metric_by_class(
        model, test_loader,
        metric=quantus.MPRT(similarity_func=quantus.similarity_func.cosine),
        explain_func=explain_func,
    )

def evaluate_avg_sensitivity(model, test_loader):
    return evaluate_metric_by_class(
        model, test_loader,
        metric=quantus.AvgSensitivity(nr_samples=10),
        explain_func=explain_func,
    )

def evaluate_metric_by_class(
    model,
    test_loader,
    metric,
    use_wrapper=False,
    explain_func=None,
    needs_a_batch=False,
    needs_s_batch=False,
    mask_key='mask',
    skip_label=None
):
    model.eval()
    wrapped_model = PredictorWrapper(model) if not use_wrapper else PredictorWrapperToSoftmax(model)
    print(wrapped_model.__class__.__name__)
    wrapped_model.eval()

    positive_scores, negative_scores = [], []

    skipped_count = 0
    for batch in test_loader:
        image = batch['image']
        label = batch['target']['label']
        mask = batch['target'].get(mask_key, None)
        if skip_label is not None and label.item() == skip_label:
            skipped_count += 1
            continue

        kwargs = {
            'model': wrapped_model,
            'x_batch': image.cpu().numpy(),
            'y_batch': label.cpu().numpy().astype(int),
        }
        if needs_a_batch:
            kwargs['a_batch'] = explain_func(image, wrapped_model)
        if needs_s_batch and mask is not None:
            kwargs['s_batch'] = mask.unsqueeze(0).cpu().numpy()
        if explain_func is not None and 'explain_func' in metric.__call__.__code__.co_varnames:
            kwargs['explain_func'] = explain_func
            kwargs['softmax'] = False

        output = wrapped_model(image)
        scores = metric(**kwargs)
        print(f"Scores: {scores}")

        if label.item() == 1:
            # positive_scores.extend(scores)
            positive_scores.append(scores)
        else:
            # negative_scores.extend(scores)
            negative_scores.append(scores)
    print(f"Skipped {skipped_count} samples with label {skip_label}.")
    return {
        "positive": positive_scores,
        "negative": negative_scores,
    }
