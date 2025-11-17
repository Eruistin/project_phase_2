"""
Multi-Shadow Model MIA with Advanced Feature Engineering
Optimized for high TPR@low FPR
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, roc_curve
import json
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import Dataset
import torch.nn.functional as F
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from collections import defaultdict
import argparse


def tokenize_dataset(ds, tok, max_len):
    ds = ds.filter(lambda ex: isinstance(ex.get("text", ""), str) and len(ex["text"].strip()) > 0)
    def _map(ex):
        out = tok(ex["text"], truncation=True, padding="max_length", max_length=max_len, return_attention_mask=True)
        out["labels"] = out["input_ids"].copy()
        return out
    ds = ds.map(_map, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class MultiShadowMIA:
    """
    Enhanced MIA using multiple shadow models with membership information
    """
    def __init__(self, target_model_path, shadow_model_info, device):
        """
        Args:
            target_model_path: Path to target model
            shadow_model_info: Dict {model_path: {"member_indices": [...], "nonmember_indices": [...]}}
            device: cuda or cpu
        """
        self.device = device
        self.shadow_model_info = shadow_model_info
        
        # Load target model
        print(f"Loading target model: {target_model_path}")
        self.target_model = self.load_model(target_model_path)
        self.target_model.to(device)
        self.target_model.eval()
        
        # Pre-load all shadow models
        self.shadow_models = {}
        print("\nLoading shadow models...")
        for model_path in shadow_model_info.keys():
            try:
                model = self.load_model(model_path)
                model.to(device)
                model.eval()
                self.shadow_models[model_path] = model
                print(f"  ✓ {model_path}")
            except Exception as e:
                print(f"  ✗ {model_path}: {e}")
        
        print(f"\nLoaded {len(self.shadow_models)} shadow models")
    
    def load_model(self, model_path):
        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, model_path)
        return model
    
    
    def get_sample_shadow_groups(self, sample_idx):
        """
        For a given sample, find which shadow models have it as member/non-member
        """
        member_models = []
        nonmember_models = []
        
        for model_path, info in self.shadow_model_info.items():
            if model_path not in self.shadow_models:
                continue
            
            if sample_idx in info.get('member_indices', []):
                member_models.append(model_path)
            elif sample_idx in info.get('nonmember_indices', []):
                nonmember_models.append(model_path)
        
        return member_models, nonmember_models
    
    @torch.no_grad()
    def compute_sample_scores(self, input_ids, attention_mask, labels, sample_idx):
        """
        Compute multiple scores for a single sample using shadow models
        """
        # Get target model outputs
        target_outputs = self.target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        target_logits = target_outputs.logits
        target_loss = target_outputs.loss.item()
        target_ppl = np.exp(target_loss)
        
        # Find relevant shadow models
        member_models, nonmember_models = self.get_sample_shadow_groups(sample_idx)
        
        # Compute statistics from shadow models
        member_losses = []
        nonmember_losses = []
        member_cos_sims = []
        nonmember_cos_sims = []
        
        # Sample from shadows (use up to 5 of each for efficiency)
        for model_path in member_models[:5]:
            model = self.shadow_models[model_path]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss.item()
            member_losses.append(loss)
            
            cos_sim = F.cosine_similarity(
                target_logits.view(-1, target_logits.size(-1)),
                outputs.logits.view(-1, outputs.logits.size(-1)),
                dim=-1
            ).mean().item()
            member_cos_sims.append(cos_sim)
        
        for model_path in nonmember_models[:5]:
            model = self.shadow_models[model_path]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss.item()
            nonmember_losses.append(loss)
            
            cos_sim = F.cosine_similarity(
                target_logits.view(-1, target_logits.size(-1)),
                outputs.logits.view(-1, outputs.logits.size(-1)),
                dim=-1
            ).mean().item()
            nonmember_cos_sims.append(cos_sim)
        
        # Compute features
        features = {}
        
        # Feature 1: Target loss/perplexity
        features['target_loss'] = target_loss
        features['target_ppl'] = target_ppl
        
        # Feature 2: Comparison with member models
        if member_losses:
            features['avg_member_loss'] = np.mean(member_losses)
            features['min_member_loss'] = np.min(member_losses)
            features['loss_to_member_ratio'] = target_loss / (np.mean(member_losses) + 1e-10)
            features['avg_member_cos'] = np.mean(member_cos_sims)
        else:
            features['avg_member_loss'] = 0
            features['min_member_loss'] = 0
            features['loss_to_member_ratio'] = 1.0
            features['avg_member_cos'] = 0
        
        # Feature 3: Comparison with non-member models
        if nonmember_losses:
            features['avg_nonmember_loss'] = np.mean(nonmember_losses)
            features['min_nonmember_loss'] = np.min(nonmember_losses)
            features['loss_to_nonmember_ratio'] = target_loss / (np.mean(nonmember_losses) + 1e-10)
            features['avg_nonmember_cos'] = np.mean(nonmember_cos_sims)
        else:
            features['avg_nonmember_loss'] = 0
            features['min_nonmember_loss'] = 0
            features['loss_to_nonmember_ratio'] = 1.0
            features['avg_nonmember_cos'] = 0
        
        # Feature 4: Relative position (sandwich score)
        if member_losses and nonmember_losses:
            member_avg = np.mean(member_losses)
            nonmember_avg = np.mean(nonmember_losses)
            
            # Higher score = closer to member models
            if nonmember_avg > member_avg:
                features['sandwich_score'] = (nonmember_avg - target_loss) / (nonmember_avg - member_avg + 1e-10)
            else:
                features['sandwich_score'] = 0.5
            
            # Perplexity-based
            member_ppl = np.exp(member_avg)
            nonmember_ppl = np.exp(nonmember_avg)
            features['ppl_sandwich'] = (nonmember_ppl - target_ppl) / (nonmember_ppl - member_ppl + 1e-10)
            
            # Cosine similarity based
            if member_cos_sims and nonmember_cos_sims:
                features['cos_sandwich'] = (features['avg_member_cos'] - features['avg_nonmember_cos'])
        else:
            features['sandwich_score'] = 0.5
            features['ppl_sandwich'] = 0.5
            features['cos_sandwich'] = 0
        
        # Feature 5: Statistical features
        features['num_member_models'] = len(member_models)
        features['num_nonmember_models'] = len(nonmember_models)
        
        return features
    
    def extract_all_features(self, dl, sample_indices):
        """
        Extract features for all samples in dataloader
        """
        all_features = []
        feature_names = None
        
        batch_idx = 0
        for batch in tqdm(dl, desc="Extracting features"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            batch_size = input_ids.size(0)
            
            for i in range(batch_size):
                sample_idx = sample_indices[batch_idx]
                
                features = self.compute_sample_scores(
                    input_ids[i:i+1],
                    attention_mask[i:i+1],
                    labels[i:i+1],
                    sample_idx
                )
                
                if feature_names is None:
                    feature_names = list(features.keys())
                
                all_features.append([features[k] for k in feature_names])
                batch_idx += 1
        
        return np.array(all_features), feature_names
    
    def compute_simple_scores(self, dl, method='sandwich'):
        """
        Compute simple scores without full feature extraction (faster)
        """
        scores = []
        
        for batch in tqdm(dl, desc="Computing scores"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            target_outputs = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            if method == 'loss':
                # Simple loss-based score
                batch_scores = -target_outputs.loss.item()
            elif method == 'ppl':
                # Perplexity-based
                batch_scores = -np.exp(target_outputs.loss.item())
            else:
                batch_scores = target_outputs.loss.item()
            
            scores.append(batch_scores)
        
        return np.array(scores)


def prepare_shadow_model_info(shadow_dir, data_info_file):
    """
    Prepare shadow model information from directory structure
    
    Args:
        shadow_dir: Directory containing shadow models
        data_info_file: JSON file mapping shadow models to their training data
        
    Returns:
        shadow_model_info: Dict with member/nonmember indices for each model
    """
    with open(data_info_file, 'r') as f:
        data_info = json.load(f)
    
    shadow_model_info = {}
    
    for model_name, info in data_info.items():
        model_path = os.path.join(shadow_dir, model_name)
        if os.path.exists(model_path):
            shadow_model_info[model_path] = info
    
    return shadow_model_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--shadow_dir", type=str, required=True)
    parser.add_argument("--shadow_info", type=str, required=True,
                       help="JSON file with shadow model training data info")
    parser.add_argument("--data_dir", type=str, default="wiki_json")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--sample_indices_file", type=str, required=True,
                       help="NPY file with sample indices")
    parser.add_argument("--label_file", type=str, default="label.npy")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--method", type=str, default='features',
                       choices=['features', 'simple'],
                       help="Use full features or simple scoring")
    parser.add_argument("--output", type=str, default="predictions.npy")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load shadow model info
    shadow_model_info = prepare_shadow_model_info(args.shadow_dir, args.shadow_info)
    print(f"Loaded info for {len(shadow_model_info)} shadow models")
    
    # Load data
    data_path = Path(args.data_dir) / args.data_file
    texts = _read_json(data_path)
    texts = [item['text'] for item in texts]
    dataset = Dataset.from_dict({"text": texts})
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    dataset = tokenize_dataset(dataset, tokenizer, args.block_size)
    dl = DataLoader(dataset, batch_size=args.batch_size)
    
    # Load sample indices
    sample_indices = np.load(args.sample_indices_file)
    
    # Initialize MIA
    mia = MultiShadowMIA(args.target_model, shadow_model_info, device)
    
    # Compute scores
    if args.method == 'features':
        features, feature_names = mia.extract_all_features(dl, sample_indices)
        print(f"\nExtracted {features.shape[1]} features: {feature_names}")
        
        # For now, use sandwich_score as the main score
        # In practice, you should train a classifier on these features
        score_idx = feature_names.index('sandwich_score') if 'sandwich_score' in feature_names else 0
        pred = features[:, score_idx]
    else:
        pred = mia.compute_simple_scores(dl, method='loss')
    
    # Evaluate
    y_true = np.load(args.label_file)
    
    auc_score = roc_auc_score(y_true, pred)
    fpr, tpr, _ = roc_curve(y_true, pred)
    
    for target_fpr in [0.001, 0.005, 0.01, 0.05]:
        if (fpr <= target_fpr).any():
            tpr_at_target = np.interp(target_fpr, fpr, tpr)
            print(f"TPR@FPR={target_fpr:.3f} = {tpr_at_target:.4f}")
    
    print(f"\nAUC = {auc_score:.4f}")
    
    # Save predictions
    np.save(args.output, pred)
    print(f"Predictions saved to {args.output}")