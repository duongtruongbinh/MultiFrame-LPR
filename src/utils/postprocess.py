"""Post-processing utilities for OCR decoding."""
from typing import Dict, List, Tuple
import torch
import numpy as np



def decode_with_confidence(
    preds: torch.Tensor,
    idx2char: Dict[int, str]
) -> List[Tuple[str, float]]:
    """CTC decode predictions with confidence scores.
    
    Args:
        preds: Log-softmax predictions of shape [Batch, Time, Classes].
        idx2char: Index to character mapping.
    
    Returns:
        List of (predicted_string, confidence_score) tuples.
    """
    probs = preds.exp()
    max_probs, indices = probs.max(dim=2)    
    indices_np = indices.detach().cpu().numpy()
    max_probs_np = max_probs.detach().cpu().numpy()
    
    result_list: List[Tuple[str, float]] = []
    
    batch_size, time_steps = indices_np.shape
    
    for b in range(batch_size):
        path = indices_np[b]
        probs_b = max_probs_np[b]
        
        pred_chars = []
        confidences = []
        last_char = 0 
        
        for t in range(time_steps):
            c = path[t]
            if c != last_char: 
                if c != 0:      
                    pred_chars.append(idx2char.get(c, ''))
                    confidences.append(probs_b[t])
                last_char = c
        
        pred_str = "".join(pred_chars)

        score = float(np.mean(confidences)) if confidences else 0.0
        result_list.append((pred_str, score))
    
    return result_list
