import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V
    

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    d_k = Q.shape[1]


    scores = np.matmul(Q, K.T) / np.sqrt(d_k)

    

    score_max = np.max(scores, axis=1, keepdims=True)
    

    attention_weights = np.exp(scores - score_max) / np.sum(np.exp(scores - score_max), axis = 1, keepdims = True)

    attention_output = np.matmul(attention_weights, V)
    return attention_output


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    assert d_model % n_heads == 0

    d_k = d_model // n_heads

    Q_heads = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0 , 2)
    K_heads = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V_heads = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    head_outputs = []

    for i in range(n_heads):
        head_output = self_attention(Q_heads[i], K_heads[i], V_heads[i])
        head_outputs.append(head_output)

    output = np.concatenate(head_outputs, axis = -1)
    return output

        