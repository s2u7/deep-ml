import numpy as np

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    # Your code here
    batch_size, seq_len, d_model = X.shape
    d_k = d_model // num_heads

    W_q, W_k, W_v, W_o = weights['W_q'], weights['W_k'], weights['W_v'], weights['W_o']

    W1, b1 = weights['W1'], weights['b1']
    W2, b2 = weights['W2'], weights['b2']

    gamma1, beta1 = weights['gamma1'], weights['beta1']
    gamma2, beta2 = weights['gamma2'], weights['beta2']

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    Q = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    scores_stable = scores - np.max(scores, axis=-1, keepdims=True)
    attn_weights = np.exp(scores_stable) 
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
    attn_output  = attn_weights @ V

    # concatenate heads
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    attn_output = attn_output @ W_o

    # Add & norm
    X1 = X + attn_output
    mean1 = np.mean(X1, axis=-1, keepdims=True)
    var1 = np.var(X1, axis=-1, keepdims=True)
    X1_norm = gamma1 * (X1 - mean1) / np.sqrt(var1 + eps) + beta1


    # position wise FF N    
    ff_hidden = X1_norm @ W1 + b1
    ff_hidden = np.maximum(0, ff_hidden)
    ff_output = ff_hidden @ W2 + b2

    X2 = X1_norm + ff_output
    mean2 = np.mean(X2, axis=-1, keepdims=True)
    var2 = np.var(X2, axis=-1, keepdims=True)
    output = gamma2 * (X2 - mean2) / np.sqrt(var2 + eps) + beta2

    return output



    



