import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Computes scaled dot-product attention.
    Q, K, V are numpy arrays with shapes:
        Q: (..., seq_len_q, depth)
        K: (..., seq_len_k, depth)
        V: (..., seq_len_k, depth_v)
    mask: optional array broadcastable to (..., seq_len_q, seq_len_k)

    Returns:
        output: (..., seq_len_q, depth_v)
        attention_weights: (..., seq_len_q, seq_len_k)
    """
    scores = np.matmul(Q, K.transpose(0, 2, 1))
    depth = Q.shape[-1]
    scores = scores / np.sqrt(depth)
    if mask is not None:
        scores = scores + (mask * -1e9)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    output = np.matmul(attention_weights, V)

    return output, attention_weights
