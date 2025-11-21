import numpy as np
from attention import scaled_dot_product_attention

def main():
    Q = np.array([[[1., 0., 1.]]])   
    K = np.array([[[1., 0., 1.],
                   [0., 1., 0.]]])   
    V = np.array([[[2., 0.],
                   [0., 2.]]])       

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("Attention Weights:\n", weights)
    print("Output:\n", output)

if __name__ == "__main__":
    main()
