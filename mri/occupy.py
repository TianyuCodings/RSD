import torch
import time

# Ensure GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Matrix size (adjust if needed)
MATRIX_SIZE = 8192  # Large enough to keep GPU busy

# Create random matrices on GPU
def generate_matrices():
    A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    return A, B

# Infinite loop for continuous GPU load
try:
    while True:
        A, B = generate_matrices()
        start_time = time.time()

        # Matrix multiplication
        C = torch.matmul(A, B)

        # Ensure computation finishes
        torch.cuda.synchronize()

        end_time = time.time()

except KeyboardInterrupt:
    print("Script interrupted. Exiting...")