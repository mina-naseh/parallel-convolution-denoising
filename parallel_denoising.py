from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from jax import grad, numpy as jnp
from tensorflow.keras.datasets import mnist

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load the MNIST dataset (only process 0 loads the dataset)
if rank == 0:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = x_train[0]
    y_true = x.copy()

    # Add salt-and-pepper noise
    num_corrupted_pixels = 100
    for _ in range(num_corrupted_pixels):
        i, j = np.random.randint(0, x.shape[0]), np.random.randint(0, x.shape[1])
        x[i, j] = np.random.choice([0, 255])

    # Normalize images
    y_true = y_true.astype(np.float32) / 255.0
    x = x.astype(np.float32) / 255.0
else:
    x, y_true = None, None

# Broadcast the data to all processes
x = comm.bcast(x if rank == 0 else None, root=0)
y_true = comm.bcast(y_true if rank == 0 else None, root=0)

# Define convolution function
def convolution_2d(x, kernel):
    input_height, input_width = x.shape
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    padded_x = jnp.pad(x, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    output_data = jnp.zeros_like(x)

    for i in range(input_height):
        for j in range(input_width):
            region = padded_x[i:i + kernel_height, j:j + kernel_width]
            output_data = output_data.at[i, j].set(jnp.sum(region * kernel))
    return output_data

# Initialize kernel
kernel = jnp.array([[0.01, 0.0, 0.0],
                    [-1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]])  # Random kernel for horizontal edge detection

# Divide image rows among processes
rows_per_process = x.shape[0] // size
start_row = rank * rows_per_process
end_row = (rank + 1) * rows_per_process if rank != size - 1 else x.shape[0]

# Local computation of convolution
local_x = x[start_row:end_row, :]
local_result = convolution_2d(local_x, kernel)

# Gather results from all processes
gathered_result = None
if rank == 0:
    gathered_result = np.zeros_like(x)
comm.Gather(local_result, gathered_result, root=0)

# Process 0 combines results and computes the loss
if rank == 0:
    loss = jnp.mean((gathered_result - y_true) ** 2)
    print(f"Final Loss: {loss}")

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(x, cmap='gray')
    plt.title("Noisy Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(y_true, cmap='gray')
    plt.title("Target (Clean Image)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(gathered_result, cmap='gray')
    plt.title("Denoised Image")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("mpi_output.png")
