import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import jax
import jax.numpy as jnp
from jax import grad
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load the MNIST dataset (only rank 0 loads the data)
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
    x = None
    y_true = None

# Broadcast the input and target data to all ranks
x = comm.bcast(x, root=0)
y_true = comm.bcast(y_true, root=0)

# Define convolution function using JAX
def convolution_2d(x, kernel):
    input_height, input_width = x.shape
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2

    # Pad the input array
    padded_x = jnp.pad(x, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Initialize the output matrix
    output_data = jnp.zeros_like(x)

    # Perform the convolution operation
    for i in range(input_height):
        for j in range(input_width):
            region = padded_x[i:i + kernel_height, j:j + kernel_width]
            output_data = output_data.at[i, j].set(jnp.sum(region * kernel))

    return output_data

# Define loss function
def loss_fn(kernel, x, y_true):
    y_pred = convolution_2d(x, kernel)
    return jnp.mean((y_pred - y_true) ** 2)  # Mean squared error

# Initialize kernel
kernel = jnp.array([[0.01, 0.0, 0.0],
                    [-1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]])  # Random kernel for horizontal edge detection

# Gradient of the loss function w.r.t. the kernel
loss_grad = grad(loss_fn)

# Training loop (distributed)
learning_rate = 0.01
num_iterations = 50
local_kernel = kernel.copy()

losses = []
for i in range(rank, num_iterations, size):  # Distribute iterations across ranks
    gradients = loss_grad(local_kernel, x, y_true)
    local_kernel -= learning_rate * gradients  # Update kernel with gradient descent

    # Compute the loss
    current_loss = loss_fn(local_kernel, x, y_true)
    losses.append(current_loss)

    # Print progress on each rank
    print(f"Rank {rank}, Iteration {i}, Loss: {current_loss:.4f}")

# Gather all the trained kernels on rank 0
all_kernels = comm.gather(local_kernel, root=0)

# Aggregate the results on rank 0 (e.g., average the kernels)
if rank == 0:
    final_kernel = jnp.mean(jnp.array(all_kernels), axis=0)

    # Visualize results
    plt.figure(figsize=(8, 6))

    # Display original noisy image
    plt.subplot(2, 2, 1)
    plt.imshow(x, cmap='gray')
    plt.title("Noisy Image")
    plt.axis('off')

    # Display target clean image
    plt.subplot(2, 2, 2)
    plt.imshow(y_true, cmap='gray')
    plt.title("Target (Clean Image)")
    plt.axis('off')

    # Display denoised image
    y_denoised = convolution_2d(x, final_kernel)
    plt.subplot(2, 2, 3)
    plt.imshow(y_denoised, cmap='gray')
    plt.title("Denoised Image")
    plt.axis('off')

    # Display final kernel
    plt.subplot(2, 2, 4)
    plt.imshow(final_kernel, cmap='gray')
    plt.title("Final Kernel")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("./plots/denoising_parallel.png")
    print("Denoised image and kernel saved in './plots/denoising_parallel.png'")
