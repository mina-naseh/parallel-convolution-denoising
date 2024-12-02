import jax
import jax.numpy as jnp

def convolution_2d_org(x, kernel):
    # Get the dimensions of the input image
    input_height, input_width = x.shape
    
    # Get the dimensions of the kernel
    kernel_height, kernel_width = kernel.shape
    
    # Calculate padding size for height and width (assumes symmetric padding)
    pad_height, pad_width = kernel_height // 2, kernel_width // 2

    # Add zero padding around the input array to maintain output size
    padded_x = jnp.pad(x, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Initialize an output array with zeros, same shape as the input image
    output_data = jnp.zeros_like(x)

    # Iterate over each pixel in the input image
    for i in range(input_height):
        for j in range(input_width):
            # Extract the region of the padded image that matches the kernel size
            region = padded_x[i:i + kernel_height, j:j + kernel_width]
            
            # Perform element-wise multiplication between the region and kernel, then sum the result
            output_data = output_data.at[i, j].set(jnp.sum(region * kernel))

    # Return the final convolved output
    return output_data


# Original: Get the dimensions of the input image
# New: This step is unnecessary because `jax.lax.conv_general_dilated` handles dimensions automatically.
# JAX automatically infers the spatial dimensions from the input tensor `x`.

# Original: Get the dimensions of the kernel
# New: Similarly, JAX takes care of kernel dimensions, no manual handling is needed.

# Original: Calculate padding size for height and width
# New: Use `padding="SAME"` directly in `jax.lax.conv_general_dilated`, which applies symmetric padding internally.
# This avoids manual computation of padding.

# Original: Add zero padding around the input array
# New: Padding is automatically handled by `jax.lax.conv_general_dilated` using the `padding` argument.
# No need for `jnp.pad`.

# Original: Initialize an output array with zeros
# New: `jax.lax.conv_general_dilated` returns a properly sized output array, so no pre-allocation is required.

# Original: Nested loops to process each pixel
# New: Replace the nested loops with a vectorized operation using JAX's built-in convolution function.
# The entire convolution is done in one call to `jax.lax.conv_general_dilated`.

# Original: Element-wise multiplication and summation
# New: This operation is inherently part of `jax.lax.conv_general_dilated`, optimized for batch operations.

# Optimized Function
def convolution_2d(x, kernel):
    # Flip the kernel for proper convolution operation
    kernel = jnp.flip(kernel)
    
    # Add batch and channel dimensions to the input image
    x = x[jnp.newaxis, ..., jnp.newaxis]
    
    # Add input/output channel dimensions to the kernel
    kernel = kernel[jnp.newaxis, ..., jnp.newaxis]
    
    # Perform the convolution operation with symmetric padding
    result = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1, 1),  # Stride of 1 means sliding one pixel at a time
        padding="SAME"  # Automatically pads to keep the output size the same as the input
    )
    
    # Remove the batch and channel dimensions from the output
    return result[0, ..., 0]
