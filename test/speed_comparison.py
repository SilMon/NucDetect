import timeit

# Time sobel execution
t_sk = timeit.timeit(
    setup=
    """
from skimage.filters import sobel
from skimage import io
img = io.imread("demo.tif")[..., 2]
    """,
    stmt="sobel(img)",
    number=300
)
# Time JIT execution
t_jit = timeit.timeit(
    setup=
    """
from numba import jit
from skimage import io
import numpy as np

# Load image
img = io.imread("demo.tif")[..., 2]

# Define sobel sx and sy
sx = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]
sy = [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
]

@jit(nopython=True)
def sobel(img):
    # Create new array
    output = np.empty((img.shape[0], img.shape[1]), dtype='float32')
    # Pad image
    img = np.pad(img, 1, mode='constant')
    
    # Iterate over all elements of img
    for y in range(1, len(img.shape[0]) - 1):
        for x in range(1, len(img.shape[1])):
            
        
    """,
    stmt="""
    
    
    """,
    number=300
)
# Time CUDA execution
t_cuda = timeit.timeit(
    setup=
    """
from skimage.filters import sobel
from skimage import io
img = io.imread("demo.tif")[..., 2]
    """,
    stmt="sobel(img)",
    number=300
)
print(f"SKI: Finished in{t_sk/300: .4f} secs")
print(f"JIT: Finished in{t_jit/300: .4f} secs")
print(f"CUDA: Finished in{t_cuda/300: .4f} secs")