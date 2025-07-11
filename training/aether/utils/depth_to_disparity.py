import numpy as np
import imageio.v3 as iio
from postprocess_utils import depth_to_disparity, colorize_depth

def main():
    # Load depth data from a file
    depth = np.load("depth.npy")

    # Convert depth to disparity
    disparity, dmax = depth_to_disparity(depth + 1e-6, sqrt_disparity=True)

    # Save the disparity data to a file
    iio.imwrite(
        f"d2d.mp4",
        (colorize_depth(disparity) * 255).astype(np.uint8),
        fps=8,
    )
    # np.save("disparity.npy", disparity)

if __name__ == "__main__":
    main()