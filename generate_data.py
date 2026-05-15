import os
import numpy as np
from data.surfaces import generate_dataset

def main():
    print("Starting dataset generation...")
    # Generating 50,000 surfaces might take hours. 
    # For a quicker test, you can change n_surfaces to 500 or 1000.
    X, y = generate_dataset(
        n_surfaces=50_000, 
        n_paths=100_000, 
        base_seed=42,
        verbose=True
    )
    
    os.makedirs("data/saved", exist_ok=True)
    np.save("data/saved/X.npy", X)
    np.save("data/saved/y.npy", y)
    print("Dataset saved to data/saved/")

if __name__ == "__main__":
    main()
