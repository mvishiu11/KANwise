import numpy as np
import pandas as pd
import os

def generate_synthetic_data():
    np.random.seed(42)  # For reproducibility
    x = np.random.rand(1000, 1) * 10  # 1000 samples, 1 feature
    noise = np.random.normal(0, 2, x.shape)  # Noise with mean = 0 and std dev = 2
    y = 2 * x.squeeze() + 3 + noise.squeeze()  # Ensure y is also squeezed
    data = pd.DataFrame({'Feature': x.squeeze(), 'Target': y.squeeze()})  # Use squeeze to ensure 1D
    data.to_csv(os.path.join('data', 'synthetic_data.csv'), index=False)

def solve_heat_equation(num_points=100):
    x = np.linspace(0, 1, num=num_points)  # Spatial domain discretized
    t = np.linspace(0, 1, num=num_points)  # Time discretization
    # Example function: u = sin(pi * x) * exp(-pi^2 * t)
    # Using a meshgrid to calculate u for each (x,t) pair
    X, T = np.meshgrid(x, t, indexing='ij')
    u = np.sin(np.pi * X) * np.exp(-np.pi**2 * T)

    # Flatten the arrays to create a 1D list of x, t, and u values
    data = pd.DataFrame({
        'Position': X.flatten(),
        'Time': T.flatten(),
        'Temperature': u.flatten()
    })
    data.to_csv(os.path.join('data', 'heat_equation_data.csv'), index=False)

def generate_complex_function_data():
    x = np.linspace(-10, 10, 1000)
    y = np.sin(x) * np.log(x**2 + 1)
    data = pd.DataFrame({'X': x, 'Y': y})
    data.to_csv(os.path.join('data', 'function_discovery_data.csv'), index=False)

generate_synthetic_data()
solve_heat_equation()
generate_complex_function_data()


