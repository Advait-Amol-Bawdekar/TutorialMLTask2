import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

def load_data(filepath):
    """
    Imports Deformation Gradient (F), First Piola-Kirchhoff Stress (P), 
    and Strain Energy Density (W) from a text file.
    
    Data format: Columns 0-8 (F), 9-17 (P), 18 (W).
    """
    # Load the raw numerical data from the specified text file using numpy
    data = np.loadtxt(filepath)
    
    # Reshape the flat 9-element vectors into 3x3 matrices. 
    # '-1' automatically computes the number of rows (N) based on the file size.
    F = data[:, 0:9].reshape(-1, 3, 3)
    P = data[:, 9:18].reshape(-1, 3, 3)
    
    # Extract the scalar energy density (W) from the last column
    W = data[:, 18]
    
    return F, P, W

def compute_invariants(F):
    """
    Calculates the transversely isotropic invariants (I1, J, I4, I5) for a given 
    deformation gradient F. These invariants are frame-indifferent (objective).
    """
    # Calculate the right Cauchy-Green deformation tensor C = F^T * F
    C = jnp.dot(F.T, F)
    
    # Calculate the volume ratio J as the determinant of F
    J = jnp.linalg.det(F)
    
    # First invariant I1 is the trace (sum of diagonal elements) of C
    I1 = jnp.trace(C)
    
    # Define the transversely isotropic structural tensor G_ti (preferred fiber direction)
    G_ti = jnp.array([[4.0, 0.0, 0.0], 
                      [0.0, 0.5, 0.0], 
                      [0.0, 0.0, 0.5]])
    
    # Invariant I4 couples the deformation tensor C with the structural tensor G_ti
    I4 = jnp.trace(jnp.dot(C, G_ti))
    
    # Invariant I5 couples the cofactor of C (cof(C) = J^2 * C^-1) with G_ti
    cof_C = (J ** 2) * jnp.linalg.inv(C)
    I5 = jnp.trace(jnp.dot(cof_C, G_ti))
    
    return jnp.array([I1, J, I4, I5])

def analytical_potential(F):
    """
    Evaluates the analytical transversely isotropic hyperelastic potential W(F).
    """
    I1, J, I4, I5 = compute_invariants(F)
    # Energy density formula provided in Eq. 3 of the tutorial
    return 8.0 * I1 + 10.0 * (J ** 2) - 56.0 * jnp.log(J) + 0.2 * (I4 ** 2 + I5 ** 2) - 44.0

# JAX automatically derives the stress tensor P = dW/dF from the scalar potential W(F) via autodiff.
analytical_stress = jax.grad(analytical_potential)

# Vectorize functions so they can process batches of (N, 3, 3) arrays simultaneously without python loops.
v_compute_invariants = jax.vmap(compute_invariants)
v_analytical_potential = jax.vmap(analytical_potential) 
v_analytical_stress = jax.vmap(analytical_stress)

def main():
    base_path = "."
    
    # ---------------------------------------------------------
    # 1. VERIFY MATRICES IN TERMINAL
    # ---------------------------------------------------------
    F_uni, P_uni, _ = load_data(os.path.join(base_path, 'calibration', 'uniaxial.txt'))
    print("--- 1. MATRICES RESHAPING VERIFICATION ---")
    print(f"Data successfully reshaped to (N, 3, 3). Total points in uniaxial: {F_uni.shape[0]}")
    print("First F matrix [3x3]:\n", F_uni[0])
    print("First P matrix [3x3]:\n", P_uni[0])
    print("\n--- 2. JAX INVARIANT, ENERGY, & STRESS VALIDATION (ALL CASES) ---")
    
    # ---------------------------------------------------------
    # 2. COMPREHENSIVE VALIDATION & VISUALIZATION (3x3 Grid)
    # ---------------------------------------------------------
    # Dictionary containing ALL calibration and test datasets
    datasets = {
        'Uniaxial':   ('calibration', 'uniaxial.txt'),
        'Biaxial':    ('calibration', 'biaxial.txt'),
        'Pure Shear': ('calibration', 'pure_shear.txt'),
        'Mixed Test': ('test', 'mixed_test.txt') 
    }
    
    # Setup a 3x3 grid window to plot all tensor stress components against deformation components
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('All 9 Components Across All Load Paths: $P_{ij}$ vs $F_{ij}$', fontsize=16)
    
    for label, (folder, filename) in datasets.items():
        # Load F, expected P, and expected W from data text files
        F, P_expected, W_expected = load_data(os.path.join(base_path, folder, filename))
        
        # Load expected invariants from the corresponding file in the invariants folder
        I_expected = np.loadtxt(os.path.join(base_path, 'invariants', f'I_{filename}'))
        
        # Compute predictions using JAX
        I_computed = v_compute_invariants(F)
        W_computed = v_analytical_potential(F)
        P_computed = v_analytical_stress(F)
        
        # Validate predictions against text file answer keys (using an absolute/relative tolerance of 1e-4)
        inv_match = np.allclose(I_computed, I_expected, rtol=1e-4, atol=1e-4)
        energy_match = np.allclose(W_computed, W_expected, rtol=1e-4, atol=1e-4)
        stress_match = np.allclose(P_computed, P_expected, rtol=1e-4, atol=1e-4)
        
        # Print verification status for this specific load case to the terminal
        print(f"[{label}] -> Invariants: {inv_match} | Energy (W): {energy_match} | Stress (P): {stress_match}")
        
        # Populate the 3x3 subplot grid with data curves for this load path
        for i in range(3):
            for j in range(3):
                ax = axes[i, j]
                ax.plot(F[:, i, j], P_expected[:, i, j], marker='.', markersize=2, label=label)
                ax.set_xlabel(f'Deformation Gradient $F_{{{i+1}{j+1}}}$')
                ax.set_ylabel(f'Stress $P_{{{i+1}{j+1}}}$')
                ax.grid(True)
    
    # Consolidate legends from the first subplot to the top right of the figure window
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Render the interactive plot
    plt.show()

if __name__ == "__main__":
    main()