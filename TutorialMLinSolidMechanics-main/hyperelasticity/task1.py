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

def compute_S_and_E(F, P):
    """
    Converts Deformation Gradient (F) and First Piola-Kirchhoff Stress (P)
    into the symmetric, work-conjugate Reference Configuration pair:
      - Green-Lagrange Strain Tensor: E = 0.5 * (F^T * F - I)
      - Second Piola-Kirchhoff Stress Tensor: S = F^-1 * P
      
    Inputs:
      F: numpy array of shape (N, 3, 3)
      P: numpy array of shape (N, 3, 3)
    Outputs:
      E: numpy array of shape (N, 3, 3)
      S: numpy array of shape (N, 3, 3)
    """
    # Compute Right Cauchy-Green tensor C = F^T * F for all N load steps
    # 'nij,nik->njk' transposes the first matrix (F^T) and multiplies by the second (F)
    C = np.einsum('nij,nik->njk', F, F)
    
    # Compute Green-Lagrange strain tensor E = 0.5 * (C - I)
    I = np.eye(3)
    E = 0.5 * (C - I)
    
    # Invert F for each load step to pull P back to the reference configuration
    F_inv = np.linalg.inv(F)
    
    # Compute Second Piola-Kirchhoff stress S = F^-1 * P
    S = np.einsum('nij,njk->nik', F_inv, P)
    
    return E, S

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
        'Mixed Test': ('test', 'mixed_test.txt'),
        'Biaxial Test': ('test', 'biax_test.txt')
    }
    
    # Setup a 3x3 grid window to plot all tensor stress components against deformation components
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('All 9 Components Across All Load Paths: $P_{ij}$ vs load steps', fontsize=16)
    # Setup Figure 2: Work-Conjugate Stress-Strain (S vs E) and Energy (W) Comparisons
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle(r'Transversely Isotropic Analysis: Work-Conjugate $\mathbf{S}$ vs $\mathbf{E}$ & Strain Energy $W$', fontsize=16)
    # Define distinct line styles and markers for each dataset to prevent complete overlap
    styles = {
        'Uniaxial':   {'linestyle': '-',  'marker': '.', 'linewidth': 2.0},
        'Biaxial':    {'linestyle': '--', 'marker': 'x', 'linewidth': 1.5},
        'Pure Shear': {'linestyle': '-.', 'marker': '^', 'linewidth': 1.5},
        'Mixed Test': {'linestyle': ':',  'marker': 'o', 'linewidth': 1.5}
    }
    
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
                # Fetch style configuration for the current dataset, with a safe fallback
                cfg = styles.get(label, {'linestyle': '-', 'marker': '.', 'linewidth': 1.5})
                 # Create a load step array from 1 to N (number of rows in the dataset)
                load_steps = np.arange(1, len(P_expected) + 1)
                ax.plot(load_steps, P_expected[:, i, j], 
                        linestyle=cfg['linestyle'], 
                        marker=cfg['marker'], 
                        markersize=3, 
                        linewidth=cfg['linewidth'], 
                        alpha=0.8, 
                        label=label)
                        
                ax.set_xlabel('Load Step')
                ax.set_ylabel(f'Stress $P_{{{i+1}{j+1}}}$')
                ax.grid(True)

        # ---------------------------------------------------------
        # NEW: WORK-CONJUGATE (S vs E) & ENERGY (W) VISUALIZATION
        # ---------------------------------------------------------
        # Convert F and P_expected to Green-Lagrange Strain (E) and 2nd PK Stress (S)
        E_tensor, S_tensor = compute_S_and_E(F, P_expected)
        
        # Extract volumetric strain (J - 1) and structural invariant I4 from computed invariants
        J_val = I_computed[:, 1]
        I4_val = I_computed[:, 2]
        vol_strain = J_val - 1.0
        
        # Fetch the dataset line style from our styles dictionary
        cfg = styles.get(label, {'linestyle': '-', 'marker': '.', 'linewidth': 1.5})
        plot_kwargs = dict(linestyle=cfg['linestyle'], marker=cfg['marker'], 
                           markersize=3, linewidth=cfg['linewidth'], alpha=0.8, label=label)

        # 1. Top-Left: Fiber Path (S_11 vs E_11) -> Stiff longitudinal fiber direction (e_1)
        axes2[0, 0].plot(E_tensor[:, 0, 0], S_tensor[:, 0, 0], **plot_kwargs)
        axes2[0, 0].set_title('Fiber Direction Path ($S_{11}$ vs $E_{11}$)')
        axes2[0, 0].set_xlabel(r'Longitudinal Strain $E_{\parallel} = E_{11}$')
        axes2[0, 0].set_ylabel(r'PK2 Stress $S_{11}$')
        axes2[0, 0].grid(True)

        # 2. Top-Middle: Transverse Path (S_22 vs E_22) -> Compliant isotropic matrix plane (e_2)
        axes2[0, 1].plot(E_tensor[:, 1, 1], S_tensor[:, 1, 1], **plot_kwargs)
        axes2[0, 1].set_title('Transverse Matrix Path ($S_{22}$ vs $E_{22}$)')
        axes2[0, 1].set_xlabel(r'Transverse Strain $E_{\perp} = E_{22}$')
        axes2[0, 1].set_ylabel(r'PK2 Stress $S_{22}$')
        axes2[0, 1].grid(True)

        # 3. Top-Right: Shear Coupling Path (S_12 vs 2*E_12) -> Fiber-matrix shear
        # Using engineering shear strain Gamma_12 = 2 * E_12 (or E_12 directly)
        axes2[0, 2].plot(E_tensor[:, 0, 1], S_tensor[:, 0, 1], **plot_kwargs)
        axes2[0, 2].set_title('Fiber-Matrix Shear Coupling ($S_{12}$ vs $E_{12}$)')
        axes2[0, 2].set_xlabel(r'Shear Strain $E_{12} = \frac{1}{2}\Gamma_{\mathrm{shear}}$')
        axes2[0, 2].set_ylabel(r'PK2 Shear Stress $S_{12}$')
        axes2[0, 2].grid(True)

        # 4. Bottom-Left: Out-of-Plane Transverse Path (S_33 vs E_33)
        axes2[1, 0].plot(E_tensor[:, 2, 2], S_tensor[:, 2, 2], **plot_kwargs)
        axes2[1, 0].set_title('Out-of-Plane Transverse Path ($S_{33}$ vs $E_{33}$)')
        axes2[1, 0].set_xlabel(r'Transverse Strain $E_{33}$')
        axes2[1, 0].set_ylabel(r'PK2 Stress $S_{33}$')
        axes2[1, 0].grid(True)

        # 5. Bottom-Middle: Strain Energy W vs Fiber Strain E_11
        axes2[1, 1].plot(E_tensor[:, 0, 0], W_expected, **plot_kwargs)
        axes2[1, 1].set_title(r'Stored Energy $W$ vs Fiber Strain $E_{\parallel}$')
        axes2[1, 1].set_xlabel(r'Fiber Strain $E_{\parallel} = E_{11} = \frac{1}{2}(\lambda_{\mathrm{fiber}}^2 - 1)$')
        axes2[1, 1].set_ylabel(r'Strain Energy Density $W$')
        axes2[1, 1].grid(True)

        # 6. Bottom-Right: Strain Energy W vs Volumetric Strain (J - 1)
        axes2[1, 2].plot(vol_strain, W_expected, **plot_kwargs)
        axes2[1, 2].set_title(r'Stored Energy $W$ vs Volumetric Strain ($J - 1$)')
        axes2[1, 2].set_xlabel(r'Volumetric Strain $J - 1 = \det(\mathbf{F}) - 1$')
        axes2[1, 2].set_ylabel(r'Strain Energy Density $W$')
        axes2[1, 2].grid(True)
    
    # Consolidate legends from the first subplot to the top right of the figure window
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Add legend and layout formatting for Figure 2
    handles2, labels2 = axes2[0, 0].get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))
    fig2.legend(by_label2.values(), by_label2.keys(), loc='upper right', fontsize=12)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Render the interactive plot
    plt.show()

if __name__ == "__main__":
    main()