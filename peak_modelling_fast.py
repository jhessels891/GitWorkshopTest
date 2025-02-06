import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma, gammainc
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import matplotlib.pyplot as plt


start = time.time()
# ================================
# 1. Data Loading and Preparation
# ================================

# Load adjusted traffic data
print("Loading adjusted traffic data...")
adjusted_traffic = pd.read_csv("smoothed_web_traffic.csv")
adjusted_traffic = adjusted_traffic.rename(
    columns={"scaled_visits_web": "observed", "smoothed_visits": "baseline", "time_numeric": "minutes_from_start"}
)
adjusted_traffic['datetime'] = pd.to_datetime(adjusted_traffic['datetime'])
adjusted_traffic = adjusted_traffic.sort_values(by="datetime")

# Load ad data
print("Loading ad data...")
grouped_df = pd.read_csv("ad_grouping.csv")

# subsets
grouped_df = grouped_df[:300]
first_ad_date = grouped_df['date'].min()
last_ad_date = grouped_df['date'].max()
adjusted_traffic = adjusted_traffic[(adjusted_traffic['datetime'] >= first_ad_date) & (adjusted_traffic['datetime'] <= last_ad_date)]
print("length of subset: ", len(adjusted_traffic))

# Observed and baseline traffic
Z = adjusted_traffic['observed'].values
lambda_t = adjusted_traffic['baseline'].values
total_time = len(adjusted_traffic)  # Total number of time intervals

# ================================
# 2. Spread Function
# ================================

def generalised_gamma_integral(tau1, tau2, alpha, phi, psi):
    """Integral of the generalized gamma density over [tau1, tau2]."""
    tau1, tau2 = max(0, tau1), max(0, tau2)
    if tau1 >= tau2:
        return 0.0
    numerator = gammainc(psi, (tau1 * alpha) ** phi) - gammainc(psi, (tau2 * alpha) ** phi)
    return numerator / gamma(psi)


def spread_function_limited(t, s, d, alpha, phi, psi):
    """Compute the limited spread function V_s(t) with a cutoff point d."""
    if t <= s:
        return 0.0
    alpha, phi, psi = map(float, [alpha, phi, psi])
    normalization = generalised_gamma_integral(0, d, alpha, phi, psi)
    if isinstance(normalization, np.ndarray):
        normalization = normalization.item()  # Extract scalar if it's an array
    if normalization == 0:
        return 0.0  # Avoid division by zero
    if s < t <= s + d:
        numerator = generalised_gamma_integral(t - s - 1, t - s, alpha, phi, psi)
    else:
        return 0.0
    return numerator / normalization


def compute_spread_matrix(grouped_ads, total_time, alpha, phi, psi):
    """Compute the spread function matrix V for all groups and time intervals in parallel."""
    def compute_group(matrix_index, row):
        start_time = row["minutes_from_start"]
        return [spread_function_limited(t, start_time, d=15, alpha=alpha, phi=phi, psi=psi) for t in range(total_time)]

    V_matrix = Parallel(n_jobs=-1)(
        delayed(compute_group)(matrix_index, row) for matrix_index, (_, row) in enumerate(grouped_ads.iterrows())
    )
    return np.array(V_matrix)

# ================================
# 3. Log-Likelihoods
# ================================


def log_likelihood_theta(theta, Z, V_matrix, lambda_t):
    """Log-likelihood function for optimizing theta."""
    mu_t = np.dot(theta, V_matrix)
    mu_t_total = np.clip(mu_t + lambda_t, 1e-8, None)
    return -np.sum(Z * np.log(mu_t_total) - mu_t_total)


def log_likelihood_gamma(params, Z, grouped_ads, lambda_t, total_time, theta):
    """Log-likelihood function for optimizing spread function parameters."""
    alpha, phi, psi = map(float, params)
    if alpha <= 0 or phi <= 0 or psi <= 0:
        return np.inf
    V_matrix = compute_spread_matrix(grouped_ads, total_time, alpha, phi, psi)
    mu_t = np.dot(theta, V_matrix)
    mu_t_total = np.clip(mu_t + lambda_t, 1e-8, None)
    # ✅ Log-likelihood term
    logL = np.sum(Z * np.log(mu_t_total) - mu_t_total)

    # ✅ Penalty term for large φ (quadratic penalty)
    penalty_strength = 0.1  # Adjust strength of penalization
    penalty = penalty_strength * (phi ** 2)  # Penalize large values of φ

    return -(logL - penalty)

# ================================
# 4. Block Coordinate Descent with Tracking
# ================================
print("Initializing parameters...")
alpha, phi, psi = np.array([0.1, 0.5, 1.0], dtype=np.float64)
# Initialize theta (matching R, starting at 1 for all ads)
theta = np.ones(len(grouped_df))

# Stopping criteria parameters
tolerance = 1e-3
max_outer_iterations = 7  # Max iterations for safety
prev_params = np.array([alpha, phi, psi])  # Track previous parameters

# Track optimization progress
function_values = []
iteration_counts = []
param_history = []  # Track [alpha, phi, psi] per iteration

for iteration in range(max_outer_iterations):
    print(f"\n=== Block Coordinate Descent Iteration {iteration + 1} ===")

    # ✅ Recompute V_matrix before optimizing θ to reflect updated γ
    V_matrix = compute_spread_matrix(grouped_df, total_time, alpha, phi, psi)

    # Step 1: Optimize theta_j per group in parallel
    def optimize_theta_group(group, theta_values):
        group_indices = group.index.to_numpy()
        theta_group = theta_values[group_indices]  # Start from previous theta values

        print(f"Optimizing theta with group length {len(group)}")
        V_matrix_group = V_matrix[group_indices, :]  # ✅ Use updated V_matrix

        result = minimize(
            log_likelihood_theta,
            theta_group,
            args=(Z, V_matrix_group, lambda_t),
            method='L-BFGS-B',
            bounds=[(0, None)] * len(theta_group),  # Constrain theta to ≥ 0
            options={'disp': False, 'maxiter': 200}
        )
        return result.x, group_indices  # Return optimized theta and indices

    print("Optimizing theta in Parallel...")
    theta_updates = Parallel(n_jobs=-1)(
        delayed(optimize_theta_group)(group, theta) for _, group in grouped_df.groupby('group_id')
    )

    for theta_vals, indices in theta_updates:
        theta[indices] = theta_vals  # Update theta globally
    np.set_printoptions(suppress=True, precision=4)  # ✅ Suppresses scientific notation, rounds to 4 decimals

    print("Updated theta:", theta)

    # Step 2: Optimize alpha, phi, psi
    print("Optimizing spread function parameters...")
    result_gamma = minimize(
        log_likelihood_gamma,
        np.array([alpha, phi, psi], dtype=np.float64),
        args=(Z, grouped_df, lambda_t, total_time, theta),
        method='L-BFGS-B',
        bounds=[(1e-4, None), (1e-4, 5), (1e-4, None)],  # Match R's lower bound constraints, constrained phi <5 to test
        options={'disp': True, 'maxiter': 200}
    )
    alpha, phi, psi = result_gamma.x
    soft_max = 5.0  # Adjust if needed
    phi = min(phi, soft_max)  # ✅ Apply soft thresholding
    new_params = np.array([alpha, phi, psi])

    # Track progress
    function_values.append(result_gamma.fun)  # Store function value
    iteration_counts.append(result_gamma.nit)  # Store iterations used
    param_history.append(new_params)  # Store parameter values

    print(f"Optimized alpha: {alpha}, phi: {phi}, psi: {psi}")
    if np.abs(np.mean(new_params - prev_params)) > np.abs(np.mean(theta - theta.mean())):
        print("⚠️ Warning: γ is changing more than θ!")

    # Stopping condition: check if parameters have stabilized
    if np.linalg.norm(new_params - prev_params) < tolerance:
        print(f"\nConverged in {iteration + 1} iterations.")
        break

    # ✅ Update parameters for the next iteration
    prev_params = new_params.copy()

# Final results
print("\nFinal optimized parameters:")
print(f"Theta: {theta}")
print(f"Alpha: {alpha}, Phi: {phi}, Psi: {psi}")

grouped_df['thetas'] = theta # / GRP????
print(grouped_df['thetas'].head())

end_time = time.time()
elapsed_time = end_time - start
print(f"\nTotal Running Time: {elapsed_time:.2f} seconds")

# === PLOT RESULTS ===
plt.figure(figsize=(12, 5))

# Function value over iterations
plt.subplot(1, 2, 1)
plt.plot(function_values, marker='o', linestyle='-')
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood Value')
plt.title('Function Value Over Iterations')

# Parameter evolution
param_history = np.array(param_history)
plt.subplot(1, 2, 2)
plt.plot(param_history[:, 0], label="Alpha")
plt.plot(param_history[:, 1], label="Phi")
plt.plot(param_history[:, 2], label="Psi")
plt.xlabel('Iteration')
plt.ylabel('Parameter Values')
plt.title('Parameter Convergence')
plt.legend()

plt.show()
