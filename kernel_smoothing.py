import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from data_preprocessing import filter_data, aggregate_rescale_data


def gaussian_kernel(u):
    """
    Gaussian kernel function.
    """
    return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)


def local_linear_kernel_smoother(x, y, h, num_workers=None):
    """
    Local linear regression smoother using Gaussian kernel weights.
    """
    n = len(y)
    smoothed = np.zeros(n)

    def compute_smoothed_point(i):
        x_center = x[i]
        # Neighborhood within +/- 3h
        left_bound = x_center - 3 * h
        right_bound = x_center + 3 * h
        mask = (x >= left_bound) & (x <= right_bound)

        x_neighbors = x[mask]
        y_neighbors = y[mask]
        distances = x_neighbors - x_center

        # Compute Gaussian weights
        weights = gaussian_kernel(distances / h)

        # Local linear regression design: [1, (x - x_center)]
        X = np.column_stack((np.ones_like(distances), distances))
        W = np.diag(weights)
        XtW = X.T @ W

        try:
            beta = np.linalg.pinv(XtW @ X) @ (XtW @ y_neighbors)
        except np.linalg.LinAlgError:
            print("error in computing beta")
            # Fallback if there's a numerical issue
            beta = np.array([y[i], 0.0])
        return beta[0]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(compute_smoothed_point, range(n)),
            total=n, desc=f"Smoothing (h={h})"
        ))
    smoothed = np.array(results)
    return smoothed


# 3) Scoring Functions
def poisson_log_likelihood(y_true, y_pred):
    """
    Poisson log-likelihood.
    Higher is better. Typical range might be negative.
    """
    y_obs = np.round(y_true).astype(int)
    lambdas = np.maximum(y_pred, 1e-8)

    max_y = np.max(y_obs)
    if max_y == 0:
        # If everything is 0, or near 0, return a very poor LL
        return -1e9

    # Precompute log-factorials
    log_factorials = np.zeros(max_y + 1)
    for k in range(1, max_y + 1):
        log_factorials[k] = log_factorials[k - 1] + np.log(k)

    ll = np.mean(y_obs * np.log(lambdas) - lambdas - log_factorials[y_obs])
    return ll


def mean_absolute_deviation_score(y_true, y_pred):
    """
    Mean Absolute Deviation (MAD) - we negate it so higher is "better."
    Normally you would minimize MAD, so to keep the same logic (maximize),
    we return negative MAD.
    """
    return -np.mean(np.abs(y_true - y_pred))


def mean_squared_error_score(y_true, y_pred):
    return -np.mean((y_true - y_pred) ** 2)


def get_score(y_true, y_pred, method="poisson"):
    if method == "poisson":
        return poisson_log_likelihood(y_true, y_pred)
    elif method == "mad":
        return mean_absolute_deviation_score(y_true, y_pred)
    elif method == "mse":
        return mean_squared_error_score(y_true, y_pred)
    else:
        raise ValueError(f"Unknown scoring method: {method}")


# --------------------------------------------------
# 4) Bandwidth Selection via Cross-Validation
# --------------------------------------------------
def select_optimal_bandwidth(x, y, bandwidths, folds=5, num_workers=None, scoring="mad"):
    best_score = -np.inf
    best_bandwidth = None
    scores_list = []

    for h in tqdm(bandwidths, desc="Bandwidth Search"):
        cv_scores = []
        kf = KFold(n_splits=folds, shuffle=True)
        for train_idx, val_idx in kf.split(x):
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]

            # Smooth training portion
            y_train_smoothed = local_linear_kernel_smoother(
                x_train, y_train, h, num_workers=num_workers
            )

            # Approximate predictions on the validation set
            # (For a more exact approach, you'd re-run the smoother for x_val points specifically.
            #  But here, for demonstration, we map x_val to the nearest x_train.)
            y_val_pred = []
            for xv in x_val:
                idx_nearest = np.argmin(np.abs(x_train - xv))
                y_val_pred.append(y_train_smoothed[idx_nearest])
            y_val_pred = np.array(y_val_pred)

            # Calculate the chosen score
            score = get_score(y_val, y_val_pred, method=scoring)
            cv_scores.append(score)

        avg_score = np.mean(cv_scores)
        scores_list.append((h, avg_score))
        if avg_score > best_score:
            best_score = avg_score
            best_bandwidth = h

    # Optional: Plot the bandwidth vs. score
    hvals, llvals = zip(*scores_list)
    plt.figure(figsize=(8, 5))
    plt.plot(hvals, llvals, marker='o')
    plt.title(f"Cross-Val '{scoring}' vs. Bandwidth")
    plt.xlabel("Bandwidth")
    plt.ylabel(f"Mean {scoring.upper()}")
    plt.grid(True)
    plt.show()

    return best_bandwidth


# Main Demonstration (Choose subset size + Score method)
def main(filtered_data, data, max_observations=None, scoring="mad"):
    # 5a) Read and prepare data
    aggr_data = filtered_data
    aggr_data['datetime'] = pd.to_datetime(aggr_data['datetime'])
    aggr_data['time_numeric'] = (aggr_data['datetime'] - aggr_data[
        'datetime'].min()).dt.total_seconds() / 60.0  # in minutes

    # Optionally limit to a smaller subset
    if max_observations is not None and max_observations < len(aggr_data):
        aggr_data = aggr_data.head(max_observations)
        print(f"Using only the first {max_observations} observations for testing.")

    # Replace zero visits with a small constant
    aggr_data.loc[aggr_data['scaled_visits_web'] == 0, 'scaled_visits_web'] = 1e-6

    x_vals = aggr_data['time_numeric'].values
    y_vals = aggr_data['scaled_visits_web'].values

    # 5b) Select a bandwidth via cross-validation with chosen scoring
    candidate_bandwidths = range(2, 40, 3)
    best_h = select_optimal_bandwidth(
        x_vals, y_vals, bandwidths=candidate_bandwidths,
        folds=3, num_workers=8, scoring=scoring
    )
    print(f"Best bandwidth found (scoring={scoring}): {best_h}")

    # Apply the best bandwidth to full data set
    full_data = data
    full_data['datetime'] = pd.to_datetime(full_data['datetime'])
    full_data['time_numeric'] = (full_data['datetime'] - full_data['datetime'].min()).dt.total_seconds() / 60.0

    x = full_data['time_numeric'].values
    y = full_data['scaled_visits_web'].values
    smoothed_y = local_linear_kernel_smoother(x, y, h=best_h, num_workers=None)
    full_data['smoothed_visits'] = smoothed_y
    full_data.to_csv("smoothed_web_traffic.csv", index=False)

    # 5d) Plot original vs smoothed
    plt.figure(figsize=(12, 6))
    plt.plot(full_data['datetime'], full_data['scaled_visits_web'],
             label='Original Visits', alpha=0.6)
    plt.plot(full_data['datetime'], full_data['smoothed_visits'],
             label='Smoothed Visits', color='red', linewidth=2)
    plt.legend()
    plt.xlabel("Datetime")
    plt.ylabel("Website Visits")
    plt.title("Local Linear Kernel Smoothing of Website Visits")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage:
    web_traffic = pd.read_csv('web_data_with_product_types.csv')
    ad_data = pd.read_csv('Web + broadcasting data - Broadcasting data.csv', delimiter=';')

    # Aggregate and rescale full data
    scaled_full_data = aggregate_rescale_data(web_traffic)

    # Remove 30-minute periods after each ad
    filtered_traffic_data = filter_data(web_traffic, ad_data)

    # Aggregate and rescale filtered data
    scaled_filtered_web_traffic = aggregate_rescale_data(filtered_traffic_data)

    # Test with 50k rows and MAD scoring
    main(
        filtered_data=scaled_filtered_web_traffic,
        data=scaled_full_data,
        max_observations=None,
        scoring="mad")
