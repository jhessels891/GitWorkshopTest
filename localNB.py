import numpy as np
import pandas as pd
from patsy import dmatrix
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import NegativeBinomial
import matplotlib.pyplot as plt

from data_preprocessing import filter_data, aggregate_rescale_data

# Load datasets
web_traffic = pd.read_csv('web_data_with_product_types.csv')
ad_data = pd.read_csv('Web + broadcasting data - Broadcasting data.csv', delimiter=';')

# Aggregate and rescale full data
scaled_full_data = aggregate_rescale_data(web_traffic)
print("len(scaled_web_traffic)", len(scaled_full_data))
print("mean and var full scaled data:", np.mean(scaled_full_data['scaled_visits_web']), np.var(scaled_full_data['scaled_visits_web']))

# Remove 30-minute periods after each ad
filtered_traffic_data = filter_data(web_traffic, ad_data)

# Aggregate and rescale filtered data
scaled_filtered_web_traffic = aggregate_rescale_data(filtered_traffic_data)

# Use the filtered dataset
data = scaled_filtered_web_traffic

# Step 1: Extract hour_of_day and day_of_week from 'datetime'
data['datetime'] = pd.to_datetime(data['datetime'])
data['hour_of_day'] = data['datetime'].dt.hour + data['datetime'].dt.minute / 60
data['day_of_week'] = data['datetime'].dt.dayofweek  # 0 = Monday, 6 = Sunday

# Step 2: Create spline basis for 'hour_of_day' and 'day_of_week'
hour_of_day_spline = dmatrix("bs(hour_of_day, df=6, include_intercept=False)", data, return_type='dataframe')
day_of_week_spline = dmatrix("bs(day_of_week, df=6, include_intercept=False)", data, return_type='dataframe')

# Combine spline basis into a single design matrix
X = pd.concat([hour_of_day_spline, day_of_week_spline], axis=1)
y = data['scaled_visits_web']

# Step 3: Fit a Negative Binomial GAM
model = GLM(y, X, family=NegativeBinomial())
result = model.fit()

# Print model summary
print(result.summary())

# Step 4: Visualize smoothers
plt.figure(figsize=(12, 6))

# Hour of day effects
hour_smooth = result.params[:hour_of_day_spline.shape[1]]
plt.subplot(1, 2, 1)
plt.plot(hour_of_day_spline @ hour_smooth, label='Hour of Day Effect')
plt.xlabel('Hour of Day')
plt.ylabel('Effect')
plt.title('Smoother for Hour of Day')
plt.legend()

# Day of week effects
day_smooth = result.params[hour_of_day_spline.shape[1]:]
plt.subplot(1, 2, 2)
plt.plot(day_of_week_spline @ day_smooth, label='Day of Week Effect')
plt.xlabel('Day of Week')
plt.ylabel('Effect')
plt.title('Smoother for Day of Week')
plt.legend()

plt.tight_layout()
plt.show()

def local_nb_regression(data, target_time, bandwidth):
    """
    Local Negative Binomial Regression.

    Parameters:
    - data (pd.DataFrame): Data containing predictors and response.
    - target_time (float): The time for which to compute the local regression.
    - bandwidth (float): Kernel bandwidth for weighting.

    Returns:
    - predicted_value (float): Predicted response for the target time.
    """
    # Kernel weights
    weights = np.exp(-((data['time'] - target_time) ** 2) / (2 * bandwidth ** 2))
    weights /= weights.sum()

    # Fit Negative Binomial GLM
    model = GLM(data['response'], data[['hour_of_day', 'day_of_week']], family=NegativeBinomial(), freq_weights=weights)
    result = model.fit()

    # Predict for target time
    predicted_value = result.predict(target_time)
    return predicted_value
