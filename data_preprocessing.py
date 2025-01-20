import pandas as pd
from datetime import timedelta
import numpy as np
def filter_data(traffic_data, ads_data):
    # Convert web traffic datetime column to datetime format
    traffic_data['datetime'] = pd.to_datetime(traffic_data['datetime'])

    # Combine 'date' and 'time' columns in commercials data into a single datetime column
    ads_data['airing_time'] = pd.to_datetime(
        ads_data['date'] + ' ' + ads_data['time'],
        format='%m/%d/%Y %I:%M:%S %p'
    )

    # Define the exclusion window (30 minutes after each ad)
    exclusion_window = timedelta(minutes=30)
    ads_data['exclusion_end'] = ads_data['airing_time'] + exclusion_window

    # Efficient filtering using a merge
    # Step 1: Sort both datasets by datetime
    traffic_data = traffic_data.sort_values('datetime').reset_index(drop=True)
    ads_data = ads_data[['airing_time', 'exclusion_end']].sort_values('airing_time').reset_index(drop=True)

    # Step 2: Merge web traffic data with ad airing windows
    merged = pd.merge_asof(
        traffic_data,
        ads_data,
        left_on='datetime',
        right_on='airing_time',
        direction='backward'
    )

    # Step 3: Exclude rows where datetime is within an exclusion window
    filtered_traffic_data = merged[
        ~((merged['datetime'] >= merged['airing_time']) &
          (merged['datetime'] <= merged['exclusion_end']))
    ].drop(columns=['airing_time', 'exclusion_end'])
    # Save the filtered traffic data to a CSV file
    filtered_traffic_data.to_csv('filtered_web_traffic.csv', index=False)
    # Summary
    print(f"Original Data: {len(traffic_data)} rows")
    print(f"Filtered Data: {len(filtered_traffic_data)} rows")

    return filtered_traffic_data

def aggregate_rescale_data(data):
    # Aggregate web visits per minute
    aggr_data = data.groupby('datetime')['visits_web'].sum().reset_index()

    # Rescale web visits
    min_nonzero = np.min(aggr_data['visits_web'][aggr_data['visits_web'] > 0])
    print("min_nonzero:", min_nonzero)
    scaling_factor = 1 / min_nonzero
    print("scaling factor: ", scaling_factor)
    aggr_data['scaled_visits_web'] = np.round(aggr_data['visits_web'] * scaling_factor).astype(int)

    return aggr_data

def rescale_data(data):
    # Rescale web visits
    min_nonzero = np.min(data['visits_web'][data['visits_web'] > 0])
    print("min_nonzero:", min_nonzero)
    scaling_factor = 1 / min_nonzero
    print("scaling factor: ", scaling_factor)
    data['scaled_visits_web'] = np.round(data['visits_web'] * scaling_factor).astype(int)

    return data