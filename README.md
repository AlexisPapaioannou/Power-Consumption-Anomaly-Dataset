# Power Consumption Anomaly Dataset

## Overview
This repository contains a dataset of power consumption readings with anomaly patterns associated with various device malfunctions. The data can be used for developing and testing machine learning models for anomaly detection, predictive maintenance, and other related applications.

## Dataset Description
The dataset consists of power consumption readings from various devices. Each entry includes a timestamp, device identifier, power consumption value, and an anomaly label indicating whether the reading is normal or indicative of a malfunction.

### Data Fields
- `timestamp`: The date and time when the reading was taken.
- `device_id`: A unique identifier for the device.
- `power_consumption`: The power consumption reading (in watts).
- `anomaly_label`: Indicates whether the reading is normal (`0`) or an anomaly (`1`).

### Example Data
| timestamp           | device_id | power_consumption | anomaly_label |
|---------------------|-----------|-------------------|---------------|
| 2024-05-01 00:00:00 | device_01 | 150               | 0             |
| 2024-05-01 00:01:00 | device_01 | 145               | 0             |
| 2024-05-01 00:02:00 | device_01 | 200               | 1             |
| 2024-05-01 00:03:00 | device_02 | 180               | 0             |

## Usage
You can use this dataset to train machine learning models for detecting anomalies in power consumption data. The following is an example of how to load and preprocess the dataset using Python and pandas.

### Loading the Dataset
```python
import pandas as pd

# Load the dataset
data = pd.read_csv('power_consumption_anomalies.csv')

# Display the first few rows of the dataset
print(data.head())

### Preprocessing
```python
# Convert the timestamp column to datetime format
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Set the timestamp column as the index
data.set_index('timestamp', inplace=True)

# Display summary statistics
print(data.describe())

### Example Model Training
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

# Split the data into training and testing sets
X_train, X_test = train_test_split(data[['power_consumption']], test_size=0.2, random_state=42)


### Repository Structure
- `README.md`: This readme file.
- `power_consumption_anomalies.csv`: The dataset file.
- `notebooks/`: Jupyter notebooks with examples of data analysis and model training.
- `scripts/`: Python scripts for preprocessing and model training.


### Contributing
Contributions are welcome! If you have any suggestions for improving the dataset or examples, please open an issue or submit a pull request.

### License
This project is licensed under the MIT License. See the LICENSE file for details.

### Contact
For any questions or issues, please contact your-email@example.com.
# Train an Isolation Forest model
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)

# Predict anomalies on the test set
y_pred = model.predict(X_test)



