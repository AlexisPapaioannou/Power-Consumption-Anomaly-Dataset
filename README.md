# Power Consumption Anomaly Dataset

## Overview
This repository contains a dataset of power consumption readings with anomaly patterns associated with various device malfunctions. The data can be used for developing and testing machine learning models for anomaly detection, predictive maintenance, and other related applications.

## Dataset Description
The dataset consists of power consumption readings from various devices. Each entry includes a ctime, power consumption value (activePower), and an anomaly label (label) indicating whether the reading is normal or indicative of a malfunction.

### Data Fields
- `ctime`: The date and time when the reading was taken.
- `activePower`: The power consumption reading (in watts).
- `label`: Indicates whether the reading is normal (`0`) or an anomaly (`1`).

### Example Data
| ctime               | activePower      | label         |
|---------------------|------------------|---------------|
| 2020-03-19 16:10:00 | 77               | 0             |
| 2020-03-19 16:11:00 | 73               | 0             |
| 2020-03-19 16:12:00 | 73               | 1             |
| 2020-03-19 16:13:00 | 72               | 0             |

## Usage
You can use this dataset to train machine learning models for detecting anomalies in power consumption data. The following is an example of how to load and preprocess the dataset using Python and pandas.

#### Loading the Dataset
```python
import pandas as pd

# Load the dataset
data = pd.read_csv('power_consumption_anomalies.csv')

# Display the first few rows of the dataset
print(data.head())
```

#### Preprocessing
```python
# Convert the timestamp column to datetime format
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Set the timestamp column as the index
data.set_index('timestamp', inplace=True)

# Display summary statistics
print(data.describe())
```

#### Example Model Training
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

# Split the data into training and testing sets
X_train, X_test = train_test_split(data[['power_consumption']], test_size=0.2, random_state=42)
```

## Repository Structure
- `README.md`: This readme file.
- `data/`: Directory containing the dataset files.
- `data/Dishwasher/`: Directory containing the dataset files for 3 different brands of Dishwashers.
- `data/Dryer/`: Directory containing the dataset files for 1 brand of Dryer.
- `data/Fridge/`: Directory containing the dataset files for 3 different brands of Fridge.
- `data/Washing_Machine/`: Directory containing the dataset files for 3 different brands of Washing_Machine.
- `data/Water_Heater/`: Directory containing the dataset files for 3 different brands of Water_Heater.

## Contributing
Contributions are welcome! If you have any suggestions for improving the dataset or examples, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or issues, please contact your-email@example.com.



