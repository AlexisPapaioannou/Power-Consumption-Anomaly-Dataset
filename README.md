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

### Specific Malfunctions

| Device                     | Malfunction (%)   | Part                                                  |
|:---------------------------|:-----------------|:------------------------------------------------------|
| Refrigerators and Freezers | 10%              | Damaged Door Seals                                    |
|                           | 23%             | Faulty Thermostats                                    |
|                           | 50%              | Compressor                                            |
|                           | 7.5%            | Minor                                                 |
|                           | 15%, 70%         | Major                                                 |
| Washing machine            | 15%             | Water Heater Malfunctions                             |
|                           | 10%              | Sensor Malfunctions:                                  |
|                           | 15%             | Worn Out Belts or Motors                              |
|                           | 76%             | Heating phase                                         |
|                           | 10%              | Minor                                                 |
|                           | 20%, 30%         | Major                                                 |
| Dishwashers               | 12.50            | Heating Element Issues                                |
|                           | 7.5%            | Water Inlet Valve Problems:                           |
|                           | 15%             | Faulty Thermostat:                                    |
|                           | 10%              | Spray Arm Issues                                      |
|                           | 34%            | Heating phase                                         |
|                           | 7.5%            | Minor                                                 |
|                           | 20%, 35%         | Major                                                 |
| Dryer                     | 12.5%            | Clogged Lint Filter                                   |
|                           | 15%             | Faulty Thermostat:                                    |
|                           | 7.5%            | Worn Seals:                                           |
|                           | 25%             | Inefficient Heating Elements or Gas Valves            |
|                           | 7.5%            | Minor                                                 |
|                           | 20%, 35%         | Major                                                 |
| Water heater              | 15%             | Faulty Thermostats:                                   |
|                           | 18%             | Sediment Build-Up                                     |
|                           | 25%             | Malfunctioning Heating Element (for electric heaters) |
|                           | 50%              | Slight overheating                                    |
|                           | 10%              | Minor                                                 |
|                           | 15%             | Major                                                 |

## Usage
You can use this dataset to train machine learning models for detecting anomalies in power consumption data. The following is an example of how to load and preprocess the dataset using Python and pandas.

#### Loading the Dataset
```python
import pandas as pd

# Load the dataset
data = pd.read_csv('data/Dishwasher/Dishwasher_1/anomaly_Faulty_Thermostat/dishwasher_1_day10_ANOMALIES.csv')

# Display the first few rows of the dataset
print(data.head())
```

#### Preprocessing
```python
# Convert the timestamp column to datetime format
data['ctime'] = pd.to_datetime(data['ctime'])

# Set the timestamp column as the index
data.set_index('ctime', inplace=True)

# Display summary statistics
print(data.describe())
```

#### Example Model Training
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

# Split the data into training and testing sets
X_train, X_test = train_test_split(data[['activePower']], test_size=0.2, random_state=42)
*
*
*
*
**
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



