<a href="https://sandbox.zenodo.org/doi/10.5072/zenodo.77044"><img src="https://sandbox.zenodo.org/badge/807105539.svg" alt="DOI"></a>


# Power Consumption Anomaly Dataset

## Overview
This repository contains a dataset of power consumption readings with anomaly patterns associated with various device malfunctions. The data can be used for developing and testing machine learning models for anomaly detection, predictive maintenance, and other related applications.

## Table of Contents
- [Dataset Description](#dataset-description)
- [Data Fields](#data-fields)
- [Example Data](#example-data)
- [Specific Malfunctions](#specific-malfunctions)
- [Usage](#usage)
  - [Loading the Dataset](#loading-the-dataset)
  - [Preprocessing](#preprocessing)
  - [Plot Data](#plot-data)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

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
| 2020-03-19 16:12:00 | 73               | 0             |
| 2020-03-19 16:13:00 | 72               | 0             |

### Specific Malfunctions

<img src="https://github.com/AlexisPapaioannou/Power-Consumption-Anomaly-Dataset/assets/48021731/b87d4975-b102-45f4-9569-68e1e68ed291" alt="Error Descriptions" width="400"/>

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

#### Plot data
```python
plt.figure(figsize=(10, 6))
plt.plot(data.index , data.activePower , label='Power Consumption')
plt.xlabel('Time')
plt.ylabel('Power Consumption')
plt.title('Dishwasher Power Consumption with Anomalies')
plt.legend()
plt.show()
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
For any questions or issues, please contact alexis.papaioannou13@gmail.com

## Acknowledgments
Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

- **[pandas](https://pandas.pydata.org/)**
- **[scikit-learn](https://scikit-learn.org/)**
- **[Matplotlib](https://matplotlib.org/)**
- **[Stack Overflow](https://stackoverflow.com/)**
- **Open Source Community**
- **Online Courses and Tutorials**





