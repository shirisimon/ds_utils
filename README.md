
## Objective

Utility functions for faster exploratory data analysis (EDA), data preprocessing and predictive models evaluation. 
The evaluation module includes mainly functions which compute evaluation metrics adjusted to the population distribution after over-sampling.

## Motivation

Making it faster and easier to conduct data validation and evaluation. 

## Examples

Plot the distributions of all the columns in the dataframe overlaid with respect to the values in the 'target' column:
```python
eda.Basic().plot_multiple_distributions_overlay(df, 'target')
```

Calculate the adjusted lift at the the top % of the population, given the predictions, the positive label ratio in the population, and the population size:

```python
evaluation.adjusted_lift(predictions_data, target_column, positive_label, top_perc_of_population, 
population_ratio, population_size)
```

## Installation

Download the 'utils' directory to your local machine.
At the top of your code add:
```python
import sys
sys.path.insert(0, "/utils/directory/path")
```
Then to import, for example, the eda module add:
```python
from utils import eda
```
