# outlier_plotting
More graceful handling of outliers in plots. Currently supports most of seaborn categorical scatter, distributional and estimate plots.

## Functionality

`handle_outliers` remove outliers from the plot and show them as text boxes. It can be used with most `seaborn` plotting function that works with long-form data as well as `kdeplot`.

Notable exceptions are:

- `countplot`
- `lineplot`
- `scatterplot`

Please not that only inliers are passed into the plotting function, consequently density estimates and functionals are only computed on that subset and **not** representative of the whole data.

### Example

```python
from outlier_plotting.sns import handle_outliers
import seaborn as sns
from matplotlib import pyplot as plt

plt.title('Showing Outliers')
sns.boxplot(data=df, y = 'type', x='value')
plt.show()

plt.title('With Outlier Handling')
handle_outliers(data=df, y = 'type', x='value', plotter=sns.boxplot)
plt.show()
```


![png](https://github.com/wendli01/outlier_plotting/raw/master/images/output_7_0.png)

![png](https://github.com/wendli01/outlier_plotting/raw/master/images/output_7_1.png)

For more examples, see [examples.ipynb](https://github.com/wendli01/outlier_plotting/blob/master/examples.ipynb).


# Installation

## conda

`conda env create -f environment.yml`

##PyPI

[`pip install outlier-plotting`](https://pypi.org/project/outlier-plotting/)
