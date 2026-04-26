# Data

The raw dataset is not committed to this repository.

For local runs, place the CSV file here:

```text
data/raw/IMDB Dataset.csv
```

Then set this inside the notebook:

```python
DATA_PATH_OVERRIDE = "data/raw/IMDB Dataset.csv"
```

Expected columns:

```text
review
sentiment
```
