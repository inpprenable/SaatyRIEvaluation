# Python Saaty Consistency Index

A Python script to generate the **Random Consistency Index (RI)** for the Analytic Hierarchy Process (AHP) based on the method introduced by **Thomas L. Saaty**.

In Saaty's original work, the RI was computed from 500 random matrices using the Saaty scale of comparison values: [1, 2, 3, 4, 5, 6, 7, 8, 9].  
This implementation allows:
- A higher number of matrices (thanks to the use of PyTorch),
- Customizable Saaty scales (default: [1,3,5,7,9]).

Proposed Random Consistency Indices with their **68% confidence intervals** are saved in the file:  
[`saaty_value.csv`](saaty_value.csv)

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

To generate the Random Consistency Index for square matrices:

```bash
python genRI.py <min_size> <max_size> <nb_experiments>
```

Example:

```bash
python genRI.py 3 15 1000
```

To save the output to a CSV file:

```bash
python genRI.py 3 15 1000 -o output.csv
```

---

## Output

The output CSV contains the following columns:

- `size`: Matrix size
- `avg_CI`: Average consistency index
- `std_CI`: Standard deviation
- `nb_exp`: Number of experiments
- `confidence_interval`: 68% confidence interval

---

## References

Saaty, T. L., & Vargas, L. G. (2012). *Models, methods, concepts & applications of the analytic hierarchy process* (Vol. 175). Springer Science & Business Media.  
[DOI: 10.1007/978-1-4614-3597-6](https://doi.org/10.1007/978-1-4614-3597-6)
