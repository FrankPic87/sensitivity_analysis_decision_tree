# sensitivity_analysis_decision_tree
This repository contains the Python implementation of the analytical framework presented in the paper **“Comprehensive Framework for Sensitivity Analysis in Decision Tree-based Classification”**, submitted to **IEEE I²MTC 2026**.

The framework provides a **Monte Carlo–free sensitivity analysis** method for Decision Tree classifiers.
Instead of relying on iterative simulations, uncertainties on the input measurements—modeled as Gaussian distributions—are **propagated analytically** through the structure of a trained Decision Tree.
This allows computing the **probability of correct classification** for a given instance in a deterministic, fast, and interpretable way.

---

## **Features**

* - Extracts all decision conditions (feature thresholds and signs) for each leaf
* - Models feature uncertainties as Gaussian PDFs
* - Propagates uncertainties analytically
* - Computes the probability of reaching each leaf
* - Aggregates probabilities to estimate **per-class correct classification probability**
* - Works with any scikit-learn DecisionTreeClassifier

---

## **Repository Contents**

| File                          | Description                                                    |
| ----------------------------- | -------------------------------------------------------------- |
| **`sensitivity_analysis.py`** | Main script implementing the analytical framework              |
| **`tree_model.joblib`**       | Pretrained Decision Tree model (trained on the *Iris* dataset) |

The provided Decision Tree model serves as a **working example** for running and testing the sensitivity analysis.

---

## **Method Overview**

The analytical framework is structured as follows:

### **1. Extract Leaf Conditions**

The function `get_leaf_conditions_and_class()` recursively inspects the tree and extracts:
* the sequence of conditions (feature, threshold, sign) required to reach each leaf
* the predicted class associated with each leaf

### **2. Model Input Uncertainty**

Each feature is assumed to follow a **Gaussian distribution** defined by:
* a theoretical mean
* a theoretical standard deviation

These represent model input uncertainty.

### **3. Analytical Probability Computation**

For each leaf:
* the conditions define an interval for each feature
* the probability that the feature lies in this interval is computed using the **Gaussian CDF** (via `erf`)
* only features that appear in the leaf's path are considered

The product of these probabilities yields the **probability of reaching that leaf**.

### **4. Correct Classification Probability**

Leaf probabilities are aggregated **class-wise** based on the class predicted at each leaf.
The result is a dictionary:

```python
{ class_index : probability_of_correct_classification }
```

---

## **Example Usage**

The following block demonstrates how to compute the analytical classification probability:
```python
probs = compute_class_correct_probs_erf(
    get_leaf_conditions_and_class(tree),
    feature_value,
    feature_std_dev
)
print(f"Analytical Probability of Correct Classification: {probs}")
```

Steps performed:

1. Load the Decision Tree from `tree_model.joblib`
2. Define the theoretical mean and sigma of each feature
3. Run the analytical sensitivity analysis
4. Print the per-class probabilities

This example is already included in `sensitivity_analysis.py`.

---

## **Conda Environment**

A minimal Conda environment is provided.
Below are the **relevant dependencies** required to run the code:

### **Core**

* Python 3.13
* pip
* setuptools
* wheel

### **Scientific stack**

* numpy
* scipy
* scikit-learn
* joblib

### **Optional (plots)**

* matplotlib

These are sufficient to execute the analytical framework and load the provided Decision Tree model.

---

## **Citation**

If you use this code or methodology in academic work, please cite:

**“Comprehensive Framework for Sensitivity Analysis in Decision Tree-based Classification,”
submitted to IEEE I²MTC 2026.**
