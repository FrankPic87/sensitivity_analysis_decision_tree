import numpy as np
import math
import joblib

def aggregate_conditions_by_class(leaf_info):
    """
    Group all leaf conditions by class.

    Parameters:
        leaf_info: dict {leaf_id: {"conditions": [(f, thr, sign), ...], "class": int}}

    Returns:
        dict {class: list of conditions [(f, thr, sign), ...]}.
        Each list contains all the conditions of the leaves belonging to that class.
    """
    class_conditions = {}

    for leaf_id, info in leaf_info.items():
        cls = info["class"]
        if cls not in class_conditions:
            class_conditions[cls] = []
        # Add the conditions of this leaf to the corresponding class
        class_conditions[cls].append(info["conditions"])
    
    return class_conditions

# --- Function to extract leaf conditions numerically ---
def get_leaf_conditions_and_class(tree):
    """
    Returns a dictionary:
    {
        leaf_id: {
            "conditions": [(feature_index, threshold, sign), ...],
            "class": class_index
        }
    }

    Where:
        - sign = +1  → (x[feature] - threshold) > 0
        - sign = -1  → (x[feature] - threshold) <= 0
    """
    leaf_info = {}

    def recurse(node_id, path):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        # If it is a leaf
        if left == right:
            class_index = np.argmax(tree.value[node_id])
            leaf_info[node_id] = {
                "conditions": path.copy(),
                "class": int(class_index)
            }
            return

        feature = tree.feature[node_id]
        threshold = tree.threshold[node_id]

        # Left: (x[feature] - threshold) <= 0
        recurse(left, path + [(feature, threshold, -1)])
        # Right: (x[feature] - threshold) > 0
        recurse(right, path + [(feature, threshold, +1)])

    recurse(0, [])
    return leaf_info

def compute_class_correct_probs_erf(leaf_info, feature_theoretical_val, feature_theoretical_sigma):
    """
    Calculate the probability of correct classification for each class using erf,
    considering only the features that appear in the leaf's conditions.

    Parameters:
        leaf_info: dict {leaf_id: {"conditions": [(f, thr, sign), ...], "class": int}}
        feature_theoretical_val: list or array of feature means
        feature_theoretical_sigma: list or array of feature standard deviations

    Returns:
        dict {class: correct_probability}
    """
    n_classes = max(info["class"] for info in leaf_info.values()) + 1
    
    leaf_probs = {}  # leaf_id -> per-class probability

    def prob_feature_in_interval(f, lower, upper):
        mu = feature_theoretical_val[f]
        sigma = feature_theoretical_sigma[f]
        # CDF using erf
        def cdf(x):
            if np.isneginf(x):
                return 0.0
            return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))
        return max(0.0, cdf(upper) - cdf(lower))

    # --- Calculate probability for each leaf considering only the features present in its conditions ---
    for leaf_id, info in leaf_info.items():
        # Intervals for the features present in the leaf
        intervals = {}  # f -> [lower, upper]
        for (f, thr, s) in info["conditions"]:
            if f not in intervals:
                intervals[f] = [-np.inf, np.inf]
            if s == -1:  # <=0
                intervals[f][1] = min(intervals[f][1], thr)
            else:        # >0
                intervals[f][0] = max(intervals[f][0], np.nextafter(thr, np.inf))
        # Calculate per-class probabilities for this leaf
        probs = np.zeros(n_classes)
        for cls in range(n_classes):
            p = 1.0
            for f in intervals:  # Only consider features present in the leaf's conditions
                lower, upper = intervals[f]
                p *= prob_feature_in_interval(f, lower, upper)
            probs[cls] = p
        leaf_probs[leaf_id] = probs

    # --- Aggregate correct classification probabilities per class ---
    class_correct_prob = np.zeros(n_classes)
    for leaf_id, probs in leaf_probs.items():
        pred_cls = leaf_info[leaf_id]["class"]
        for cls in range(n_classes):
            if pred_cls == cls:
                class_correct_prob[cls] += probs[cls]

    return {cls: class_correct_prob[cls] for cls in range(n_classes)}

if __name__ == "__main__":
    
    # Load the pre-trained model
    model = joblib.load('tree_model.joblib')
    tree = model.tree_

    # Definitionf of feature values and standard deviations (Gaussian pdf)
    std_dev = 0.2
    feature_std_dev = [std_dev, std_dev, std_dev, std_dev]
    feature_value= [0, 0, 4.7, 0.7]
    
    # Calculate analytical results using the mathematical model
    probs = compute_class_correct_probs_erf(
                    get_leaf_conditions_and_class(tree),
                    feature_value,
                    feature_std_dev
                )
    print(f"Analytical Probability of Correct Classification: {probs}")