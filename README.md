# Predictive Quality Monitoring for Additive Manufacturing

## Our Dataset

We have **200 samples** (manufacturing runs) with:

- **96 features**: Numbers that describe the audio, voltage, and current signals (like average values, peaks, frequencies, etc.)
- **4 classes** (types of outcomes):
  - ✅ **ok**: The part was manufactured correctly (50 samples)
  - ⚠️ **oil**: Oil-related error occurred (50 samples)
  - ⚠️ **offset**: Printer head misalignment error (50 samples)
  - ⚠️ **tilt**: Build platform tilt error (50 samples)

Each sample is like a "fingerprint" - the 96 numbers together describe what happened during that manufacturing run.

## The Three Machine Learning Models We Use

We test three different algorithms to see which one works best. Think of them as three different "detective methods":

### 1. **SVM (Support Vector Machine)**

- **Simple explanation**: Draws boundaries (like lines or curves) to separate different classes
- **Analogy**: Like drawing lines on a map to separate "good neighborhood" from "bad neighborhoods"
- **Good for**: Finding clear patterns and boundaries in the data

### 2. **kNN (k-Nearest Neighbors)**

- **Simple explanation**: Looks at the "k closest neighbors" (most similar examples) to make a decision
- **Analogy**: "If my 5 closest neighbors all have oil errors, I probably have an oil error too"
- **Good for**: When similar things tend to have similar outcomes

### 3. **Gradient Boosting (GB)**

- **Simple explanation**: Combines many simple "decision trees" (like a flowchart) that learn from each other's mistakes
- **Analogy**: Like a team of experts where each expert learns from the previous one's mistakes
- **Good for**: Complex patterns that need multiple steps to understand

## What is Feature Selection?

We have **96 features** (numbers), but not all of them are equally useful. Some might be:

- **Very useful**: Strongly indicate whether there's an error
- **Somewhat useful**: Help a little bit
- **Not useful**: Just noise that confuses the model

**Forward Selection** is like building a team:

1. Start with the single best feature
2. Try adding each remaining feature one by one
3. Keep the one that improves performance the most
4. Repeat until adding more features doesn't help

**Why do this separately for each model?** Because different models "think" differently:

- SVM might care most about voltage patterns
- kNN might care most about audio frequencies
- Gradient Boosting might need a mix of different types

So we find the best feature set for each model independently.

## How We Evaluate Models

We split our 200 samples into:

- **Training set (160 samples)**: Used to teach the models
- **Test set (40 samples)**: Used to see how well they learned (like a final exam)

Then we measure performance using 4 metrics:

### 1. **Accuracy**

- **What it means**: Out of 100 predictions, how many were correct?
- **Example**: 90% accuracy = 90 out of 100 predictions were right
- **Simple**: The overall "correctness" score

### 2. **Precision**

- **What it means**: When the model says "this is an oil error," how often is it right?
- **Example**: 85% precision = Out of 100 times it predicted "oil error," 85 were actually oil errors
- **Simple**: "When I say X, how often am I right?"

### 3. **Recall**

- **What it means**: Out of all actual oil errors, how many did the model catch?
- **Example**: 80% recall = Out of 100 real oil errors, it found 80 of them
- **Simple**: "How many problems did I catch?"

### 4. **F1 Score**

- **What it means**: A balance between precision and recall
- **Why it matters**: You want both high precision (not too many false alarms) AND high recall (catch most problems)
- **Simple**: The "balanced" score that considers both

**"Macro" average**: We calculate these metrics for each class (ok, oil, offset, tilt) separately, then average them. This ensures all classes are treated equally, even if one class is easier to predict.

## What This Project Does (Step by Step)

When you run the code, it:

1. **Loads the data** from `dataset/All_Features_New.csv`
2. **Splits the data** into training (80%) and test (20%) sets
3. **Trains baseline models** using all 96 features:
   - Tests SVM with all features
   - Tests kNN with all features
   - Tests Gradient Boosting with all features
4. **Runs forward selection** for each model:
   - Finds the best subset of features for SVM
   - Finds the best subset of features for kNN
   - Finds the best subset of features for Gradient Boosting
5. **Tests the improved models** using only the selected features
6. **Compares results** to see:
   - Which model works best
   - Whether feature selection improved performance
   - How many features each model actually needs

## How to Run

### Prerequisites

- Python 3.8 or higher
- Basic command line knowledge

### Step 1: Install Dependencies

Open a terminal/command prompt in this project folder and run:

```bash
pip install -r requirements.txt
```

This installs:

- `pandas`: For reading and working with data tables
- `numpy`: For mathematical operations
- `scikit-learn`: The machine learning library (contains SVM, kNN, Gradient Boosting)
- `matplotlib`: For creating graphs and plots

### Step 2: Run the Training Pipeline

```bash
python -m src.train_all_models
```

### What You'll See

The program will print progress as it runs:

```
[1/9] Loading dataset...
Loaded dataset: 200 samples, 96 features

[2/9] Performing stratified train/test split...
  Train set: 160 samples
  Test set: 40 samples

[3/9] Building model pipelines...
  ✓ SVM pipeline created
  ✓ kNN pipeline created
  ✓ Gradient Boosting pipeline created

[4/9] Evaluating baseline models (all 96 features)...
  Evaluating SVM baseline...
  Evaluating kNN baseline...
  Evaluating GB baseline...

[5/9] Running forward feature selection...
  Forward Selection for SVM
  [Progress updates...]

[6/9] Evaluating models with selected features...
[7/9] Generating results summary...
[8/9] Saving results...
[9/9] Detailed Classification Reports
```

### Understanding the Results

At the end, you'll see a table like this:

```
Model     Variant     Accuracy    F1-Macro    Precision-Macro    Recall-Macro    Features
SVM       Baseline    0.8500      0.8400      0.8300            0.8500          96
SVM       Selected    0.9000      0.8900      0.8800            0.9000          15
kNN       Baseline    0.8000      0.7900      0.7800            0.8000          96
kNN       Selected    0.8500      0.8400      0.8300            0.8500          22
GB        Baseline    0.8800      0.8700      0.8600            0.8800          96
GB        Selected    0.9200      0.9100      0.9000            0.9200          18
```

**What to look for:**

- **Higher numbers = Better performance** (closer to 1.0 = perfect)
- **Compare Baseline vs Selected**: Did feature selection help?
- **Compare Features column**: How many features did each model actually need?
- **Best overall model**: Which has the highest scores?

### Output Files

After running, you'll find:

- **`results/svm_forward_selection_history.json`**: Detailed history of which features SVM selected
- **`results/knn_forward_selection_history.json`**: Detailed history of which features kNN selected
- **`results/gb_forward_selection_history.json`**: Detailed history of which features Gradient Boosting selected
- **`results/plots/svm_feature_selection_curve.png`**: Graph showing how SVM's performance changed as features were added
- **`results/plots/knn_feature_selection_curve.png`**: Graph showing how kNN's performance changed
- **`results/plots/gb_feature_selection_curve.png`**: Graph showing how Gradient Boosting's performance changed

The graphs show: "As we add more features, does performance keep improving?" Usually, you'll see performance improve quickly at first, then level off.

## Project Structure

```
predictive-final-project/
├── dataset/
│   ├── All_Features_New.csv          # The main data file (200 samples, 96 features)
│   └── audio_voltage_current_plots/   # Visualization plots (not used in ML)
├── src/
│   ├── __init__.py                    # Makes src a Python package
│   ├── data_loading.py                # Loads and prepares the data
│   ├── models.py                      # Creates the three ML models
│   ├── feature_selection.py          # Finds the best features for each model
│   ├── evaluation.py                  # Calculates accuracy, F1, precision, recall
│   └── train_all_models.py           # Main script that runs everything
├── results/                           # Created automatically when you run
│   ├── *.json                         # Feature selection history files
│   └── plots/                         # Performance graphs
├── requirements.txt                   # Python packages needed
└── README.md                          # This file
```

## Key Concepts Summary

| Concept               | Simple Explanation                                                                        |
| --------------------- | ----------------------------------------------------------------------------------------- |
| **Feature**           | A number that describes something about the manufacturing process (e.g., average voltage) |
| **Class**             | The category/type of outcome (ok, oil, offset, tilt)                                      |
| **Training**          | Teaching the model using examples                                                         |
| **Testing**           | Checking if the model learned correctly                                                   |
| **Baseline**          | Using all features (no selection)                                                         |
| **Selected**          | Using only the best features (after selection)                                            |
| **Cross-validation**  | Testing the model multiple times to be sure it works well                                 |
| **Forward Selection** | Building the feature set one feature at a time, keeping the best ones                     |

## Tips for Understanding Results

1. **If Selected > Baseline**: Feature selection helped! The model works better with fewer, better features.
2. **If Selected ≈ Baseline**: All features are useful, or the model doesn't benefit much from selection.
3. **If one model is much better**: That algorithm is better suited for this type of problem.
4. **If all models are similar**: The problem might be relatively easy, or all methods work equally well.

## Common Questions

**Q: Why do we need 3 different models?**  
A: Different models have different strengths. By testing all three, we find which works best for this specific problem.

**Q: Why not just use all 96 features?**  
A: Some features might be redundant or noisy. Using fewer, better features can:

- Improve accuracy (less confusion)
- Train faster
- Be easier to understand
- Reduce overfitting (memorizing instead of learning)

**Q: What if the results are bad?**  
A: Low scores (e.g., < 70%) might mean:

- The features don't contain enough information
- The classes are too similar
- We need different features or models
- More data might help

**Q: What's a "good" score?**  
A: It depends on the problem, but generally:

- **> 90%**: Excellent
- **80-90%**: Good
- **70-80%**: Acceptable
- **< 70%**: Needs improvement

## Next Steps (If You Want to Experiment)

- Try different hyperparameters (model settings) in `src/models.py`
- Adjust the `min_improvement` threshold in forward selection
- Try different train/test split ratios
- Add more evaluation metrics
- Visualize the confusion matrices

---

**Note**: This project focuses on **classical ML models** (SVM, kNN, Gradient Boosting). Other team members may be working on neural networks (NN, CNN) for comparison.
