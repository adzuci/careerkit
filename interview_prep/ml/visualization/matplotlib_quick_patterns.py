"""
Matplotlib Quick Patterns for ML Interviews

These are the visualization patterns you'll use most often.
Focus on clarity over aesthetics - interviewers want to see you
can quickly visualize data to inform decisions.

Usage:
    python interview_prep/ml/visualization/matplotlib_quick_patterns.py
"""

import numpy as np

# Import matplotlib with backend that works without display
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def basic_line_plot():
    """
    Line plot for training curves, time series, etc.
    
    Use for: Loss curves, metric progression, time series
    """
    print("Creating: Basic Line Plot")
    
    epochs = np.arange(1, 11)
    train_loss = np.exp(-epochs * 0.3) + np.random.randn(10) * 0.05
    val_loss = np.exp(-epochs * 0.25) + np.random.randn(10) * 0.05 + 0.1
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, 'b-', label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, 'r--', label='Val Loss', marker='s')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save instead of show (for non-interactive environments)
    plt.savefig('/tmp/line_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to /tmp/line_plot.png")


def histogram_distribution():
    """
    Histogram for understanding data distributions.
    
    Use for: Feature distributions, target variable balance, residuals
    """
    print("Creating: Histogram")
    
    data = np.concatenate([
        np.random.normal(0, 1, 1000),
        np.random.normal(4, 0.5, 500)
    ])
    
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=50, edgecolor='black', alpha=0.7)
    
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Feature Distribution')
    plt.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
    plt.legend()
    
    plt.savefig('/tmp/histogram.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to /tmp/histogram.png")


def scatter_plot():
    """
    Scatter plot for relationships between variables.
    
    Use for: Feature relationships, predictions vs actuals, embeddings
    """
    print("Creating: Scatter Plot")
    
    x = np.random.randn(100)
    y = 2 * x + 1 + np.random.randn(100) * 0.5
    
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), 'r--', label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
    
    plt.xlabel('Feature X')
    plt.ylabel('Target Y')
    plt.title('Feature vs Target Relationship')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('/tmp/scatter.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to /tmp/scatter.png")


def bar_chart():
    """
    Bar chart for categorical comparisons.
    
    Use for: Class distributions, model comparison, feature importance
    """
    print("Creating: Bar Chart")
    
    categories = ['Model A', 'Model B', 'Model C', 'Model D']
    accuracy = [0.85, 0.92, 0.78, 0.89]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, accuracy, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{acc:.2f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.ylim(0, 1.0)
    
    plt.savefig('/tmp/bar_chart.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to /tmp/bar_chart.png")


def confusion_matrix_plot():
    """
    Confusion matrix for classification results.
    
    Use for: Classification evaluation, error analysis
    """
    print("Creating: Confusion Matrix")
    
    # Simulated confusion matrix
    cm = np.array([
        [50, 5, 3],
        [2, 45, 8],
        [1, 4, 42]
    ])
    classes = ['Class A', 'Class B', 'Class C']
    
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plt.savefig('/tmp/confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to /tmp/confusion_matrix.png")


def subplots_example():
    """
    Multiple plots in one figure.
    
    Use for: Comparing multiple views, dashboard-style visualization
    """
    print("Creating: Subplots")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot 1: Line
    x = np.linspace(0, 10, 100)
    axes[0, 0].plot(x, np.sin(x))
    axes[0, 0].set_title('Line Plot')
    
    # Plot 2: Histogram
    axes[0, 1].hist(np.random.randn(1000), bins=30)
    axes[0, 1].set_title('Histogram')
    
    # Plot 3: Scatter
    axes[1, 0].scatter(np.random.randn(50), np.random.randn(50))
    axes[1, 0].set_title('Scatter')
    
    # Plot 4: Bar
    axes[1, 1].bar(['A', 'B', 'C'], [3, 7, 5])
    axes[1, 1].set_title('Bar Chart')
    
    plt.tight_layout()
    plt.savefig('/tmp/subplots.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to /tmp/subplots.png")


def roc_curve():
    """
    ROC curve for binary classification.
    
    Use for: Model evaluation, threshold selection
    """
    print("Creating: ROC Curve")
    
    # Simulated ROC data
    fpr = np.array([0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    tpr = np.array([0, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 1.0])
    
    # Calculate AUC (simplified)
    auc = np.trapz(tpr, fpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {auc:.2f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('/tmp/roc_curve.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to /tmp/roc_curve.png")


def feature_importance_plot():
    """
    Feature importance visualization.
    
    Use for: Model interpretation, feature selection
    """
    print("Creating: Feature Importance")
    
    features = ['age', 'income', 'credit_score', 'debt_ratio', 'employment_years']
    importance = [0.25, 0.35, 0.20, 0.12, 0.08]
    
    # Sort by importance
    idx = np.argsort(importance)
    
    plt.figure(figsize=(8, 5))
    plt.barh([features[i] for i in idx], [importance[i] for i in idx])
    
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    
    plt.savefig('/tmp/feature_importance.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to /tmp/feature_importance.png")


# =============================================================================
# QUICK REFERENCE
# =============================================================================

def print_quick_reference():
    """Print quick matplotlib reference."""
    reference = """
    ╔═══════════════════════════════════════════════════════════╗
    ║              MATPLOTLIB QUICK REFERENCE                   ║
    ╠═══════════════════════════════════════════════════════════╣
    ║ BASIC SETUP:                                              ║
    ║   import matplotlib.pyplot as plt                         ║
    ║   fig, ax = plt.subplots(figsize=(8, 5))                  ║
    ║                                                           ║
    ║ COMMON PLOTS:                                             ║
    ║   plt.plot(x, y)           # Line plot                    ║
    ║   plt.scatter(x, y)        # Scatter plot                 ║
    ║   plt.bar(x, heights)      # Bar chart                    ║
    ║   plt.hist(data, bins=30)  # Histogram                    ║
    ║   plt.imshow(matrix)       # Heatmap/Image                ║
    ║                                                           ║
    ║ STYLING:                                                  ║
    ║   plt.plot(x, y, 'r--', label='data')  # Red dashed       ║
    ║   plt.xlabel('X'), plt.ylabel('Y')     # Labels           ║
    ║   plt.title('Title')                   # Title            ║
    ║   plt.legend()                         # Show legend      ║
    ║   plt.grid(True, alpha=0.3)            # Add grid         ║
    ║                                                           ║
    ║ LINE STYLES: '-' solid, '--' dashed, ':' dotted          ║
    ║ MARKERS: 'o' circle, 's' square, '^' triangle            ║
    ║ COLORS: 'b' blue, 'r' red, 'g' green, 'k' black          ║
    ║                                                           ║
    ║ SUBPLOTS:                                                 ║
    ║   fig, axes = plt.subplots(2, 2)                          ║
    ║   axes[0, 0].plot(x, y)                                   ║
    ║                                                           ║
    ║ SAVE/SHOW:                                                ║
    ║   plt.savefig('plot.png', dpi=100, bbox_inches='tight')   ║
    ║   plt.show()               # Display (interactive)        ║
    ║   plt.close()              # Clear figure memory          ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(reference)


# =============================================================================
# INTERVIEW TIPS
# =============================================================================

def print_interview_tips():
    """Print interview visualization tips."""
    tips = """
    INTERVIEW VISUALIZATION TIPS:
    ─────────────────────────────────────────────────────────────
    
    1. ALWAYS LABEL YOUR AXES
       - plt.xlabel(), plt.ylabel(), plt.title()
       - Shows professionalism and communication skills
    
    2. START SIMPLE
       - plt.plot() or plt.scatter() first
       - Add styling only if time permits
    
    3. VERBALLY DESCRIBE WHAT YOU'D PLOT
       - "I would visualize the loss curve to check for convergence"
       - "A scatter plot would show the relationship between X and Y"
    
    4. KNOW WHEN TO PLOT WHAT:
       - Distributions → Histogram
       - Relationships → Scatter
       - Over time → Line plot
       - Comparisons → Bar chart
       - Classification → Confusion matrix, ROC
    
    5. QUICK EDA SEQUENCE:
       - df['column'].hist()           # Distribution
       - df.plot.scatter(x='a', y='b') # Relationship
       - df['cat'].value_counts().plot.bar()  # Categories
    
    6. IF SHORT ON TIME:
       - Describe the plot verbally
       - "I would plot a histogram here to check the distribution"
       - Focus on the insight, not the visualization
    """
    print(tips)


if __name__ == "__main__":
    print("=" * 60)
    print("MATPLOTLIB QUICK PATTERNS")
    print("=" * 60)
    
    print_quick_reference()
    print_interview_tips()
    
    print("\nGenerating example plots...")
    print("-" * 60)
    
    basic_line_plot()
    histogram_distribution()
    scatter_plot()
    bar_chart()
    confusion_matrix_plot()
    subplots_example()
    roc_curve()
    feature_importance_plot()
    
    print("-" * 60)
    print("All plots saved to /tmp/")
    print("=" * 60)
