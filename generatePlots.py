"""
generatePlots.py

Creates a confusion matrix based on data points.
"""
import matplotlib.pyplot as plt
import numpy as np


def generate_confusion_matrix(tp, tn, fp, fn, kind):
    """Generates and visualizes a confusion matrix.

    Args:
        tp (int): Number of true positives.
        tn (int): Number of true negatives.
        fp (int): Number of false positives.
        fn (int): Number of false negatives.
        kind (str): Whether this plot is for the Baseline or Optimal model.
    """
    # Construct the confusion matrix manually
    confusion_matrix = np.array([[tp, fp], [fn, tn]])

    # Plot the confusion matrix
    plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')

    # Add labels to the matrix
    plt.title(f"{kind} Model Confusion Matrix", pad=20)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Positive', 'Negative'])
    plt.yticks(tick_marks, ['Positive', 'Negative'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Add text annotations for values in the matrix
    for i in range(2):
        for j in range(2):
            color = 'white' if confusion_matrix[i, j] > confusion_matrix.max() / 2 else 'black'
            plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center', color=color)

    # Adjust subplot spacing to add a gap between title and graph
    plt.tight_layout(pad=3)

    # Save the plot to a file with DPI 300
    plt.savefig(f"confusionMatrix{kind}.png", dpi=300)

    # Show the plot
    plt.show()


def main():
    """
    Main function that generates the confusion matrices
    for the baseline and optimal models.
    """
    # Confusion matrix for baseline model
    generate_confusion_matrix(590, 560, 125, 94, "Baseline")

    # Confusion matrix for optimal model
    generate_confusion_matrix(620, 610, 75, 64, "Optimal")


if __name__ == "__main__":
    main()
