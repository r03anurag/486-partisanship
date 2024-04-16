import matplotlib.pyplot as plt
import numpy as np


def generate_confusion_matrix(tp, tn, fp, fn):

    # Construct the confusion matrix manually
    confusion_matrix = np.array([[tp, fp], [fn, tn]])

    # Plot the confusion matrix
    plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')

    # Add labels to the matrix
    plt.title('Confusion Matrix', pad=20)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Positive', 'Negative'])
    plt.yticks(tick_marks, ['Positive', 'Negative'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Add text annotations for values in the matrix
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center', color='black')

    # Adjust subplot spacing to add a gap between title and graph
    plt.tight_layout(pad=3)

    # Save the plot to a file with DPI 300
    plt.savefig('confusionMatrix.png', dpi=300)

    # Show the plot
    plt.show()


def main():
    generate_confusion_matrix(601, 629, 56, 83)


if __name__ == "__main__":
    main()
