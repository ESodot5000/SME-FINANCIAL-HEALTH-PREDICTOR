import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)
from preprocessing import load_data, prepare_features
from config import MODEL_PATH, TARGET_CLASSES


def load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    Print classification report for Low, Medium, High.
    """
    # TODO: y_pred = model.predict(X_test)
    y_pred = model.predict(X_test)
    # TODO: print(classification_report(y_test, y_pred, target_names=TARGET_CLASSES))
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, target_names=TARGET_CLASSES))
    # TODO: print F1 macro score
    print("F1 Macro Score")
    F1_Macro = f1_score(y_test, y_pred, average = 'macro')
    print(f'Overall F1 Macro Score:{round(F1_Macro,4)}')
    # TODO: return y_pred
    return y_pred


def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix for all three financial health classes."""
    # TODO: sns.heatmap on confusion_matrix(y_test, y_pred)
    # Use TARGET_CLASSES as labels
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=TARGET_CLASSES,
        yticklabels=TARGET_CLASSES
    )
    plt.title('Confusion Matrix -SME Financial Health Predictor')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    


if __name__ == '__main__':
    df = load_data()
    _, X_test, _, y_test, encoders = prepare_features(df)
    model = load_model()
    y_pred = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plt.show()
