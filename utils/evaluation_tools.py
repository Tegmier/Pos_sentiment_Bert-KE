from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

def metrics_cal(total_z, total_z_pred):
    flattened_preds = [item for sentence in total_z for item in sentence]
    flattened_labels = [item for sentence in total_z_pred for item in sentence]
    recall = recall_score(flattened_labels, flattened_preds, average='micro')
    precision = precision_score(flattened_labels, flattened_preds, average='micro')
    f1 = f1_score(flattened_labels, flattened_preds, average='micro')

    cm = confusion_matrix(flattened_labels, flattened_preds)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.figure(figsize=(8, 6))
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    plt.savefig('plot/Confusion Matrix Heatmap.png')

    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'F1 Score: {f1:.4f}')