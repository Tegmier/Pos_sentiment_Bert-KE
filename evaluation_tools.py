from sklearn.metrics import precision_score, recall_score, f1_score


def metrics_cal(total_z, total_z_pred):
    flattened_preds = [item for sentence in total_z for item in sentence]
    flattened_labels = [item for sentence in total_z_pred for item in sentence]
    recall = recall_score(flattened_labels, flattened_preds, average='micro')
    precision = precision_score(flattened_labels, flattened_preds, average='micro')
    f1 = f1_score(flattened_labels, flattened_preds, average='micro')

    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'F1 Score: {f1:.4f}')