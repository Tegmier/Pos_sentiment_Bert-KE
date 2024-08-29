from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def keyphrase_acc_cal(total_z, total_z_pred, type, logger):
    # Keyphrase acc cal
    total_count, acc_count = 0, 0
    
    for z, z_pred in zip(total_z, total_z_pred):
        for batch in range(z.shape[0]):
            acc = True
            total_count+=1
            for position in range(z.shape[1]):
                if z[batch, position] != 0:
                    if z_pred[batch, position]!=z[batch, position]:
                        acc = False
                        break
            if acc is True:
                acc_count+=1
    
    logger.info(f'Keyphrase accuracy calculation for {type} is {acc_count/total_count}')

def metrics_cal(total_z, total_z_pred, type, logger):
    # Score Calculation
    flattened_preds = [item for sentence in total_z for item in sentence]
    flattened_labels = [item for sentence in total_z_pred for item in sentence]
    recall = recall_score(flattened_labels, flattened_preds, average='micro')
    precision = precision_score(flattened_labels, flattened_preds, average='micro')
    f1 = f1_score(flattened_labels, flattened_preds, average='micro')
    cm = confusion_matrix(flattened_labels, flattened_preds)

    # draw confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix Heatmap for ' + type)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('plot/Confusion Matrix Heatmap for '+ type + '.png')
    plt.show()
    
    logger.info("------------------------------------------------")
    logger.info(f'Recall for {type}: {recall:.4f}')
    logger.info(f'Precision for {type}: {precision:.4f}')
    logger.info(f'F1 Score for {type}: {f1:.4f}')
    logger.info(f'Plotting Confusion Matrix Heatmap for {type} has finished!')
    logger.info("------------------------------------------------")