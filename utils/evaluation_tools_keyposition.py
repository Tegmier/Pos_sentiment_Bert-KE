from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
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
    
    logger.info(f'Keyphrase accuracy calculation in keyposition for {type} is {acc_count/total_count}')

def metrics_cal_keyposition(total_z, total_z_pred, type, logger):
    # Score Calculation
    flattened_preds = [item for sentence in total_z_pred for item in sentence]
    flattened_labels = [item for sentence in total_z for item in sentence]
    recall = recall_score(flattened_labels, flattened_preds, average='micro')
    precision = precision_score(flattened_labels, flattened_preds, average='micro')
    f1 = f1_score(flattened_labels, flattened_preds, average='micro')
    cm = confusion_matrix(flattened_labels, flattened_preds)
    if type == 'y task':
        labels = [0, 1]
    else:
        labels = [0,1,2,3,4]
    recall_n = recall_score(flattened_labels, flattened_preds, average=None, labels=labels)
    precision_n = precision_score(flattened_labels, flattened_preds, average=None, labels=labels)
    f1_n = f1_score(flattened_labels, flattened_preds, average=None, labels=labels)

    # draw confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix Heatmap for ' + type + ' in keyposition')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('plot/Confusion Matrix Heatmap for '+ type + ' in keyposition.png')
    plt.show()
    
    logger.info("------------------------------------------------")
    logger.info("AVERAGE = MICRO: ")
    logger.info(f'Recall in keyposition for {type}: {recall:.8f}')
    logger.info(f'Precision in keyposition for {type}: {precision:.8f}')
    logger.info(f'F1 Score in keyposition for {type}: {f1:.8f}')
    logger.info("AVERAGE = NONE: ")
    logger.info(f'Recall for in keyposition {type}: {recall_n}')
    logger.info(f'Precision for in keyposition {type}: {precision_n}')
    logger.info(f'F1 Score for in keyposition {type}: {f1_n}')
    logger.info(f'Plotting Confusion Matrix Heatmap in keyposition for {type} has finished!')
    logger.info("------------------------------------------------")