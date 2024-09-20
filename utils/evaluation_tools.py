from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

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

def z_task_unmatched_word_cal(total_z, total_z_pred, input_ids, tokenizer, logger):
    flattened_preds = [item for sentence in total_z_pred for item in sentence]
    flattened_labels = [item for sentence in total_z for item in sentence]
    flattened_inputids = [item for sentence in input_ids for item in sentence]
    # print(len(flattened_preds), len(flattened_labels), len(flattened_inputids))
    wrong_list_inputids = []
    for i in range(len(flattened_labels)):
        if flattened_labels[i] != 0 and flattened_preds[i] == 0:
            wrong_list_inputids.append(flattened_inputids[i])
    wrong_list_word=tokenizer.convert_ids_to_tokens(wrong_list_inputids)
    word_count = Counter(wrong_list_word)
    sorted_word_count = word_count.most_common(30)
    logger.info(f"word cloud summary")
    logger.info(f"Total times of wrongly predicted words is {len(wrong_list_inputids)}")
    logger.info(f"Total number of wrongly predicted words is {len(word_count)}")
    with open('plot/wordcloud_output.txt', 'w') as file:
        for item in sorted_word_count:
            file.write(f"{item}\n") 
    wordcloud = WordCloud(stopwords=set(), width=800, height=400, background_color='white').generate(" ".join(wrong_list_word))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('plot/wordcloud.png')
    


def metrics_cal(total_z, total_z_pred, type, logger):
    # Score Calculation
    flattened_preds = [item for sentence in total_z_pred for item in sentence]
    flattened_labels = [item for sentence in total_z for item in sentence]
    cm = confusion_matrix(flattened_labels, flattened_preds)
    # draw confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix Heatmap for ' + type)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('plot/Confusion Matrix Heatmap for '+ type + '.png')
    # plt.show()
    if type == 'y task':
        labels = [0, 1]
        average = "binary"
        recall = recall_score(flattened_labels, flattened_preds, average=average, labels=labels)
        precision = precision_score(flattened_labels, flattened_preds, average=average, labels=labels)
        f1 = f1_score(flattened_labels, flattened_preds, average=average, labels=labels)
        # recall_n = recall_score(flattened_labels, flattened_preds, average=None, labels=labels)
        # precision_n = precision_score(flattened_labels, flattened_preds, average=None, labels=labels)
        # f1_n = f1_score(flattened_labels, flattened_preds, average=None, labels=labels)
        logger.info("----------------FOR Y TASK--------------------------------")  
        logger.info(f"AVERAGE = {average}: ")
        logger.info(f'Precision for {type}: {precision:.3f}')
        logger.info(f'Recall for {type}: {recall:.3f}')
        logger.info(f'F1 Score for {type}: {f1:.3f}')
        logger.info("AVERAGE = NONE: ")
        # logger.info(f'Recall for {type}: {recall_n}')
        # logger.info(f'Precision for {type}: {precision_n}')
        # logger.info(f'F1 Score for {type}: {f1_n}')
        logger.info(f'Plotting Confusion Matrix Heatmap for {type} has finished!')
        logger.info("---------------------------------------------------------")
        logger.info("\n\n")
    else:
        labels = [0,1,2,3,4]
        # average = "weighted"
        # recall_w = recall_score(flattened_labels, flattened_preds, average=average, labels=labels)
        # precision_w = precision_score(flattened_labels, flattened_preds, average=average, labels=labels)
        # f1_w = f1_score(flattened_labels, flattened_preds, average=average, labels=labels)

        recall_n = recall_score(flattened_labels, flattened_preds, average=None, labels=labels)
        precision_n = precision_score(flattened_labels, flattened_preds, average=None, labels=labels)
        f1_n = f1_score(flattened_labels, flattened_preds, average=None, labels=labels)

        recall_ma = recall_score(flattened_labels, flattened_preds, average='macro', labels=labels)
        precision_ma = precision_score(flattened_labels, flattened_preds, average='macro', labels=labels)
        f1_ma = f1_score(flattened_labels, flattened_preds, average='macro', labels=labels)
        logger.info("----------------FOR Z TASK--------------------------------")
        logger.info("AVERAGE = weighted: ")
        # logger.info(f'Recall for {type}: {recall_w:.8f}')
        # logger.info(f'Precision for {type}: {precision_w:.8f}')
        # logger.info(f'F1 Score for {type}: {f1_w:.8f}')
        logger.info("AVERAGE = macro: ")
        logger.info(f'Precision for {type}: {precision_ma:.3f}')
        logger.info(f'Recall for {type}: {recall_ma:.3f}')   
        logger.info(f'F1 Score for {type}: {f1_ma:.3f}')
        logger.info("AVERAGE = NONE: ")
        logger.info(f'Precision for {type}: {precision_n}')
        logger.info(f'Recall for {type}: {recall_n}')
        logger.info(f'F1 Score for {type}: {f1_n}')
        logger.info(f'Plotting Confusion Matrix Heatmap for {type} has finished!')
        logger.info("---------------------------------------------------------")
        logger.info("\n\n")

