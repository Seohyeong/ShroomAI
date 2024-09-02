import matplotlib.pyplot as plt

def get_result(history):
    result = {}
    result['train_acc'] = history.history['accuracy']
    result['val_acc'] = history.history['val_accuracy']
    result['train_loss'] = history.history['loss']
    result['val_loss'] = history.history['val_loss']
    return result


# TODO: break up this function
def get_top_n_acc(validation, model):
    count = 0
    top1, top2, top3, top4, top5 = 0, 0, 0, 0, 0

    gt_labels = []
    predictions = []
    for imgs, labels in tqdm(validation.take(len(validation)), total=len(validation)):
        pred_logits = model.predict(imgs, verbose=0)
        for idx, pred_logit in enumerate(pred_logits):
            preds = pred_logit.argsort()[-5:][::-1] # shape: bs, X
            label = int(labels[idx]) # shape: bs
            if label in preds:
                top5 += 1
            if label in preds[:4]:
                top4 += 1
            if label in preds[:3]:
                top3 += 1
            if label in preds[:2]:
                top2 += 1
            if label == preds[0]:
                top1 += 1
            count += 1

            gt_labels.append(label)
            predictions.append(preds)

    print('\ntotal # of samples: {}'.format(count))
    return {
        'top_1_acc': top1/count,
        'top_2_acc': top2/count,
        'top_3_acc': top3/count,
        'top_4_acc': top4/count,
        'top_5_acc': top5/count
    }, gt_labels, predictions


def plot(history, desc):
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    lr = history.history['lr']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('acc')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Accuracy ({})'.format(desc))

    plt.subplot(2, 1, 2)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot(lr, label='Learning Rate')
    plt.legend(loc='upper right')
    plt.ylabel('loss')
    plt.ylim([0,1.0])
    plt.title('Loss/LR ({})'.format(desc))
    plt.xlabel('epoch')
    plt.show()