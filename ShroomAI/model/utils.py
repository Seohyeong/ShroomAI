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




############################

def imshow(inp, title=None):
    """
    Display image for Tensor.
    
    Usage:

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

    

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def visualize_model_predictions(model,img_path):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        imshow(img.cpu().data[0])
        
        model.train(mode=was_training)