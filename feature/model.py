from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn, no_grad, save
from tqdm import tqdm
import torch
import time
import datetime
import numpy as np
from sklearn.metrics import f1_score

def import_model_tokenizer(num_labels, name='bert'):
    if name == 'bert':
        tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
        model = BertForSequenceClassification.from_pretrained("monologg/kobert",
                                                              num_labels = num_labels)
    return model, tokenizer

def train_model(model, train_loader, val_loader, lr, epochs, device, model_dir):
    optimizer = AdamW(model.parameters(),
                  lr = lr, 
                  eps = 1e-8 
                )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)
    total_t0 = time.time()
    training_stats = []
    best_acc = 0
    best_loss = 100
    for epoch in range(epochs):
        print("")
        print('='*30 + ' Epoch {:} / {:} '.format(epoch + 1, epochs)+ '='*30)
        print("  < Training >")
        t0 = time.time()
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(tqdm(train_loader)):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['label'].to(device)

            model.zero_grad()        
            loss, logits = model(b_input_ids, 
                                 token_type_ids=None, 
                                 attention_mask=b_input_mask, 
                                 labels=b_labels)[:]
            total_train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.4f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        print("")
        print("  < Validation >")
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        total_eval_f1 = 0

        for batch in tqdm(val_loader):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['label'].to(device)
            with no_grad():        
                (loss, logits) = model(b_input_ids, 
                                       token_type_ids=None, 
                                       attention_mask=b_input_mask,
                                       labels=b_labels)[:]

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)
            total_eval_f1 += f1_score(label_ids,
                                      np.argmax(logits, axis=1).flatten(),
                                      average='macro')


        avg_val_accuracy = total_eval_accuracy / len(val_loader)
        print("  Accuracy: {0:.4f}".format(avg_val_accuracy))
        avg_val_f1 = total_eval_f1 / len(val_loader)
        print("  F1-Score: {0:.4f}".format(avg_val_f1))

        avg_val_loss = total_eval_loss / len(val_loader)

        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.4f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Acc': avg_val_accuracy,
                'Valid. F1': avg_val_f1,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
        # (avg_val_accuracy > best_acc) or 
        if avg_val_loss < best_loss:
            best_acc = avg_val_accuracy
            best_loss = avg_val_loss
            # save(model.state_dict(),
            #      model_dir+'best_model-epoch-{:}-loss-{:4}.pt'.format(epoch+1,best_loss))
            save(model.state_dict(),
                 model_dir+'best_model.pt')

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss

    Parameters
    ----------
    elapsed : int
        time delta second

    Returns
    ----------
    str
        time delta hh:mm:ss form
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(preds, labels):
        '''
        convert model result from score to label and return accuracy

        Parameters
        ----------
        preds : array
            model prediction result (score)
        
        labels : array
            true label

        Returns
        ----------
        float 
            classification accuracy

        '''
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)