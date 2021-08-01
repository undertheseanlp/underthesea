import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from transformers import logging
from preprocessing import *
from model import BertForSequenceClassifier

fnames = dict(
    restaurant=dict(
        train="/Users/taidnguyen/Desktop/Sentence-level-Restaurant/Train.txt",
        test="/Users/taidnguyen/Desktop/Sentence-level-Restaurant/Test.txt",
        dev="/Users/taidnguyen/Desktop/Sentence-level-Restaurant/Dev.txt"
    ),
    hotel=dict(
        train="/Users/taidnguyen/Desktop/Sentence-level-Hotel/Train_Hotel.txt",
        test="/Users/taidnguyen/Desktop/Sentence-level-Hotel/Test_Hotel.txt",
        dev="/Users/taidnguyen/Desktop/Sentence-level-Hotel/Dev_Hotel.txt"
    )
)


def main():
    entity = "hotel"

    # Args
    parser = argparse.ArgumentParser(description="Process some steps")
    parser.add_argument("--dev_path", type=str, default=fnames[entity]["dev"])
    parser.add_argument("--train_path", type=str, default=fnames[entity]["train"])
    parser.add_argument("--test_path", type=str, default=fnames[entity]["test"])
    parser.add_argument("--pretrained_bert_name", type=str, default="vinai/phobert-base")
    parser.add_argument("--max_sequence_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=69)
    args = parser.parse_args()

    seed_everything(args.seed)
    tokenizer = Tokenizer4Bert(args.max_sequence_len, args.pretrained_bert_name)

    # Train
    train = ABSADataset(args.dev_path, tokenizer)
    train_X, train_y = train.sentence_input_ids, train.aspect_input_ids
    train_attention_mask = train.attention_masks
    train_dataset = TensorDataset(torch.tensor(train_X, dtype=torch.long),
                                  torch.tensor(train_attention_mask, dtype=torch.long),
                                  torch.tensor(train_y, dtype=torch.long))

    # Test
    test = ABSADataset(args.test_path, tokenizer)
    test_X, test_y = test.sentence_input_ids, test.aspect_input_ids
    test_attention_mask = test.attention_masks
    test_dataset = TensorDataset(torch.tensor(test_X, dtype=torch.long),
                                 torch.tensor(test_attention_mask, dtype=torch.long),
                                 torch.tensor(test_y, dtype=torch.long))

    # Model
    logging.set_verbosity_debug()

    config = BertConfig.from_pretrained(
        'bert-base-uncased',
        architectures=['BertForSequenceClassification'],
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_dropout_prob=0.1,
        num_labels=len(train.unique_asp_lists)
    )
    model = BertForSequenceClassifier(config=config)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Creating optimizer and lr schedulers
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * args.epochs)

    # Train
    seed_everything(args.seed)

    def evaluate(test_dataloader):

        model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        for batch in test_dataloader:
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            with torch.no_grad():
                outputs = model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(test_dataloader)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        return loss_val_avg, predictions, true_vals

    for epoch in tqdm(range(1, args.epochs + 1)):
        model.train()
        loss_train_total = 0
        progress_bar = tqdm(train_dataloader, desc='Epoch {:1d}'.format(args.epochs), leave=False, disable=False)
        for batch in progress_bar:
            model.zero_grad()
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2]
            }

            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
            torch.save(model.state_dict(), f'model_tries/BERT_epoch_{epoch}.model')

            tqdm.write(f'\nEpoch {epoch}')
            loss_train_avg = loss_train_total / len(train_dataloader)
            tqdm.write(f'Training loss: {loss_train_avg}')

            val_loss, predictions, true_vals = evaluate(test_dataloader)
            # val_f1 = f1_score_func(predictions, true_vals)
            tqdm.write(f'Validation loss: {val_loss}')
            # tqdm.write(f'F1 Score (Weighted): {val_f1}')


if __name__ == '__main__':
    main()
