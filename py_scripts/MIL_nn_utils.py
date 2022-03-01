import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

class MILRestsDataset(Dataset):
  def __init__(self, revid_2_sentcid_2_data_raw, tokenizer, max_seq_len_rev, max_seq_len_sentc):
    ids_rev = list()
    id_sentc_lists = list()
    tokens_sentc_lists = list()
    senti_sentc_lists = list()
    ratings_rev = list()

    for id_rev,sentences in revid_2_sentcid_2_data_raw.items():
      id_sentc_list = list()
      tokens_sentc_list = list()
      senti_sentc_list = list()
      ratings_temp = list()
      for i in range(1,max_seq_len_rev+1):
        id_sentc = str(i)
        if id_sentc in sentences.keys():
          sentence = sentences[id_sentc]
          id_sentc_list.append(id_sentc)        
          # sentence.keys() = ['sentiments', 'tokens', 'labeled with C', 'id review', 'rating review']
          tokens_sentc_list.append(sentence['tokens'])
          ratings_temp.append(sentence['rating review'])
          sentiment_scores = list()
          if sentence['sentiments']:
            for sentiment in sentence['sentiments']:    
              sentiment_scores.append(sentiment['sentiment score'])
          else:
            sentiment_scores.append(0)
          senti_sentc_list.append(np.mean(sentiment_scores)) 
        else:
          # padding sentence is inserted
          id_sentc =  '-'+id_sentc # padding sentence is indicated by a '-' in id_sentc
          id_sentc_list.append(id_sentc)        
          tokens_sentc_list.append([''])
          senti_sentc_list.append(np.mean(0)) 
      ids_rev.append(id_rev)
      id_sentc_lists.append(id_sentc_list)
      tokens_sentc_lists.append(tokens_sentc_list)
      senti_sentc_lists.append(senti_sentc_list)
      assert ratings_temp[0] == np.mean(ratings_temp)
      ratings_rev.append(ratings_temp[0])
      
    self.ids_rev = ids_rev
    self.id_sentc_lists = id_sentc_lists
    self.tokens_sentc_lists = tokens_sentc_lists
    self.senti_sentc_lists = senti_sentc_lists
    self.ratings_rev = ratings_rev
    self.tokenizer = tokenizer
    self.max_seq_len_sentc = max_seq_len_sentc
    self.max_seq_len_rev = max_seq_len_rev

  def __len__(self):
    return len(self.ratings_rev)

  def __getitem__(self, idx):
    id_rev = self.ids_rev[idx]
    rating_rev = torch.tensor(self.ratings_rev[idx],dtype=torch.long)
    # revs are already padded! sentences get padded in function 'tokenize'
    id_sentc_list = self.id_sentc_lists[idx]
    tokens_sentc_list = self.tokens_sentc_lists[idx]
    senti_sentc_list = [torch.tensor(s,dtype=torch.long) for s in self.senti_sentc_lists[idx]]
    tokenids_sentc_list = list()
    attmasks_sentc_list = list()
    for tokens in tokens_sentc_list:
      bert_encoded = tokenizer(tokens,#tokens,
                            is_split_into_words=True,
                            padding='max_length',
                            truncation=True,
                            return_token_type_ids=False,
                            return_attention_mask=True,
                            return_tensors="pt",max_length= self.max_seq_len_sentc)
      tokenids_sentc_list.append(bert_encoded['input_ids'].flatten())
      attmasks_sentc_list.append(bert_encoded['attention_mask'].flatten())

    return {
        'id_rev':id_rev,
        'rating_rev':rating_rev,
        'id_sentc_list':id_sentc_list,
        'senti_sentc_tensor':torch.stack(senti_sentc_list),
        'tokenids_sentc_tensor':torch.stack(tokenids_sentc_list),
        'attmasks_sentc_tensor':torch.stack(attmasks_sentc_list)
    }

def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler):
  model.to(device)
  model = model.train() # sets the model to training mode
  loss_fn.to(device)

  losses = []
  correct_predictions = 0
  n_examples = 0

  for batch,d in enumerate(data_loader):
    tokenids_sentc_tensor = d["tokenids_sentc_tensor"].to(device)
    attmasks_sentc_tensor = d["attmasks_sentc_tensor"].to(device)
    targets = d["rating_rev"].to(device)

    preds = model(
        tokenids_sentc_tensor,
        attmasks_sentc_tensor
        )
    # compute predictions and loss
    loss = loss_fn(preds, targets)

    correct_predictions += torch.sum(torch.round(preds) == targets)
    n_examples += targets.shape[0]
    losses.append(loss.item())

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip exploding gradient
    optimizer.step()
    scheduler.step()
    if batch % 20 == 0: # to get more details while training
      loss_temp = loss.item()
      size = len(data_loader)
      print(f"loss: {loss_temp:>7f}  [{batch:>5d}/{size:>5d}]")

  acc = correct_predictions.double() / n_examples
  return acc, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device):
  model.to(device)
  model = model.eval()
  loss_fn.to(device)

  losses = []
  correct_predictions = 0
  n_examples = 0

  with torch.no_grad(): # no_grad() reduces memory consumption for computations, if you re sure not to use gradients in ".backward()"
    for d in data_loader:
      tokenids_sentc_tensor = d["tokenids_sentc_tensor"].to(device)
      attmasks_sentc_tensor = d["attmasks_sentc_tensor"].to(device)
      targets = d["rating_rev"].to(device)
      
      preds = model(
          tokenids_sentc_tensor,
          attmasks_sentc_tensor
          )
      # compute predictions and loss
      loss = loss_fn(preds, targets)

      correct_predictions += torch.sum(torch.round(preds) == targets)
      n_examples += targets.shape[0]
      losses.append(loss.item())
  acc = correct_predictions.double() / n_examples
  return acc, np.mean(losses)

def get_predictions(model, data_loader, device):
  model = model.eval()
  
  predictions = []
  predictions_exact = []
  true_values = []

  with torch.no_grad():
    for d in data_loader:

      tokenids_sentc_tensor = d["tokenids_sentc_tensor"].to(device)
      attmasks_sentc_tensor = d["attmasks_sentc_tensor"].to(device)
      targets = d["rating_rev"].to(device)

      preds_exact = model(
          tokenids_sentc_tensor,
          attmasks_sentc_tensor
          )

      preds = torch.round(preds_exact)

      predictions.extend(preds)
      predictions_exact.extend(preds_exact)
      true_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  predictions_exact = torch.stack(predictions_exact).cpu()
  true_values = torch.stack(true_values).cpu()
  return predictions, predictions_exact, true_values

def remove_padded_sentc(padded,id_sentc_list):

  without_padding = []
  id_sentc_list_transp = torch.transpose(torch.stack([torch.tensor([int(s) for s in l]) for l in id_sentc_list]),
                                        0,1)
  for i,row in enumerate(id_sentc_list_transp):
    new_line = []
    for j,a in enumerate(row):
      assert torch.abs(a)==j+1
      if int(a) < 0:
        break
      new_line.append(padded[i,j].item())
    without_padding.append(new_line)

  return without_padding

def get_predictions_senti(model_senti, data_loader,device):
  model_senti = model_senti.eval()
  
  predictions = []
  predictions_exact = []
  true_values = []

  with torch.no_grad():
    for d in data_loader:
      tokenids_sentc_tensor = d["tokenids_sentc_tensor"].to(device)
      attmasks_sentc_tensor = d["attmasks_sentc_tensor"].to(device)
      targets_padded = d["senti_sentc_tensor"].to(device)

      preds_senti_exact_padded = model_senti(
          tokenids_sentc_tensor,
          attmasks_sentc_tensor
          )

      id_sentc_list = d['id_sentc_list']
      preds_senti_exact = remove_padded_sentc(preds_senti_exact_padded,id_sentc_list)
      preds_senti = [[round(i) for i in row] for row in preds_senti_exact]
      targets = remove_padded_sentc(targets_padded,id_sentc_list)


      predictions.extend(preds_senti)
      predictions_exact.extend(preds_senti_exact)
      true_values.extend(targets)

  return predictions, predictions_exact, true_values