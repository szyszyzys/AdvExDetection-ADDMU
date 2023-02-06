import os
import glob
import pdb
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
"""
Loader class called from main.py
 - loads relevant csv file (attack,model,dataset)
 - split into val, test csv and save it to cache
 - random sample necessary file and split into sampled-test-seed{seed} , and save it to cache
"""

class AttackLoader():
  def __init__(self, args, logger, data_type="standard"):
    self.cache_dir = "./attack-log/cache"
    self.logger = logger
    self.scenario = args.scenario
    self.max_adv_num_dict = {'imdb':1000, 'ag-news':2000, 'sst2':1000, 'snli': 2000, 'paws': 2000, 'mnli': 1000, 'hate': 1000, 'yelp': 1000}
    self.max_adv_num = self.max_adv_num_dict[args.dataset]
    self.args = args

    if data_type == "standard":
      self.root = "./attack-log/"
      self.data_dir = os.path.join(self.root, args.dataset)
      self.model_dir = os.path.join(self.data_dir, args.model_type)
      self.csv_dir = os.path.join(self.model_dir, args.attack_type)
      print('csv_dir', self.csv_dir)
      csv_files = glob.glob(os.path.join(self.csv_dir,"*.csv"))
      print('csv_files', csv_files)
      assert len(csv_files) == 1, f"{len(csv_files)} exists in {self.csv_dir}"
      self.csv_file = csv_files[0]
      self.seed = logger.seed
      self.val_ratio = 0.3 if args.dataset!="sst2" else 0.0
      self.cache_dir = os.path.join(self.cache_dir, args.dataset, args.model_type, args.attack_type)
      if not os.path.isdir(self.cache_dir):
        os.makedirs(self.cache_dir)
      self.split_csv_to_testval()

  def split_csv_to_testval(self):
    self.logger.log.info(f"Splitting {self.csv_file}")
    np.random.seed(self.seed)
    df = pd.read_csv(self.csv_file)
    num_samples = len(df)
    indices = np.random.permutation(range(num_samples))
    split_point = int(num_samples*self.val_ratio)

    valset = df.iloc[indices[:split_point]]
    if self.args.dataset == "sst2":
      # For sst2, test / validation split is already given
      val_path = os.path.join(self.data_dir,"val")
      print(val_path)
      csv_files = glob.glob(os.path.join(val_path, f"{self.args.model_type}*{self.args.attack_type}.csv"))
      assert len(csv_files) == 1, f"{len(csv_files)} exists in validation path {csv_files}"
      valset = pd.read_csv(csv_files[0])
      val_path = os.path.join(self.cache_dir, "val.csv")
      valset.to_csv(val_path)
    elif self.val_ratio == 0 :
      print(f"Skipping validation set")
    else:
      val_path = os.path.join(self.cache_dir, "val.csv")
      valset.to_csv(val_path)
    testset = df.iloc[indices[split_point:]]
    testpath = os.path.join(self.cache_dir, "test.csv")
    testset.to_csv(testpath)
    self.logger.log.info("test/val split saved in cache")

  def get_attack_from_csv(self, dtype='test', batch_size=64,
                          model_wrapper=None):
    def clean_text(t):
      t = t.replace("[[[[Premise]]]]: ", "")
      t = t.replace("[[[[Hypothesis]]]]: ", "")
      t = t.replace("[", "")
      t = t.replace("]", "")
      t = t.replace("<SPLIT>", " <SPLIT> ")
      return t

    df = pd.read_csv(os.path.join(self.cache_dir, f"{dtype}.csv"))
    df.loc[df.result_type == 'Failed', 'result_type'] = 0
    df.loc[df.result_type == 'Successful', 'result_type'] = 1
    df.loc[df.result_type == 'Skipped', 'result_type'] = -1

    assert self.scenario in ['s1', 's2'], "Check split type"
    if self.scenario == 's1':
      num_samples = df.shape[0]
      num_adv = (df.result_type == 1).sum()
      """
      Procedure: 
       1. randomly sample N samples from testset and attain adversarial samples.
       2. from the remaining testset randomly sample clean samples (around N)  
      
      How to Choose N (target_sample): 
      number of random samples to take is determined by (# of desired adv. samples / success rate of adv. attack) 
      N = (# of desired adv. samples / success rate of adv. attack) = (# of desired adv. samples / # of adv. samples) / (# of total samples) 
      split_ratio : N/ # of total samples = (# of desired adv. samples / # of adv. samples) = max_adv_num / num_adv 
      max_adv_num is dataset dependent and decremented by 10 until attaining this is possible without causing clean/adv class imbalance
      """
      max_adv_num = self.max_adv_num
      adv_sr = num_adv / num_samples
      task_acc = (df.result_type!=-1).sum() / num_samples
      target_samples = max_adv_num * (1/adv_sr) * (1/task_acc) # Expected number of required sample to attain max_adv_num adversarial samples
      split_ratio = target_samples/num_samples      # ratio to attain max_adv_num number of adv. samples
      while split_ratio >= 0.4 and max_adv_num > 0: # Make sure clean:adv ratio can be maintained
        split_ratio = max_adv_num / num_adv
        max_adv_num -= 10
      if split_ratio >= 0.4 or max_adv_num < 0:
        raise Exception(
          f"Dataset is too small to sample enough adverserial samples. Total: {num_samples}, Adv.: {num_adv}")

      np.random.seed(self.seed)
      rand_idx = np.arange(num_samples)
      np.random.shuffle(rand_idx)

      # Subset 1
      split_point = int(num_samples * split_ratio)
      split_idx = rand_idx[:split_point]
      split = df.iloc[split_idx].copy()
      if self.args.include_fae :
        adv = split.loc[split.result_type!=-1]
        adv.loc[:, 'result_type'] = 1
      else:
        # Only take sucessful adv. attempts
        adv = split.loc[split.result_type == 1]
      adv = adv.rename(columns={"perturbed_text": "text"})
      num_adv_samples = adv.shape[0]
      # Subset 2
      other_split_idx = rand_idx[split_point:split_point + num_adv_samples] #Find equal number of clean samples
      other_split = df.iloc[other_split_idx].copy()
      clean = other_split  # Use correct and incorrect samples
      clean.loc[:, 'result_type'] = 0
      clean = clean.rename(columns={"original_text": "text"})
      testset = pd.concat([adv, clean], axis=0)

    elif self.scenario == 's2':
      num_samples = df.shape[0]
      num_adv = (df.result_type == 1).sum()
      max_adv_num = self.max_adv_num
      adv_sr = num_adv / num_samples
      task_acc = (df.result_type != -1).sum() / num_samples

      np.random.seed(self.seed)
      rand_idx = np.arange(num_samples)
      np.random.shuffle(rand_idx)

      split_point = min(num_samples, int(max_adv_num / (adv_sr*task_acc)))
      split_idx = rand_idx[:split_point]
      split = df.iloc[split_idx]

      adv_samples = split.copy()
      adv_samples = adv_samples.loc[adv_samples.result_type == 1]
      adv_samples = adv_samples.rename(columns={'perturbed_text': 'text'})
      clean_samples = split.copy()
      clean_samples['result_type'] = 0
      clean_samples = clean_samples.rename(columns={'original_text': 'text'})
      testset = pd.concat([clean_samples, adv_samples], axis=0)

    # if 'nli' in self.csv_file:  # For NLI dataset, only get the hypothesis, which is attacked
    #   df['original_text'] = df['original_text'].apply(lambda x: x.split('>>>>')[1])
    #   testset['text'] = testset['text'].apply(lambda x: x.split('>>>>')[1])

    df['original_text'] = df['original_text'].apply(clean_text)
    df['perturbed_text'] = df['perturbed_text'].apply(clean_text)
    testset['text'] = testset['text'].apply(clean_text)
    testset.to_csv(os.path.join(self.cache_dir, f'sampled-{dtype}-{self.seed}.csv'))

    if model_wrapper:
      self.__sanity_check(df, model_wrapper, batch_size)

    return testset, df

  def __sanity_check(self, df, model_wrapper, batch_size):
    dataset = df[['perturbed_text', 'original_text']]
    gt = df['ground_truth_output'].tolist()
    # Compute Acc. on dataset
    num_samples = len(dataset)
    num_batches = int((num_samples // batch_size) + 1)
    target_adv_indices = []

    correct = 0
    adv_correct = 0
    total = 0
    adv_pred = []
    clean_pred = []

    with torch.no_grad():
      for i in tqdm(range(num_batches)):
        lower = i * batch_size
        upper = min((i + 1) * batch_size, num_samples)
        adv_examples = dataset['perturbed_text'][lower:upper].tolist()
        clean_examples = dataset['original_text'][lower:upper].tolist()
        labels = gt[lower:upper]

        y = torch.LongTensor(labels).to(model_wrapper.model.device)
        output = model_wrapper.inference(adv_examples)
        preds = torch.max(output.logits, dim=1).indices
        adv_pred.append(preds.cpu().numpy())
        adv_correct += y.eq(preds).sum().item()
        adv_error_idx = preds.ne(y)

        output = model_wrapper.inference(clean_examples)
        preds = torch.max(output.logits, dim=1).indices
        clean_pred.append(preds.cpu().numpy())
        correct += y.eq(preds).sum().item()
        clean_correct_idx = preds.eq(y)
        total += preds.size(0)

        target_adv_idx = torch.logical_and(adv_error_idx, clean_correct_idx)
        target_adv_indices.append(target_adv_idx.cpu().numpy())

    """
    Sanity Check : prediction results should be equivalent to FGWS predictions 
    """
    self.logger.log.info("Sanity Check for testset")
    target_adv_indices = np.concatenate(target_adv_indices, axis=0)
    adv_pred = np.concatenate(adv_pred, axis=0)
    clean_pred = np.concatenate(clean_pred, axis=0)
    fgws_adv_pred = df['perturbed_output'].values
    fgws_adv_pred[np.isnan(fgws_adv_pred)] = adv_pred[np.isnan(fgws_adv_pred)]
    adv_pred_diff = (np.not_equal(adv_pred, fgws_adv_pred)).sum()
    clean_pred_diff = (np.not_equal(clean_pred, df['original_output'].values)).sum()
    incorrect_indices = np.not_equal(df['original_output'].values, df['ground_truth_output'].values)
    self.logger.log.info(f"# of adv. predictions different : {adv_pred_diff}")
    self.logger.log.info(f"# of clean predictions different : {clean_pred_diff}")
    self.logger.log.info(f"Clean Accuracy {correct / total}")
    self.logger.log.info(f"Robust Accuracy {adv_correct / total}")
    self.logger.log.info(f"Adv. Success Rate {target_adv_indices.sum() / total}")
