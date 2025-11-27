# dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SASRecDataset(Dataset):
	def __init__(self, user_train, usernum, itemnum, maxlen):
		self.user_train = user_train
		self.usernum = usernum
		self.itemnum = itemnum
		self.maxlen = maxlen
		self.user_list = list(user_train.keys())

	def __len__(self):
		return len(self.user_train)

	def __getitem__(self, idx):
		uid = self.user_list[idx]
		seq = np.zeros([self.maxlen], dtype=np.int32)
		pos = np.zeros([self.maxlen], dtype=np.int32)
		neg = np.zeros([self.maxlen], dtype=np.int32)

		user_items = self.user_train[uid]
		next_item = user_items[-1]
		construct_index = self.maxlen - 1
		items_set = set(user_items)

		# back to front
		for i in reversed(user_items[:-1]):
			seq[construct_index] = i
			pos[construct_index] = next_item
			neg[construct_index] = self._random_neq(1, self.itemnum+1, items_set)
			next_item = i
			construct_index -= 1
			if construct_index == -1:
				break

		return seq, pos, neg

	# random get a item from [l,r) which do not in s
	def _random_neq(self, l, r, s):
		item = np.random.randint(l, r)
		while item in s:
			item = np.random.randint(l, r)
		return item

def SASRec_collate_fn(batch):
    seqs, poss, negs = zip(*batch)
    return (
        torch.LongTensor(np.array(seqs)),
        torch.LongTensor(np.array(poss)),
        torch.LongTensor(np.array(negs))
    )


class BERT4RecDataset(Dataset):
	def __init__(self, user_train, usernum, itemnum, maxlen):
		self.user_train = user_train
		self.usernum = usernum
		self.itemnum = itemnum
		self.maxlen = maxlen
		self.user_list = list(user_train.keys())

	def __len__(self):
		return len(self.user_train)

	def __getitem__(self, idx):
		uid = self.user_list[idx]
		seq = np.zeros([self.maxlen], dtype=np.int32)

		user_items = self.user_train[uid]
		next_item = user_items[-1]
		construct_index = self.maxlen - 1
		items_set = set(user_items)

		# back to front
		for i in reversed(user_items[:-1]):
			seq[construct_index] = i
			next_item = i
			construct_index -= 1
			if construct_index == -1:
				break

		return seq

def BERT4Rec_collate_fn(batch):
    stacked_seqs = np.stack(batch)
    return torch.LongTensor(stacked_seqs)
