# SASRec.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from dataloader import SASRecDataset, SASRec_collate_fn
from utils import data_partition, evaluate, plot_training_curves

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)
    
    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(
                self.relu(
                    self.dropout1(
                        self.conv1(inputs.transpose(-1, -2))
                    )
                )
            )
        )
        return outputs.transpose(-1, -2)

class SASRec(nn.Module):
    def __init__(self, item_num, args):
        super(SASRec, self).__init__()
        self.item_num = item_num
        self.dev = args.device
        self.hidden_units = args.hidden_units
        
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate, batch_first=True)
            )
            self.forward_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))
        
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
    
    def log2feats(self, log_seqs):
        seqs = self.item_emb(log_seqs.to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        
        batch_size, seq_len = log_seqs.shape
        positions = torch.arange(1, seq_len + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1)
        positions = positions * (log_seqs != 0).long().to(self.dev)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)
        
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        
        for i in range(len(self.attention_layers)):
            normed_seqs = self.attention_layernorms[i](seqs)
            attn_output, _ = self.attention_layers[i](normed_seqs, normed_seqs, normed_seqs, attn_mask=attention_mask)
            seqs = seqs + attn_output
            
            normed_seqs = self.forward_layernorms[i](seqs)
            ffn_output = self.forward_layers[i](normed_seqs)
            seqs = seqs + ffn_output
            
        return self.last_layernorm(seqs)
    
    def forward(self, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(pos_seqs.to(self.dev))
        neg_embs = self.item_emb(neg_seqs.to(self.dev))
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits
    
    def predict(self, log_seqs, item_indices):
        """
        Args:
            log_seqs: (batch, maxlen)
            item_indices: (batch, n_items)
        """
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :] # (batch, hidden)
        item_embs = self.item_emb(item_indices.to(self.dev)) # (batch, n_items, hidden)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits

def train_epoch(model, dataloader, optimizer, args, epoch_idx):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx}", ncols=100)
    
    for log_seqs, pos_seqs, neg_seqs in pbar:
        log_seqs = log_seqs.to(args.device)
        pos_seqs = pos_seqs.to(args.device)
        neg_seqs = neg_seqs.to(args.device)
        
        pos_logits, neg_logits = model.forward(log_seqs, pos_seqs, neg_seqs)
        
        mask = (pos_seqs != 0).float()
        loss = -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-8) * mask
        loss = loss.sum() / mask.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{total_loss / num_batches:.4f}'})
    
    return total_loss / num_batches

def train_and_eval(model, dataloader, user_train, user_valid, user_test, itemnum, optimizer, args):
    print("=" * 50)
    print("Start Training SASRec")
    print(f"Total Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    
    evaluate_interval = 50
    print(f"Evaluate every {evaluate_interval} epochs")
    print("=" * 50)
    
    history = {
        'epochs': [],
        'losses': [],
        # Validation Metrics
        'valid_epochs': [],
        'valid_HR@1': [],
        'valid_HR@5': [],
        'valid_HR@10': [],
        'valid_NDCG@5': [],
        'valid_NDCG@10': [],
        # Test Metrics
        'test_epochs': [],
        'test_HR@1': [],
        'test_HR@5': [],
        'test_HR@10': [],
        'test_NDCG@5': [],
        'test_NDCG@10': []
    }
    
    best_valid_ndcg = 0.0
    
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_epoch(model, dataloader, optimizer, args, epoch)
        
        history['epochs'].append(epoch)
        history['losses'].append(train_loss)
        
        if epoch % evaluate_interval == 0:
            print(f"\n{'='*50}")
            print(f"Evaluating on Validation Set (Epoch {epoch})...")
            
            metrics = evaluate(model, user_train, user_valid, user_test, itemnum, args, mode='valid')
            
            print(f"Valid Metrics:")
            print(f"  HR@1:    {metrics['HR@1']:.4f}")
            print(f"  HR@5:    {metrics['HR@5']:.4f}")
            print(f"  HR@10:   {metrics['HR@10']:.4f}")
            print(f"  NDCG@5:  {metrics['NDCG@5']:.4f}")
            print(f"  NDCG@10: {metrics['NDCG@10']:.4f}")
            print(f"{'='*50}\n")
            
            history['valid_epochs'].append(epoch)
            history['valid_HR@1'].append(metrics['HR@1'])
            history['valid_HR@5'].append(metrics['HR@5'])
            history['valid_HR@10'].append(metrics['HR@10'])
            history['valid_NDCG@5'].append(metrics['NDCG@5'])
            history['valid_NDCG@10'].append(metrics['NDCG@10'])
            
            if metrics['NDCG@10'] > best_valid_ndcg:
                best_valid_ndcg = metrics['NDCG@10']
                torch.save(model.state_dict(), 'sasrec_best_model.pth')
                print(f"Best model saved! (NDCG@10: {best_valid_ndcg:.4f})")
        
            print(f"\n{'='*50}")
            print(f"Evaluating on Test Set (Epoch {epoch})...")
            
            metrics = evaluate(model, user_train, user_valid, user_test, itemnum, args, mode='test')
            
            print(f"Test Metrics:")
            print(f"  HR@1:    {metrics['HR@1']:.4f}")
            print(f"  HR@5:    {metrics['HR@5']:.4f}")
            print(f"  HR@10:   {metrics['HR@10']:.4f}")
            print(f"  NDCG@5:  {metrics['NDCG@5']:.4f}")
            print(f"  NDCG@10: {metrics['NDCG@10']:.4f}")
            print(f"{'='*50}\n")
            
            history['test_epochs'].append(epoch)
            history['test_HR@1'].append(metrics['HR@1'])
            history['test_HR@5'].append(metrics['HR@5'])
            history['test_HR@10'].append(metrics['HR@10'])
            history['test_NDCG@5'].append(metrics['NDCG@5'])
            history['test_NDCG@10'].append(metrics['NDCG@10'])

    print("\n" + "=" * 50)
    print("Training Completed!")
    print("Generating training curves...")
    plot_training_curves(history, model_name="SASRec")
    
    print("Loading best model for final test...")
    model.load_state_dict(torch.load('sasrec_best_model.pth'))
    
    final_metrics = evaluate(model, user_train, user_valid, user_test, itemnum, args, mode='test')
    
    print(f"\nFinal Test Metrics (Best Model):")
    print(f"  HR@1:    {final_metrics['HR@1']:.4f}")
    print(f"  HR@5:    {final_metrics['HR@5']:.4f}")
    print(f"  HR@10:   {final_metrics['HR@10']:.4f}")
    print(f"  NDCG@5:  {final_metrics['NDCG@5']:.4f}")
    print(f"  NDCG@10: {final_metrics['NDCG@10']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    class Args:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hidden_units = 128
        maxlen = 200
        dropout_rate = 0.1
        num_heads = 1
        num_blocks = 3
        lr = 0.001
        batch_size = 512
        num_epochs = 500
        dataset = 'ml-1m'
        dataset_root ='./data/'
        dataset_suffix='.txt'
    
    args = Args()
    
    dataset_path = f"{args.dataset_root}{args.dataset}{args.dataset_suffix}"
    [user_train, user_valid, user_test, usernum, itemnum] = data_partition(dataset_path)
    
    print(f"num users: {usernum}, num items: {itemnum}")
    
    dataset = SASRecDataset(user_train, usernum, itemnum, args.maxlen)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        collate_fn=SASRec_collate_fn,
        persistent_workers=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    model = SASRec(item_num=itemnum, args=args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train_and_eval(model, dataloader, user_train, user_valid, user_test, itemnum, optimizer, args)
    