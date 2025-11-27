# BERT4Rec.py
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict

from dataloader import BERT4RecDataset, BERT4Rec_collate_fn
from utils import data_partition, evaluate, plot_training_curves

class PointWiseFeedForward(nn.Module):
    """
    MLP in the embedding hidden layer, BERT4Rec use Gelu, SASRec use Relu
    """
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)
    
    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(
                self.gelu(
                    self.dropout1(
                        self.conv1(inputs.transpose(-1, -2))
                    )
                )
            )
        )
        return outputs.transpose(-1, -2)


class Bert4Rec(nn.Module):
    def __init__(self, item_num, args):
        super(Bert4Rec, self).__init__()
        
        self.item_num = item_num
        self.mask_id = item_num + 1
        self.dev = args.device
        self.hidden_units = args.hidden_units
        self.mask_prob = args.mask_prob
        # Embeddings
        # 0 is padding, item_num+1 is mask_id
        self.item_emb = nn.Embedding(
            self.item_num + 2,  
            args.hidden_units,  
            padding_idx=0
        )
        self.pos_emb = nn.Embedding(
            args.maxlen + 1,  
            args.hidden_units,  
            padding_idx=0
        )
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        
        for _ in range(args.num_blocks):
            new_attn_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            
            new_attn_layer = nn.MultiheadAttention(
                args.hidden_units,
                args.num_heads,
                args.dropout_rate,
                batch_first=True
            )
            self.attention_layers.append(new_attn_layer)
            
            new_fwd_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
        
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
    
    def seq2feats(self, log_seqs):
        """
        Args:
            log_seqs: (batch_size, maxlen) - input sequence
        Returns:
            seq_feats: (batch_size, maxlen, hidden_units)
        """
        # Item Embedding
        seqs = self.item_emb(log_seqs.to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        
        # Position Embedding
        batch_size, seq_len = log_seqs.shape
        positions = torch.arange(1, seq_len + 1, device=self.dev).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        positions = positions * (log_seqs != 0).long().to(self.dev)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)
        
        # padding mask, True mean mask the position
        padding_mask = (log_seqs == 0).to(self.dev)
        
        # LayerNorm - Attention - residual - 
        # ForwardNorm - feedforward - residual - LastNorm
        for i in range(len(self.attention_layers)):
            normed_seqs = self.attention_layernorms[i](seqs)
            attn_output, _ = self.attention_layers[i](
                normed_seqs, normed_seqs, normed_seqs,
                key_padding_mask=padding_mask
            )
            seqs = seqs + attn_output
            normed_seqs = self.forward_layernorms[i](seqs)
            ffn_output = self.forward_layers[i](normed_seqs)
            seqs = seqs + ffn_output
        
        seq_feats = self.last_layernorm(seqs)
        
        return seq_feats
        
    def masked_prediction(self, log_seqs, masked_pos):
        """
        Args:
            log_seqs: (batch_size, maxlen)
            masked_pos: (batch_size, maxlen)
        Returns:
            logits: (batch_size, maxlen, item_num)
        """
        log_feats = self.seq2feats(log_seqs)
        
        all_item_embs = self.item_emb.weight[1:self.item_num+1]
        
        logits = torch.matmul(log_feats, all_item_embs.transpose(0, 1))
        
        logits = logits / (self.hidden_units ** 0.5)

        return logits
    
    def forward(self, log_seqs, labels):
        """
        Args:
            log_seqs: (batch_size, maxlen)
            labels: (batch_size, maxlen)
        Returns:
            loss: scalar - Cross Entropy Loss
        """
        logits = self.masked_prediction(log_seqs, labels > 0)
        
        # loss in mask position
        mask_positions = (labels > 0).to(self.dev)
        
        logits_flat = logits.view(-1, self.item_num)
        labels_flat = labels.view(-1).to(self.dev)
        
        # cross_entropy need 0-based index
        labels_flat = labels_flat - 1
        
        # ignore_index=-1 ignore the -1 position which are not masked
        loss = nn.functional.cross_entropy(
            logits_flat, 
            labels_flat, 
            ignore_index=-1
        )
        
        return loss

    def predict(self, log_seqs, item_indices):
        """
        Args:
            log_seqs: (batch_size, maxlen)
            item_indices: (batch_size,) or (batch_size, num_items)
        Returns:
            logits: (batch_size,) or (batch_size, num_items)
        """
        seq_feats = self.seq2feats(log_seqs)  # (batch, maxlen, hidden)
        
        final_feat = seq_feats[:, -1, :]
        
        item_embs = self.item_emb(item_indices.to(self.dev))
        
        logits = torch.matmul(item_embs, final_feat.unsqueeze(-1)).squeeze(-1)
        
        logits = logits / (self.hidden_units ** 0.5)

        return logits


def random_mask_sequence(seq, mask_prob, item_num, mask_id):
    """
    random mask items in sequence (15%-20%)
    """
    masked_seq = seq.clone()
    labels = torch.zeros_like(seq)
    
    valid_positions = (seq > 0).nonzero(as_tuple=True)[0]
    
    if len(valid_positions) == 0:
        return masked_seq, labels
    
    num_to_mask = max(1, int(len(valid_positions) * mask_prob))
    mask_positions = np.random.choice(valid_positions.cpu().numpy(), num_to_mask, replace=False)
    
    for pos in mask_positions:
        labels[pos] = seq[pos]
        masked_seq[pos] = mask_id
    
    return masked_seq, labels

def train_epoch(model, dataloader, optimizer, args, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, seqs in enumerate(pbar):
        batch_size = seqs.size(0)
        
        masked_seqs = []
        labels_list = []
        
        for i in range(batch_size):
            masked_seq, labels = random_mask_sequence(
                seqs[i], args.mask_prob, args.itemnum, args.mask_id
            )
            masked_seqs.append(masked_seq)
            labels_list.append(labels)
        
        masked_seqs = torch.stack(masked_seqs)
        labels = torch.stack(labels_list)
        
        loss = model.forward(masked_seqs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    return avg_loss

def train(model, dataloader, user_train, user_valid, user_test, itemnum, optimizer, args):
    print("=" * 50)
    print("Start Training BERT4Rec")
    print(f"Total Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Mask Probability: {args.mask_prob}")
    
    evaluate_interval = 1
    print(f"Evaluate every {evaluate_interval} epochs")
    print("=" * 50)
    
    best_valid_ndcg = 0.0
    
    history = {
        'epochs': [],
        'losses': [],
        'valid_epochs': [],
        'valid_HR@1': [],
        'valid_HR@5': [],
        'valid_HR@10': [],
        'valid_NDCG@5': [],
        'valid_NDCG@10': [],
        'test_epochs': [],
        'test_HR@1': [],
        'test_HR@5': [],
        'test_HR@10': [],
        'test_NDCG@5': [],
        'test_NDCG@10': []
    }
    
    for epoch in range(1, args.num_epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, args, epoch)
        
        history['epochs'].append(epoch)
        history['losses'].append(avg_loss)
        
        # evaluate on validation each X epochs
        if epoch % evaluate_interval == 0:
            print(f"\n{'='*50}")
            print(f"Evaluating on Validation Set (Epoch {epoch})...")
            valid_metrics = evaluate(model, user_train, user_valid, user_test, itemnum, args, mode='valid')

            print(f"Valid Metrics:")
            print(f"  HR@1:    {valid_metrics['HR@1']:.4f}")
            print(f"  HR@5:    {valid_metrics['HR@5']:.4f}")
            print(f"  HR@10:   {valid_metrics['HR@10']:.4f}")
            print(f"  NDCG@5:  {valid_metrics['NDCG@5']:.4f}")
            print(f"  NDCG@10: {valid_metrics['NDCG@10']:.4f}")
            print(f"{'='*50}\n")
            
            history['valid_epochs'].append(epoch)
            history['valid_HR@1'].append(valid_metrics['HR@1'])
            history['valid_HR@5'].append(valid_metrics['HR@5'])
            history['valid_HR@10'].append(valid_metrics['HR@10'])
            history['valid_NDCG@5'].append(valid_metrics['NDCG@5'])
            history['valid_NDCG@10'].append(valid_metrics['NDCG@10'])
            
            # save best model
            if valid_metrics['NDCG@10'] > best_valid_ndcg:
                best_valid_ndcg = valid_metrics['NDCG@10']
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"Best model saved! (NDCG@10: {best_valid_ndcg:.4f})")
        
            print(f"\n{'='*50}")
            print(f"Evaluating on Test Set (Epoch {epoch})...")

            test_metrics = evaluate(model, user_train, user_valid, user_test, itemnum, args, mode='test')
            print(f"Test Metrics:")
            print(f"  HR@1:    {test_metrics['HR@1']:.4f}")
            print(f"  HR@5:    {test_metrics['HR@5']:.4f}")
            print(f"  HR@10:   {test_metrics['HR@10']:.4f}")
            print(f"  NDCG@5:  {test_metrics['NDCG@5']:.4f}")
            print(f"  NDCG@10: {test_metrics['NDCG@10']:.4f}")
            print(f"{'='*50}\n")
            
            history['test_epochs'].append(epoch)
            history['test_HR@1'].append(test_metrics['HR@1'])
            history['test_HR@5'].append(test_metrics['HR@5'])
            history['test_HR@10'].append(test_metrics['HR@10'])
            history['test_NDCG@5'].append(test_metrics['NDCG@5'])
            history['test_NDCG@10'].append(test_metrics['NDCG@10'])
    
    print("\n" + "=" * 50)
    print("Training Completed!")
    print("Generating training curves...")
    plot_training_curves(history, model_name="BERT4Rec")
    
    print("Loading best model for final test...")
    model.load_state_dict(torch.load('best_model.pth'))
    
    print(f"\nFinal Test Metrics:")
    print(f"  HR@1:    {final_metrics['HR@1']:.4f}")
    print(f"  HR@5:    {final_metrics['HR@5']:.4f}")
    print(f"  HR@10:   {final_metrics['HR@10']:.4f}")
    print(f"  NDCG@5:  {final_metrics['NDCG@5']:.4f}")
    print(f"  NDCG@10: {final_metrics['NDCG@10']:.4f}")
    print("=" * 50)
    
    return final_metrics, history


if __name__ == "__main__":
    class Args:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hidden_units = 64
        maxlen = 50
        dropout_rate = 0.1
        num_heads = 1
        num_blocks = 2
        lr = 0.001
        batch_size = 512
        num_epochs = 5
        mask_prob = 0.2
        dataset = 'ml-1m'
        dataset_root = './data/'
        dataset_suffix = '.txt'
        itemnum = -1
        mask_id = -1
    
    args = Args()
    
    dataset_path = f"{args.dataset_root}{args.dataset}{args.dataset_suffix}"
    [user_train, user_valid, user_test, usernum, itemnum] = data_partition(dataset_path)
    
    args.itemnum = itemnum
    args.mask_id = itemnum + 1
    
    num_batch = len(user_train) // args.batch_size
    total_actions = sum(len(actions) for actions in user_train.values())
    print(f"num batch: {num_batch}, total actions: {total_actions}, user_train len: {len(user_train)}")
    print(f"average sequence length: {(total_actions / len(user_train)):.2f}")
    
    print("Loading dataset...")
    dataset = BERT4RecDataset(user_train, usernum, itemnum, args.maxlen)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        collate_fn=BERT4Rec_collate_fn,
        persistent_workers=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Dataset size: {len(dataset)}, Batches: {len(dataloader)}")
    
    model = Bert4Rec(item_num=itemnum, args=args)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    final_metrics, history = train(model, dataloader, user_train, user_valid, user_test, itemnum, optimizer, args)
