# utils.py
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def data_partition(dataset_path):
    df = pd.read_csv(dataset_path, sep=' ', names=['user_id', 'item_id'])

    usernum = df['user_id'].max()
    itemnum = df['item_id'].max()
    
    user_data = df.groupby('user_id')['item_id'].apply(list)

    user_train = {}
    user_valid = {}
    user_test = {}

    for user_id, items in user_data.items():
        if len(items) < 4:
            raise ValueError(f"User {user_id} has less than 4 interactions.")

        user_train[user_id] = items[:-2] # 训练集：所有交互除了最后两个
        user_valid[user_id] = [items[-2]] # 验证集：倒数第二个交互
        user_test[user_id] = [items[-1]] # 测试集：最后一个交互

    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, user_train, user_valid, user_test, itemnum, args, mode='valid', num_eval_users=10000):
    """
    Args:
        mode: 'valid' or 'test'
    """
    model.eval()
    
    metrics = {
        'HR@1': 0.0,
        'HR@5': 0.0,
        'HR@10': 0.0,
        'NDCG@5': 0.0,
        'NDCG@10': 0.0
    }
    count = 0
    
    all_users = list(user_train.keys())
    if len(all_users) > num_eval_users:
        eval_users = np.random.choice(all_users, num_eval_users, replace=False)
    else:
        eval_users = all_users
    
    with torch.no_grad():
        for u in eval_users:
            if len(user_train[u]) < 1: 
                continue
            
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            
            if mode == 'valid':
                if u not in user_valid or len(user_valid[u]) == 0: continue
                target_item = user_valid[u][0]
                input_seq_list = user_train[u]
                rated_set = set(user_train[u])
            else:
                if u not in user_test or len(user_test[u]) == 0: continue
                target_item = user_test[u][0]
                input_seq_list = user_train[u] + user_valid[u]
                rated_set = set(user_train[u]) | set(user_valid[u])
                
            # back to front
            for i in reversed(input_seq_list):
                seq[idx] = i
                idx -= 1
                if idx == -1: break
            
            rated_set.add(0)
            rated_set.add(target_item)
            
            # 1 positive + 100 negative for candidate
            item_idx = [target_item]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated_set:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)
            

            seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(args.device)
            item_tensor = torch.LongTensor([item_idx]).to(args.device)

            predictions = model.predict(seq_tensor, item_tensor)
            predictions = predictions[0].cpu().numpy() 
            
            rank = (predictions.argsort()[::-1]).tolist().index(0)
            
            count += 1
            
            if rank < 1:
                metrics['HR@1'] += 1
            if rank < 5:
                metrics['HR@5'] += 1
                metrics['NDCG@5'] += 1 / np.log2(rank + 2)
            if rank < 10:
                metrics['HR@10'] += 1
                metrics['NDCG@10'] += 1 / np.log2(rank + 2)
    
    if count == 0:
        return metrics
        
    for k in metrics.keys():
        metrics[k] /= count
        
    return metrics

def plot_training_curves(history, model_name, save_path='training_curves.png'):
    """
    draw trainning curve
    Args:
        history: histroy data dic
        save_path: picture path
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{model_name} Training Curves', fontsize=16, fontweight='bold')
    save_path = f"{model_name}_training_curves.png"
    
    # Training Loss
    ax = axes[0]
    epochs = history['epochs']
    losses = history['losses']
    ax.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Validation Metrics
    ax = axes[1]
    if history['valid_epochs']:
        ax.plot(history['valid_epochs'], history['valid_HR@1'], 
                '-o', linewidth=2, markersize=5, label='HR@1')
        ax.plot(history['valid_epochs'], history['valid_HR@5'], 
                '-s', linewidth=2, markersize=5, label='HR@5')
        ax.plot(history['valid_epochs'], history['valid_HR@10'], 
                '-^', linewidth=2, markersize=5, label='HR@10')
        ax.plot(history['valid_epochs'], history['valid_NDCG@5'], 
                '-d', linewidth=2, markersize=5, label='NDCG@5')
        ax.plot(history['valid_epochs'], history['valid_NDCG@10'], 
                '-p', linewidth=2, markersize=5, label='NDCG@10')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Validation Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Test Metrics
    ax = axes[2]
    if history['test_epochs']:
        ax.plot(history['test_epochs'], history['test_HR@1'], 
                '-o', linewidth=2, markersize=5, label='HR@1')
        ax.plot(history['test_epochs'], history['test_HR@5'], 
                '-s', linewidth=2, markersize=5, label='HR@5')
        ax.plot(history['test_epochs'], history['test_HR@10'], 
                '-^', linewidth=2, markersize=5, label='HR@10')
        ax.plot(history['test_epochs'], history['test_NDCG@5'], 
                '-d', linewidth=2, markersize=5, label='NDCG@5')
        ax.plot(history['test_epochs'], history['test_NDCG@10'], 
                '-p', linewidth=2, markersize=5, label='NDCG@10')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Test Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to: {save_path}")
    plt.close()