import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 사용자 정의 모듈 (환경에 맞게 경로 확인 필요)
from model.basic_1_22 import MultiDecoderCondVAE, integrated_loss_fn
from vae_earlystopping import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 실험 설정 ---
num_experiments = 1
seeds = np.random.randint(1, 10000, size=num_experiments)
best_results = []
best_val_so_far = float('inf')
final_best_history = None 
final_best_seed = None

# 임계값 설정 (분류 지표 계산용)
threshold = 0.5 

for idx, n in enumerate(seeds):
    print(f"\n>>> [{idx+1}/{num_experiments}] Experiment with Seed: {n}")
    
    # 1. 데이터 로드 및 분할
    x_data = np.load('./data/metal.npy')
    c_data = np.load('./data/reaction.npy')
    x_train, x_test, c_train, c_test = train_test_split(x_data, c_data, random_state=n, test_size=0.4)
    x_val, x_test, c_val, c_test = train_test_split(x_test, c_test, random_state=n, test_size=0.5)

    # 2. 스케일링
    x_scaler, c_scaler = StandardScaler(), StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    c_train = c_scaler.fit_transform(c_train)
    x_val, x_test = [x_scaler.transform(x) for x in [x_val, x_test]]
    c_val, c_test = [c_scaler.transform(c) for c in [c_val, c_test]]

    # 3. 텐서 변환 및 로더
    x_train, x_val, x_test = [torch.tensor(x, dtype=torch.float32) for x in [x_train, x_val, x_test]]
    c_train, c_val, c_test = [torch.tensor(c, dtype=torch.float32) for c in [c_train, c_val, c_test]]

    train_loader = DataLoader(TensorDataset(x_train, c_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, c_val), batch_size=64, shuffle=False)

    # 4. 모델 초기화
    model = MultiDecoderCondVAE(x_dim=x_train.shape[1], c_dim=c_train.shape[1]).to(device)
    early_stopping = EarlyStopping(patience=40, min_delta=1e-9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    def get_beta_cyclical(epoch, total_epochs, n_cycles=4, beta_max=0.01):
        cycle_len = total_epochs // n_cycles
        relative_epoch = epoch % cycle_len
        return min(beta_max, (relative_epoch / (cycle_len * 0.5)) * beta_max)

    # --- 히스토리 저장용 리스트 (에포크 루프 밖에서 초기화) ---
    current_history = {'train': [], 'val': [], 'precision': [], 'recall': [], 'f1': [], 'accuracy': []}
    epochs = 300

    for epoch in range(1, epochs + 1):
        # [Train]
        model.train()
        t_loss = 0
        beta = get_beta_cyclical(epoch, epochs)
        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            optimizer.zero_grad()
            mask_logits, prob_mask, mask_out, recon_numeric, z_mu, z_logvar = model(x, c)
            loss_dict = integrated_loss_fn(mask_logits, recon_numeric, x, z_mu, z_logvar, beta=beta)
            loss_dict['loss'].backward()
            optimizer.step()
            t_loss += loss_dict['loss'].item()

        # [Validation & Metrics]
        model.eval()
        v_loss = 0
        y_true_all, y_pred_all = [], []
        
        with torch.no_grad():
            for x_v, c_v in val_loader:
                x_v, c_v = x_v.to(device), c_v.to(device)
                v_mask_logits, _, _, v_recon_numeric, v_mu, v_logvar = model(x_v, c_v)
                v_loss_dict = integrated_loss_fn(v_mask_logits, v_recon_numeric, x_v, v_mu, v_logvar, beta=beta)
                v_loss += v_loss_dict['loss'].item()

                # 지표 계산 (Binary Mask 예측 성능 확인)
                eps = 1e-6
                y_true = (x_v.abs() > eps).float()
                y_prob = torch.sigmoid(v_mask_logits)
                y_pred = (y_prob >= threshold).float()

                y_true_all.append(y_true.cpu().numpy().ravel())
                y_pred_all.append(y_pred.cpu().numpy().ravel())

        # 에포크 결과 계산
        avg_train = t_loss / len(train_loader)
        avg_val = v_loss / len(val_loader)
        
        y_true_final = np.concatenate(y_true_all)
        y_pred_final = np.concatenate(y_pred_all)
        
        prec = precision_score(y_true_final, y_pred_final, average='micro', zero_division=0)
        rec  = recall_score(y_true_final, y_pred_final, average='micro', zero_division=0)
        f1   = f1_score(y_true_final, y_pred_final, average='micro', zero_division=0)
        acc  = accuracy_score(y_true_final, y_pred_final)

        # 히스토리 업데이트
        current_history['train'].append(avg_train)
        current_history['val'].append(avg_val)
        current_history['precision'].append(prec)
        current_history['recall'].append(rec)
        current_history['f1'].append(f1)
        current_history['accuracy'].append(acc)

        if early_stopping(avg_val, model):
            print(f"Early Stopping at epoch {epoch}")
            break

    # --- 최적 실험 정보 업데이트 ---
    min_val_loss = early_stopping.best_score if hasattr(early_stopping, 'best_score') else avg_val
    best_results.append({'seed': n, 'min_val_loss': min_val_loss})

    if min_val_loss < best_val_so_far:
        best_val_so_far = min_val_loss
        final_best_history = {k: v[:] for k, v in current_history.items()} # Deep copy
        final_best_seed = n

# ================= 최종 시각화 파트 =================
res_df = pd.DataFrame(best_results)
sns.set_style("whitegrid")
plt.figure(figsize=(20, 6))

# 1. 시드별 성능 분포 (Stability 확인)
plt.subplot(1, 3, 1)
sns.boxplot(data=res_df, y='min_val_loss', color='#3498db', width=0.3)
sns.stripplot(data=res_df, y='min_val_loss', color='#e74c3c', size=6, jitter=True, alpha=0.6)
plt.title(f'Model Stability ({num_experiments} Seeds)\nBest Seed: {final_best_seed}', fontsize=13)

# 2. 최적 시드의 Loss 곡선
plt.subplot(1, 3, 2)
plt.plot(final_best_history['train'], label='Train Loss', color='#2ecc71', lw=2)
plt.plot(final_best_history['val'], label='Val Loss', color='#e67e22', lw=2)
plt.title(f'Loss Curve\n(Seed: {final_best_seed})', fontsize=13)
plt.xlabel('Epochs')
plt.legend()

# 3. 최적 시드의 4대 평가지표 곡선
plt.subplot(1, 3, 3)
h = final_best_history
plt.plot(h['precision'], label='Precision', linestyle='--', alpha=0.7)
plt.plot(h['recall'], label='Recall', linestyle='--', alpha=0.7)
plt.plot(h['f1'], label='F1-score', lw=3, color='#9b59b6')
plt.plot(h['accuracy'], label='Accuracy', lw=1.5, color='#2c3e50')
plt.title(f'Evaluation Metrics\n(Seed: {final_best_seed})', fontsize=13)
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

print(f"\n[실험 종료] 최적 시드: {final_best_seed} | 최저 검증 손실: {best_val_so_far:.6f}")