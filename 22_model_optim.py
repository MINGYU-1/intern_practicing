import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 사용자 정의 모듈 (기존 코드 유지)
from model.basic_1_22 import MultiDecoderCondVAE, integrated_loss_fn
from vae_earlystopping import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 실험 결과 저장을 위한 변수 ---
best_results = []
best_val_so_far = float('inf')
final_best_history = None  # 가장 성적이 좋은 시드의 학습 기록 저장용
final_best_seed = None

# 10번의 무작위 시드 실험 시작
seeds = np.random.randint(1, 1000, size=100)

for n in seeds:
    print(f"\n>>> Experiment with Seed: {n} starts.")
    
    # 데이터 로드 및 분할
    x_data = np.load('./data/metal.npy')
    c_data = np.load('./data/reaction.npy')
    x_train, x_test, c_train, c_test = train_test_split(x_data, c_data, random_state=n, test_size=0.4)
    x_val, x_test, c_val, c_test = train_test_split(x_test, c_test, random_state=n, test_size=0.5)

    # 스케일링
    x_scaler, c_scaler = StandardScaler(), StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    c_train = c_scaler.fit_transform(c_train)
    x_val, x_test = [x_scaler.transform(x) for x in [x_val, x_test]]
    c_val, c_test = [c_scaler.transform(c) for c in [c_val, c_test]]

    # 텐서 변환
    x_train, x_val, x_test = [torch.tensor(x, dtype=torch.float32) for x in [x_train, x_val, x_test]]
    c_train, c_val, c_test = [torch.tensor(c, dtype=torch.float32) for c in [c_train, c_val, c_test]]

    train_loader = DataLoader(TensorDataset(x_train, c_train), batch_size=64, shuffle=False)
    val_loader = DataLoader(TensorDataset(x_val, c_val), batch_size=64, shuffle=False)

    # 모델 및 초기화
    x_sample, c_sample = next(iter(train_loader))
    model = MultiDecoderCondVAE(x_dim=x_sample.shape[1], c_dim=c_sample.shape[1]).to(device)
    early_stopping = EarlyStopping(patience=40, min_delta=1e-9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    def get_beta_cyclical(epoch, total_epochs, n_cycles=4, beta_max=0.01):
        cycle_len = total_epochs // n_cycles
        relative_epoch = epoch % cycle_len
        return min(beta_max, (relative_epoch / (cycle_len * 0.5)) * beta_max)

    # 이번 시드의 학습 기록
    current_history = {'train': [], 'val': []}
    epochs = 300

    for epoch in range(1, epochs + 1):
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

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x_v, c_v in val_loader:
                x_v, c_v = x_v.to(device), c_v.to(device)
                v_mask_logits, _, _, v_recon_numeric, v_mu, v_logvar = model(x_v, c_v)
                v_loss_dict = integrated_loss_fn(v_mask_logits, v_recon_numeric, x_v, v_mu, v_logvar, beta=beta)
                v_loss += v_loss_dict['loss'].item()

        avg_train = t_loss / len(train_loader)
        avg_val = v_loss / len(val_loader)
        current_history['train'].append(avg_train)
        current_history['val'].append(avg_val)

        if early_stopping(avg_val, model):
            break

    # --- 최적 실험 정보 업데이트 ---
    min_val_loss = early_stopping.best_score if hasattr(early_stopping, 'best_score') else avg_val
    best_results.append({'seed': n, 'min_val_loss': min_val_loss})

    if min_val_loss < best_val_so_far:
        best_val_so_far = min_val_loss
        final_best_history = current_history.copy()
        final_best_seed = n

# ================= 시각화 파트 =================
res_df = pd.DataFrame(best_results)

plt.figure(figsize=(15, 6))
sns.set_style("whitegrid")

# 1. 왼쪽: 시드별 성능 분포 (보여주신 이미지 스타일)
plt.subplot(1, 2, 1)
sns.boxplot(data=res_df, y='min_val_loss', color='#3498db', width=0.3)
sns.stripplot(data=res_df, y='min_val_loss', color='#e74c3c', size=10, jitter=True, alpha=0.8)
plt.axhline(best_val_so_far, color='red', linestyle='--', alpha=0.6)
plt.title(f'Model Stability (100 Seeds)\nBest Seed: {final_best_seed}', fontsize=14)
plt.ylabel('Min Validation Loss')

# 2. 오른쪽: 최적 시드의 학습 곡선 (정상 작동 확인용)
plt.subplot(1, 2, 2)
plt.plot(final_best_history['train'], label='Train Loss', color='#2ecc71', lw=2)
plt.plot(final_best_history['val'], label='Val Loss', color='#e67e22', lw=2)
plt.title(f'Learning Curve of Best Seed ({final_best_seed})', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print(f"\n최적 시드: {final_best_seed} | 최저 손실: {best_val_so_far:.6f}")