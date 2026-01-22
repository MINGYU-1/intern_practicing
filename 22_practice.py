import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model.basic_1_22 import MultiDecoderCondVAE,integrated_loss_fn
from vae_earlystopping import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_results = []

for n in np.random.randint(1,45,size =100):
    x_data = np.load('./data/metal.npy')
    c_data = np.load('./data/reaction.npy')
    x_train,x_test,c_train,c_test = train_test_split(x_data,c_data,random_state = n,test_size = 0.4)
    x_val,x_test,c_val,c_test = train_test_split(x_test,c_test,random_state = n, test_size = 0.5)

    x_scaler = StandardScaler()
    c_scaler = StandardScaler()

    x_train = x_scaler.fit_transform(x_train)
    c_train = c_scaler.fit_transform(c_train)

    x_val,x_test = [x_scaler.transform(x) for x in [x_val,x_test]]
    c_val,c_test = [c_scaler.transform(c) for c in [c_val,c_test]]

    x_train,x_val,x_test = [torch.tensor(x, dtype = torch.float32) for x in [x_train,x_val,x_test]]
    c_train,c_val,c_test = [torch.tensor(c, dtype = torch.float32) for c in [c_train,c_val,c_test]]

    train_data = [x_train,c_train]
    val_data = [x_val,c_val]
    test_data = [x_test,c_test]
    train_data = TensorDataset(*train_data)
    val_data = TensorDataset(*val_data)
    test_data = TensorDataset(*test_data)
    datas = [train_data,val_data,test_data]
    train_loader,val_loader,test_loader = [DataLoader(x,batch_size = 64,shuffle=False) for x in datas]
    ## 학습과정
    x_sample, c_sample = next(iter(train_loader))
    model = MultiDecoderCondVAE(x_dim=x_sample.shape[1], c_dim=c_sample.shape[1]).to(device)
    early_stopping = EarlyStopping(patience = 40, min_delta=1e-9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)

    def get_beta_cyclical(epoch, total_epochs, n_cycles=4, beta_max=0.01):
        cycle_len = total_epochs // n_cycles
        relative_epoch = epoch % cycle_len
        # 각 사이클의 전반부 50% 동안만 선형적으로 증가
        beta = min(beta_max, (relative_epoch / (cycle_len * 0.5)) * beta_max)
        return beta

    # 3. 학습 루프
    history = {'train_total': [], 'val_total': [] }
    epochs = 300


    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_bce, t_mse, t_kl = 0, 0, 0, 0
        # KL Annealing: 학습 초반엔 재구성에, 후반엔 분포 학습에 집중
        beta = get_beta_cyclical(epoch,epochs)

        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            
            optimizer.zero_grad()
            
            # 모델 포워드 (제시해주신 구조)
            mask_logits, prob_mask, mask_out, recon_numeric, z_mu, z_logvar = model(x,c)
            
            # 손실 함수 계산
            loss_dict = integrated_loss_fn(mask_logits, recon_numeric,x, z_mu, z_logvar, beta=beta)
            
            loss_dict['loss'].backward()
            optimizer.step()
            
            t_loss += loss_dict['loss'].item()
            t_bce += loss_dict['bce'].item()
            t_mse += loss_dict['mse'].item()
            t_kl += loss_dict['kl'].item()

        # 4. 검증 (Validation)
        model.eval()
        v_loss = 0
        v_kl = 0
        with torch.no_grad():
            for x_v, c_v in val_loader:
                x_v, c_v = x_v.to(device), c_v.to(device)
                v_mask_logits, v_prob_mask, v_mask_out, v_recon_numeric, v_mu, v_logvar = model(x_v, c_v)
                v_loss_dict = integrated_loss_fn(v_mask_logits,  v_recon_numeric,x_v, v_mu, v_logvar, beta=beta)
                v_loss += v_loss_dict['loss'].item()
                v_kl += v_loss_dict['kl'].item()

        # 결과 기록
        avg_train_loss = t_loss / len(train_loader)
        avg_val_loss = v_loss / len(val_loader)
        avg_kl_loss = v_kl/len(val_loader)

        if early_stopping(avg_val_loss, model): 
            print(f"Early Stopping at Epoch {epoch}")
            break

        best_results.append({
        'seed': n,
        'min_val_loss': early_stopping.best_score if hasattr(early_stopping, 'best_score') else avg_val_loss
    })
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 시각화를 위한 데이터 정리 (리스트 -> 데이터프레임)
# 각 시드별로 'Best Val Loss'를 모아서 정리합니다.
res_df = pd.DataFrame(best_results)

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid") # 깔끔한 배경 설정

# 2. 이미지와 같은 느낌의 박스플롯 + 스트립플롯 조합
# y축에 Loss 값을 넣고, 전체적인 분포를 봅니다.
sns.boxplot(data=res_df, y='min_val_loss', color='#3498db', width=0.4) 
sns.stripplot(data=res_df, y='min_val_loss', color='#e74c3c', size=8, jitter=True, alpha=0.7)

# 3. 그래프 정보 추가
plt.title('Model Stability Across 10 Random Seeds', fontsize=15, pad=20)
plt.ylabel('Minimum Validation Loss', fontsize=12)
plt.xlabel('CVAE Model', fontsize=12) # x축은 모델 이름 등으로 설정

# 최저점(Best Seed)에 주석 달기
min_loss = res_df['min_val_loss'].min()
plt.axhline(min_loss, color='red', linestyle='--', alpha=0.5)
plt.text(0.25, min_loss, f' Best: {min_loss:.6f}', color='red', fontweight='bold')

plt.tight_layout()
plt.show()

best_experiment = min(best_results, key=lambda x: x['min_val_loss'])
print("\n" + "="*30)
print(f"최적의 데이터 분할 시드: {best_experiment['seed']}")
print(f"최저 검증 손실: {best_experiment['min_val_loss']:.6f}")
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 시드별 성능 편차 확인 (모델의 안정성 검증)
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
all_losses = [res['min_val_loss'] for res in best_results]
sns.boxplot(y=all_losses, color='lightblue')
sns.stripplot(y=all_losses, color='red', alpha=0.5) # 개별 데이터 포인트
plt.title('Validation Loss Distribution (10 Seeds)')
plt.ylabel('Min Val Loss')

# 2. 최적 시드의 학습 곡선 (과적합 및 수렴 확인)
plt.subplot(1, 2, 2)
# 주의: 이 그래프는 루프의 마지막 실행 결과 혹은 history를 따로 저장했을 때 유효합니다.
plt.plot(history['train_total'], label='Train Loss', color='blue', alpha=0.7)
plt.plot(history['val_total'], label='Val Loss', color='orange', alpha=0.7)
plt.title(f'Learning Curve (Best Seed: {best_experiment["seed"]})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()