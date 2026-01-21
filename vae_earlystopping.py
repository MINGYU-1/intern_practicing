import copy
import torch

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None  # 최적의 모델 가중치를 저장할 변수
        self.early_stop = False

    def __call__(self, val_loss, model):
        # loss가 개선된 경우
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # .state_dict()는 OrderedDict를 반환하므로 copy.deepcopy로 완벽히 복사합니다.
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        
        # loss 개선이 없는 경우
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

    def load_best_model(self, model):
        """학습 종료 후 최적의 가중치를 모델에 다시 로드"""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            print(f"Restored best model with loss: {self.best_loss:.6f}")