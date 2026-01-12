class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.best_state = None  # best model weights 저장

    def step(self, val_loss, model):
        # 개선되면(best 갱신)
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            # best weight 저장(파일 저장 없이 메모리에 저장)
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False  # stop 아님
        else:
            self.counter += 1
            return self.counter >= self.patience  # patience 넘으면 stop
