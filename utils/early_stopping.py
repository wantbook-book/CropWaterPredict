import numpy as np
import torch

class EarlyStopping:
    """早停法以避免过拟合"""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        参数:
        - patience: 性能没有提升的epochs数，之后训练将被停止
        - verbose: 如果为True，则打印一条信息，表明早停法被触发
        - delta: 改善被认为显著的最小变化
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                if self.verbose:
                    print("Early stopping")
                self.early_stop = True
        else:
            self.best_score = score
            self.epochs_no_improve = 0
