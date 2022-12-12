import numpy as np
from torchvision import transforms


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, current_dir=''):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta
        #print(f"delta {delta}")
        self.current_dir = current_dir
        self.best_patch = None


    def __call__(self, val_loss, patch, epoch):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, patch, epoch)
        elif score >= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}', flush=True)
            if self.counter >= self.patience:
                print("Training stopped - early stopping")
                return True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, patch, epoch)
            self.counter = 0
        return False

    def save_checkpoint(self, val_loss, patch, epoch):
        """
        Saves model when validation loss decreases.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving patch ...')
        transforms.ToPILImage()(patch).save(self.current_dir +
                                                       '/saved_patches' +
                                                       '/patch_' +
                                                       str(epoch) +
                                                       '.png', 'PNG')

        self.best_patch = patch
        self.val_loss_min = val_loss
