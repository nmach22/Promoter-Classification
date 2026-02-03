import torch
import matplotlib.pyplot as plt

from eval.train_evals import TrainEvals
from utils.get_device import get_device


class Train:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        eval_list=None,
        device=None
    ):
        self.eval_list = eval_list or [TrainEvals()]
        self.device = device or get_device()

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion

        # History tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': {},
            'val_metrics': {}
        }

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total = 0
        epoch_output = []

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.float().to(self.device)

            out = self.model(x)
            loss = self.criterion(out, y)
            epoch_output.append((out,y))

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            total += y.size(0)

        return total_loss / len(self.train_loader), epoch_output

    def val_epoch(self):
        self.model.eval()
        total_loss = 0
        total = 0

        epoch_output = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.float().to(self.device)

                out = self.model(x)
                loss = self.criterion(out, y)
                epoch_output.append((out,y))

                total_loss += loss.item()
                total += y.size(0)

        return total_loss / len(self.val_loader), epoch_output

    def train(self, num_epochs, plot=False):
        for epoch in range(num_epochs):
            train_loss, train_out = self.train_epoch()
            val_loss,val_out = self.val_epoch()

            # Store losses
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            for f in self.eval_list:
                train_eval = f.evaluate(train_out)
                val_eval = f.evaluate(val_out)

                for k in train_eval:
                    # Initialize metric lists if not exists
                    if k not in self.history['train_metrics']:
                        self.history['train_metrics'][k] = []
                        self.history['val_metrics'][k] = []

                    # Store metrics
                    self.history['train_metrics'][k].append(train_eval[k])
                    self.history['val_metrics'][k].append(val_eval[k])

                    print(f"  Train {k}: {train_eval[k]:.4f}")
                    print(f"  Val {k}: {val_eval[k]:.4f}")

        if plot:
            self.plot_history()

    def plot_history(self, figsize=(8, 6)):
        """Plot training and validation losses and metrics as separate plots."""
        epochs = range(1, len(self.history['train_loss']) + 1)

        figures = []

        # Plot loss in a separate figure
        fig_loss = plt.figure(figsize=figsize)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        figures.append(fig_loss)

        # Plot each metric in a separate figure
        for metric_name in self.history['train_metrics'].keys():
            fig_metric = plt.figure(figsize=figsize)
            plt.plot(epochs, self.history['train_metrics'][metric_name], 'b-', label=f'Train {metric_name}')
            plt.plot(epochs, self.history['val_metrics'][metric_name], 'r-', label=f'Val {metric_name}')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.title(f'Training and Validation {metric_name}')
            plt.legend()
            plt.grid(True)
            plt.show()
            figures.append(fig_metric)

        return figures
