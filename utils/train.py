import torch
import matplotlib.pyplot as plt

class Train:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        eval_list = [],
        device="cpu"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.eval_list = eval_list

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

        # return self.history

    def plot_history(self, figsize=(12, 4)):
        """Plot training and validation losses and metrics."""
        num_metrics = 1 + len(self.history['train_metrics'])
        fig, axes = plt.subplots(1, num_metrics, figsize=(figsize[0], figsize[1]))

        if num_metrics == 1:
            axes = [axes]

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Plot loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Plot each metric
        for idx, metric_name in enumerate(self.history['train_metrics'].keys(), 1):
            axes[idx].plot(epochs, self.history['train_metrics'][metric_name], 'b-', label=f'Train {metric_name}')
            axes[idx].plot(epochs, self.history['val_metrics'][metric_name], 'r-', label=f'Val {metric_name}')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric_name)
            axes[idx].set_title(f'Training and Validation {metric_name}')
            axes[idx].legend()
            axes[idx].grid(True)

        plt.tight_layout()
        plt.show()

        return fig
