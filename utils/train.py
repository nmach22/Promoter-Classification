import torch

class Train:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device="cpu"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.float().to(self.device)

            # Forward (already probability)
            probs = self.model(x)              # (B, 1) in [0,1]
            loss = self.criterion(probs, y)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            preds = (probs > 0.5).int()
            correct += (preds == y.int()).sum().item()
            total += y.size(0)

        return total_loss / len(self.train_loader), correct / total

    def val_epoch(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.float().to(self.device)

                probs = self.model(x)
                print(probs.shape,y.shape)
                loss = self.criterion(probs, y)

                total_loss += loss.item()

                preds = (probs > 0.5).int()
                correct += (preds == y.int()).sum().item()
                total += y.size(0)

        return total_loss / len(self.val_loader), correct / total

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.val_epoch()

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
