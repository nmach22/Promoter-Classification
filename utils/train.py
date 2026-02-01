import torch

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

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_out = self.train_epoch()
            val_loss,val_out = self.val_epoch()

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            for f in self.eval_list:
                f.evaluate("Train: ",train_out)
                f.evaluate("Val: ",val_out)
