import torch, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from models import ResMLP2

def train_model(X_train, y_train, X_val, y_val, num_classes, epochs=50):
    # scale features
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)

    # create DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val,   dtype=torch.float32),
                             torch.tensor(y_val,   dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64)

    # init model, loss, optimizer, scheduler
    model     = ResMLP2(in_dim=X_train.shape[1], out_dim=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-3, epochs=epochs,
        steps_per_epoch=len(train_loader), pct_start=0.3
    )

    best_acc, patience = 0.0, 8
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            scheduler.step()

        # validation step
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.extend(model(xb).argmax(dim=1).numpy())
                trues.extend(yb.numpy())
        acc = accuracy_score(trues, preds)
        print(f"Epoch {epoch:02d} | Val Acc: {acc:.3f}")

        if acc > best_acc + 1e-4:
            best_acc, best_state = acc, model.state_dict()
            patience = 8
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    return model, scaler
