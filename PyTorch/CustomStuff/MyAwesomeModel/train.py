import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from data_loader import get_loader
from model import GPT
from config import Config

# Prepare data
loader, vocab = get_loader(vocab_path=Config.VOCAB_PATH)

# Initialize model
model = GPT().to(Config.device)
optimizer = Adam(model.parameters(), lr=Config.lr)

# Training loop
for epoch in range(1, Config.epochs + 1):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(Config.device)
        y = y.to(Config.device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, Config.vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch}/{Config.epochs}, Loss: {avg_loss:.4f}")

    # Save checkpoint
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(Config.OUTPUT_DIR, f"model_epoch{epoch}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")