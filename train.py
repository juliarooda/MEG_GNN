import torch
import os
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from ray import train, tune
import tempfile
from model import GNN

def train_func(config, dataset, dataset_train, y_train):    
    # Retrieve model
    model = GNN(
        hidden_channels=config['hidden_channels'], 
        dataset=dataset
    )

    # Split train data in train and validation set 
    subset_train, subset_val, y_subset_train, y_subset_val = train_test_split(
        dataset_train, 
        y_train, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train
    )

    # Batch train and validation set 
    train_loader = DataLoader(
        subset_train, 
        config['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        subset_val, 
        config['batch_size'],
        shuffle=True
    )

    # Define optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            model.load_state_dict(checkpoint_dict["model_state"])

    num_epochs = 100
    # Loop over the dataset multiple times
    for epoch in range(num_epochs):
        # Put model in training mode
        model.train()
        
        train_loss = 0
        num_batches = 0
        # Iterate over each batch in train set
        for data in train_loader:
            # Pass batch through model to obtain prediction (logit)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch) 

            # Measure discrepancy between prediction (out) and label (data.y)
            loss = criterion(out, data.y)

            # Compute gradient of loss 
            loss.backward()

            # Update the model's parameters using the computed gradients
            optimizer.step()

            # Clear all gradients before the next iteration
            optimizer.zero_grad()

            # Add to loss 
            train_loss += loss.item()
            num_batches += 1

        model.eval()

        correct_val = 0
        val_loss = 0
        for data in val_loader:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)
            pred = out.argmax(dim=1)
            val_loss += loss.item()
            correct_val += int((pred == data.y).sum())

        val_acc = correct_val/len(val_loader.dataset)

        correct_train = 0
        for data in train_loader:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = out.argmax(dim=1)
            correct_train += int((pred == data.y).sum())

        train_acc = correct_train/len(train_loader.dataset)

        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                os.path.join(tempdir, "checkpoint.pt"),
            )

            train.report({"train_accuracy": train_acc, "val_accuracy": val_acc, "train_loss": train_loss, "val_loss": val_loss}, checkpoint=train.Checkpoint.from_directory(tempdir))

    
    # Save trained model
    # torch.save(model.state_dict(), 'best_model.pth')

def test_best_model(best_result, dataset, dataset_test):
    # Retrieve best model
    best_trained_model = GNN(hidden_channels=best_result.config['hidden_channels'], dataset=dataset)
    # best_trained_model.load_state_dict(torch.load('best_model.pth'))
    with best_result.checkpoint.as_directory() as checkpoint_dir:
        state_dict = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'))
        best_trained_model.load_state_dict(state_dict['model_state'])

    # Batch test dataset with best batch size
    test_loader = DataLoader(
        dataset_test, 
        best_result.config['batch_size'],
        shuffle=False
    )
    
    best_trained_model.eval()
    correct = 0
    for data in test_loader:
        out = best_trained_model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    print(f'accuracy: {correct/len(test_loader.dataset)}')
