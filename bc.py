import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def train(learner, observations, actions, validation_obs, validation_acts, checkpoint_path, num_epochs=100, device=None):
    """Train function for learning a new policy using BC."""
    
    best_loss = float('inf')
    best_model_state = None

    # Initialize loss function and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)

    # Create dataset and dataloader
    dataset = TensorDataset(observations, actions)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Tracking validation losses
    validation_losses = []

    for epoch in tqdm(range(num_epochs)):
        learner.train()  # Set the model to training mode
        epoch_loss = 0

        # Training loop: iterate through batches
        for batch_obs, batch_acts in dataloader:
            batch_obs = batch_obs.float().to(device)  # Move observations to the correct device and ensure float32
            batch_acts = batch_acts.float().to(device)  # Move actions to the correct device and ensure float32

            optimizer.zero_grad()  # Clear gradients
            predicted_actions = learner(batch_obs)  # Forward pass
            loss = loss_fn(predicted_actions, batch_acts)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            epoch_loss += loss.item()

        # Validation loop: evaluate on validation set
        with torch.no_grad():
            learner.eval()  # Set the model to evaluation mode
            validation_obs = validation_obs.float().to(device)  # Move validation data to the correct device and ensure float32
            validation_acts = validation_acts.float().to(device)

            val_predictions = learner(validation_obs)  # Forward pass
            val_loss = loss_fn(val_predictions, validation_acts)  # Calculate validation loss
            validation_losses.append(val_loss.item())  # Store validation loss

            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss.item()}")

        # Saving the model if the validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss.item()
            best_model_state = learner.state_dict()

    # Save the best performing checkpoint
    if checkpoint_path:
        torch.save(best_model_state, checkpoint_path)

    # Plot and save validation loss graph
    if len(validation_losses) > 0:
        plt.plot(np.arange(0, len(validation_losses)), validation_losses)
        plt.title("Validation Loss vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.savefig("BC_validation_loss.png")
        plt.show()
    else:
        print("No validation losses to plot.")

    return learner