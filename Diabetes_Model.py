import torch
import torch.nn as nn
import torch.nn.functional as F

class DiabetesModel(nn.Module):
    """
    A simple feedforward neural network to predict diabetes.

    Args:
        input_features (int): Number of input features (e.g., calories, exercise, etc.). Default is 6.
        hidden_dim1 (int): Number of neurons in the first hidden layer. Default is 16.
        hidden_dim2 (int): Number of neurons in the second hidden layer. Default is 8.
    """
    def __init__(self, input_features=6, hidden_dim1=32, hidden_dim2=16, hidden_dim3=8 ):
        super(DiabetesModel, self).__init__()

        # --- Layer Definitions ---

        # Input layer to first hidden layer
     
        self.fc1 = nn.Linear(input_features, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, 1)

        # Dropout layer (optional, helps prevent overfitting)
        # Randomly sets a fraction of input units to 0 during training
        self.dropout = nn.Dropout(0.3) # 30% dropout probability

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor containing the features.
                               Shape should be (batch_size, input_features).

        Returns:
            torch.Tensor: The output prediction (probability between 0 and 1).
                          Shape is (batch_size, 1).
        """
        # --- Forward Pass ---

        # Pass input through the first layer and apply ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        # Pass through the output layer
        x = self.fc4(x)

        # Apply Sigmoid activation to get a probability output (0 to 1)
        # Sigmoid is suitable for binary classification tasks.
        output = torch.sigmoid(x)

        return output

if __name__ == "__main__":
    # --- Example Usage ---

    # Create an instance of the model
    # Assumes 6 input features as specified in the prompt
    model = DiabetesModel(input_features=6)

    # Print the model architecture
    print(model)

    # Example input tensor (batch of 2 samples, 6 features each)
    # Replace with your actual data preprocessing and loading
    example_input = torch.randn(2, 6) # (batch_size, num_features)

    # Get predictions
    # Make sure the model is in evaluation mode if not training (e.g., model.eval())
    model.eval() # Set model to evaluation mode (disables dropout etc.)
    with torch.no_grad(): # Disable gradient calculation for inference
        predictions = model(example_input)

    print("\nExample Input:")
    print(example_input)
    print("\nExample Predictions (Probabilities):")
    print(predictions)

    # To get binary predictions (0 or 1), you can threshold the probabilities:
    binary_predictions = (predictions > 0.5).float()
    print("\nExample Binary Predictions (Threshold = 0.5):")
    print(binary_predictions)