import torch

class SpacepointClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=3, dropout=0.1):
        super(SpacepointClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Feature extraction layers
        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
        
        self.feature_extractor = torch.nn.Sequential(*layers)
        
        # Classification head
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, event_indices=None):
        """
        Forward pass of the model.
        
        Args:
            x: Spacepoint coordinates [batch_size, input_dim]
            event_indices: Event indices for each spacepoint [batch_size]
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # If event indices are provided, we could add event-aware processing here
        # For now, we'll use a simple approach, but this could be extended with:
        # - Event-level batch normalization
        # - Attention mechanisms within events
        # - Graph neural networks connecting spacepoints within events
        
        # Classification
        logits = self.classifier(features)
        
        return logits
