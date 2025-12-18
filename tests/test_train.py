import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from src.train import train_model


@pytest.mark.parametrize("test_size", [
    0.2,    
    0.3,   
    0.5,  
    0.4,
])

def test_model_accuracy(test_size, random_state=42):
    """Test that model achieves >0.9 accuracy on Iris dataset.
    
    Args:
        test_size: Proportion of the dataset to include in the test split
        random_state: Random state for reproducibility
    """
    # Load dataset
    iris = load_iris()
    
    # Train model
    model, X_test, y_test, y_pred = train_model(
        iris, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Assert accuracy > 0.9
    assert accuracy > 0.95, f"Model accuracy {accuracy:.4f} is not greater than 0.9 (test_size={test_size}, random_state={random_state})"
