from src.mlops_project.model import Model
import torch
import pytest

def test_model_output_dim():
    """Test model class"""
    model = Model()
    try: 
        input_ = torch.randint(0, 1000, (1, 10))  # Example integer input for embedding layer
        output = model(input_)
        assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
    except Exception as e:
            # If the model expects raw text and has internal tokenization (unlikely for basic torch models but possible)
            # fallback to checking if it handles the string list directly, though usually tests mock tokenized data.
            pytest.fail(f"Model forward pass failed: {e}")

def test_model_initialization():
    """Test model initialization"""
    try:
        model = Model()
    except Exception as e:
        pytest.fail(f"Model initialization failed: {e}")

def test_model_parameters():
    """Test model parameters"""
    model = Model()
    params = list(model.parameters())
    assert len(params) > 0, "Model has no parameters"

def test_model_forward_pass():
    """Test model forward pass with sample input"""
    model = Model()
    try:
        input_ = torch.randint(0, 1000, (1, 10))  # Example integer input for embedding layer
        output = model(input_)
        assert isinstance(output, torch.Tensor), "Model output is not a tensor"
    except Exception as e:
        pytest.fail(f"Model forward pass failed: {e}")

def test_model_trainable():
    """Test if model parameters are trainable"""
    model = Model()
    for param in model.parameters():
        assert param.requires_grad, "Model parameter is not trainable"
    
@pytest.mark.parametrize("batch_size", [32, 64])
def test_model_batch_size(batch_size):
    """Test model with batch input"""
    model = Model()
    try:
        input_ = torch.randint(0, 1000, (batch_size, 10))  # Batch of samples
        output = model(input_)
        assert output.shape == (batch_size, 2), f"Expected output shape ({batch_size}, 2), got {output.shape}"
    except Exception as e:
        pytest.fail(f"Model forward pass with batch input failed: {e}")


def test_error_on_invalid_input():
    """Test model raises error on invalid input"""
    model = Model()
    with pytest.raises(Exception, match="Input must be a tensor of integers"):
        input_ = "invalid input"
        model(input_)
