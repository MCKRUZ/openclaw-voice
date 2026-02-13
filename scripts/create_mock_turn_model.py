"""Create a mock Smart Turn model for testing.

This creates a simple ONNX model that can be used for testing the turn detector
without downloading the actual Smart Turn v3 model from HuggingFace.
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path


def create_mock_model(output_path: Path):
    """
    Create a mock ONNX model for testing.

    The model takes audio input [1, 128000] and outputs a probability [1, 1].
    For testing, it just returns a random probability.
    """
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        print("ERROR: onnx package not installed")
        print("Install with: pip install onnx")
        return False

    # Define model inputs and outputs
    audio_input = helper.make_tensor_value_info(
        "audio", TensorProto.FLOAT, [1, 128000]
    )
    probability_output = helper.make_tensor_value_info(
        "probability", TensorProto.FLOAT, [1, 1]
    )

    # Create a simple identity node (just passes through scaled input)
    # In reality, this would be a complex neural network
    # For testing, we'll use a Constant node
    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["probability"],
        value=helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.FLOAT,
            dims=[1, 1],
            vals=[0.5],  # Always return 0.5 probability
        ),
    )

    # Create graph
    graph_def = helper.make_graph(
        nodes=[constant_node],
        name="SmartTurnMock",
        inputs=[audio_input],
        outputs=[probability_output],
    )

    # Create model
    model_def = helper.make_model(graph_def, producer_name="mock-smart-turn")
    model_def.opset_import[0].version = 13

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_def, str(output_path))

    print(f"Mock model created at: {output_path}")
    print(f"Model size: {output_path.stat().st_size} bytes")

    return True


if __name__ == "__main__":
    from utils.config import get_models_dir

    models_dir = get_models_dir()
    model_path = models_dir / "smart_turn_v3.onnx"

    print("Creating mock Smart Turn model for testing...")
    print(f"Target path: {model_path}")
    print()

    if create_mock_model(model_path):
        print("\n✓ Mock model created successfully!")
        print("\nNOTE: This is a mock model for testing only.")
        print("For production use, download the real Smart Turn v3 model from:")
        print("https://huggingface.co/pipecat-ai/smart-turn-v3")
    else:
        print("\n✗ Failed to create mock model")
        print("Install onnx package: pip install onnx")
