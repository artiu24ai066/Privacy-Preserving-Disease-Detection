
import torch


def fedavg(client_states, client_sizes):
    """
    Federated Averaging (FedAvg).

    Parameters
    ----------
    client_states : list of dictionaries
        Updated model state_dict from each hospital.

    client_sizes : list of integers
        Number of training images at each hospital.

    Returns
    -------
    global_state : dictionary
        Aggregated global model state_dict.
    """

    # ---------------------------------------------
    # 1. Validate inputs
    # ---------------------------------------------

    if len(client_states) == 0:
        raise ValueError("No client models were provided.")

    if len(client_states) != len(client_sizes):
        raise ValueError(
            "Number of client models and client sizes must match."
        )

    if any(size <= 0 for size in client_sizes):
        raise ValueError(
            "Every client must have a positive sample count."
        )

    total_samples = sum(client_sizes)

    # ---------------------------------------------
    # 2. Calculate aggregation weights
    # ---------------------------------------------

    weights = [
        size / total_samples
        for size in client_sizes
    ]

    print("\nFedAvg aggregation weights:")

    for i, weight in enumerate(weights):
        print(f"Hospital {i + 1}: {weight:.4f}")

    # ---------------------------------------------
    # 3. Verify model structures
    # ---------------------------------------------

    reference_keys = list(client_states[0].keys())

    for client_id, state in enumerate(client_states[1:], start=2):

        if list(state.keys()) != reference_keys:
            raise ValueError(
                f"Hospital {client_id} has a different model structure."
            )

        for key in reference_keys:

            if state[key].shape != client_states[0][key].shape:
                raise ValueError(
                    f"Shape mismatch for model entry: {key}"
                )

    # ---------------------------------------------
    # 4. Aggregate model parameters
    # ---------------------------------------------

    global_state = {}

    for key in reference_keys:

        reference_tensor = client_states[0][key]

        # Floating-point parameters are weighted averaged
        if torch.is_floating_point(reference_tensor):

            averaged_tensor = torch.zeros_like(
                reference_tensor
            )

            for client_state, weight in zip(
                client_states, weights
            ):

                averaged_tensor += (
                    client_state[key].to(
                        device=averaged_tensor.device,
                        dtype=averaged_tensor.dtype
                    ) * weight
                )

            global_state[key] = averaged_tensor

        else:
            # Preserve non-floating buffers from the first client.
            # Our current GroupNorm ResNet18 does not rely on
            # BatchNorm running-statistic buffers.
            global_state[key] = reference_tensor.clone()

    print("\nGlobal model aggregation completed!")

    return global_state


# --------------------------------------------------
# 5. Standalone test
# --------------------------------------------------

if __name__ == "__main__":

    # Three small example client models
    # These are illustrative tensors, not ResNet18 weights.

    client_1 = {
        "weight": torch.tensor([1.0, 3.0])
    }

    client_2 = {
        "weight": torch.tensor([5.0, 7.0])
    }

    client_3 = {
        "weight": torch.tensor([9.0, 11.0])
    }

    client_states = [
        client_1,
        client_2,
        client_3
    ]

    # Example client dataset sizes
    client_sizes = [1, 2, 1]

    global_state = fedavg(
        client_states,
        client_sizes
    )

    print("\nAggregated example weights:")
    print(global_state["weight"])

    # Expected weighted average:
    # [(1*1) + (5*2) + (9*1)] / 4 = 5
    # [(3*1) + (7*2) + (11*1)] / 4 = 7

    expected = torch.tensor([5.0, 7.0])

    assert torch.allclose(
        global_state["weight"],
        expected
    )

    print("\nFedAvg test passed!")
