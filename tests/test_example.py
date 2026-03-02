from neurodags import add
from neurodags.datasets import generate_dummy_dataset


def test_add():
    assert add(1, 2) == 3
    assert add(-1.5, 0.5) == -1.0


if __name__ == "__main__":
    # Generate a dummy dataset for testing
    generate_dummy_dataset()
    print("Dummy dataset generated successfully.")

    # Run the test
    test_add()
    print("All tests passed.")
