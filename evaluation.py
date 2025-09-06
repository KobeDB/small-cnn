import torch

from main import (
    SmallCNN,
    MODEL_SAVE_PATH,
    evaluate_small_cnn,
    test_loader,
)

if __name__ == "__main__":
    cnn = SmallCNN(10)
    cnn.load_state_dict(torch.load(MODEL_SAVE_PATH))
    acc = evaluate_small_cnn(cnn, test_loader)
    print(f"Test accuracy: {acc}")
