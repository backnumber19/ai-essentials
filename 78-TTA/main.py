import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image


class TTASegmentation:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.eval()

        self.augmentations = [
            {"name": "original", "transform": lambda x: x, "reverse": lambda x: x},
            {
                "name": "hflip",
                "transform": lambda x: torch.flip(x, dims=[2]),
                "reverse": lambda x: torch.flip(x, dims=[2]),
            },
            {
                "name": "rot90",
                "transform": lambda x: torch.rot90(x, k=1, dims=[1, 2]),
                "reverse": lambda x: torch.rot90(x, k=-1, dims=[1, 2]),
            },
            {
                "name": "rot180",
                "transform": lambda x: torch.rot90(x, k=2, dims=[1, 2]),
                "reverse": lambda x: torch.rot90(x, k=-2, dims=[1, 2]),
            },
        ]

    def predict(self, image_tensor):
        with torch.no_grad():
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)

            image_tensor = image_tensor.to(self.device)
            prediction = self.model(image_tensor)

            if prediction.shape[1] == 1:
                prediction = torch.sigmoid(prediction)
            else:
                prediction = F.softmax(prediction, dim=1)

            return prediction.cpu()

    def predict_tta(self, image):
        to_tensor = T.ToTensor()
        image_tensor = to_tensor(image)

        predictions = []

        for aug in self.augmentations:
            augmented = aug["transform"](image_tensor)
            pred = self.predict(augmented)
            pred_reversed = aug["reverse"](pred)
            predictions.append(pred_reversed.numpy())

        ensemble_pred = np.mean(predictions, axis=0)
        final_pred = (ensemble_pred > 0.5).astype(np.uint8)

        return final_pred[0]


if __name__ == "__main__":
    # model = YourSegmentationModel()
    # model.load_state_dict(torch.load('model.pth'))

    # Placeholder for a model, replace with your actual model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x)

    model = DummyModel()

    tta = TTASegmentation(
        model, device="cpu"
    )  # Changed to "cpu" for broader compatibility
    image = Image.open(
        "test_image.jpg"
    )  # Ensure 'test_image.jpg' exists or replace with a valid path
    prediction = tta.predict_tta(image)

    print(f"Prediction shape: {prediction.shape}")
