import torch
import torch.nn as nn


class PILocalizationNet(nn.Module):
    """
    Lightweight CNN for initial localization.

    Input:
        [B, 2, 256, 256]

        Channel 0 = Reference
        Channel 1 = Search

    Output:
        [B, 2]

        normalized x, y coordinates
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # 256 -> 128
            nn.Conv2d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 128 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 32 -> 16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 16 -> 8
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 2),

            # x,y must remain between 0 and 1
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


if __name__ == "__main__":

    print("=" * 60)
    print("PI LOCALIZATION MODEL TEST")
    print("=" * 60)

    model = PILocalizationNet()

    # Dummy batch
    x = torch.randn(4, 2, 256, 256)

    with torch.no_grad():
        output = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)
    print("Output:")
    print(output)

    print("\nParameters:")
    print(
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # MPS availability
    print("\nApple Silicon GPU:")

    if torch.backends.mps.is_available():
        print("✓ MPS is available")
        print("✓ Training can use the M5 GPU")
    else:
        print("✗ MPS is NOT available")
        print("Training will use CPU")