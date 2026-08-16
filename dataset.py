import os
import cv2
import torch
import pandas as pd

from torch.utils.data import Dataset


class DriftSenseDataset(Dataset):
    """
    Dataset loader for Drift-Sense.

    Each sample contains:
        Reference image
        Search image
        Ground-truth target center (x, y)

    Original images:
        1000 x 1000

    Model input:
        256 x 256

    Target:
        normalized (x, y) in [0, 1]
    """

    def __init__(self, root_dir, split="train", image_size=256):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        self.split_dir = os.path.join(root_dir, split)
        self.manifest_path = os.path.join(
            self.split_dir,
            "manifest.csv"
        )

        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        self.df = pd.read_csv(self.manifest_path)

        required_columns = [
            "id",
            "reference_path",
            "search_path",
            "gt_x",
            "gt_y",
        ]

        missing = [
            col for col in required_columns
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_path):
        """
        Load an image from the path stored in the manifest.

        The manifest may contain:
            ./final_dataset/train/reference/00000.png

        or:
            reference/00000.png
        """

        # If manifest contains an absolute path, use it directly.
        if os.path.isabs(image_path):
            path = image_path

        # If manifest path already starts with final_dataset,
        # use it directly relative to the project root.
        elif image_path.startswith("./final_dataset") or image_path.startswith("final_dataset"):
            path = image_path

        # Otherwise, treat it as relative to the split directory.
        else:
            path = os.path.join(self.split_dir, image_path)

        image = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not load image: {path}"
            )

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA
        )

        image = image.astype("float32") / 255.0

        image = image[None, :, :]

        return torch.from_numpy(image)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        reference = self._load_image(
            row["reference_path"]
        )

        search = self._load_image(
            row["search_path"]
        )

        # Ground-truth coordinates are in original
        # 1000 x 1000 search-image coordinates.
        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])

        # Normalize to [0,1]
        target = torch.tensor(
            [
                gt_x / 1000.0,
                gt_y / 1000.0
            ],
            dtype=torch.float32
        )

        # Combine Reference + Search
        #
        # Channel 0 = Reference
        # Channel 1 = Search
        #
        # Result:
        # [2, 256, 256]
        images = torch.cat(
            [reference, search],
            dim=0
        )

        return {
            "image": images,
            "target": target,
            "gt_x": torch.tensor(gt_x),
            "gt_y": torch.tensor(gt_y),
            "id": int(row["id"]),
            "architecture": row["architecture"],
        }


if __name__ == "__main__":

    print("=" * 60)
    print("DRIFT-SENSE DATASET TEST")
    print("=" * 60)

    dataset_root = "./final_dataset"

    for split in ["train", "validation", "test"]:

        dataset = DriftSenseDataset(
            dataset_root,
            split=split,
            image_size=256
        )

        print(f"\n{split.upper()}")
        print("Samples:", len(dataset))

        sample = dataset[0]

        print(
            "Image shape:",
            sample["image"].shape
        )

        print(
            "Target:",
            sample["target"]
        )

        print(
            "GT:",
            sample["gt_x"].item(),
            sample["gt_y"].item()
        )

        print(
            "Architecture:",
            sample["architecture"]
        )