"""Minimal usage example for the CSV-based VLA_VRW benchmark split."""

from VLA_VRW_CSV_Dataset import load_benchmark_split


if __name__ == "__main__":
    train, test = load_benchmark_split(
        r"D:\FYP\VLA_VRW_CSV",
        channels=("Fz_Pz_uv",),
    )
    train_images, train_labels, train_perclos, train_metadata = train
    test_images, test_labels, test_perclos, test_metadata = test
    print("Training condition: lab")
    print("  images:", train_images.shape)
    print("  labels:", train_labels.shape)
    print("  PERCLOS:", train_perclos.shape)
    print("Evaluation condition: real")
    print("  images:", test_images.shape)
    print("  labels:", test_labels.shape)
    print("  PERCLOS:", test_perclos.shape)
