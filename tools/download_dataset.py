from datasets import load_dataset

# Download the dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.en")

# Save all downloaded splits locally
dataset.save_to_disk("./202311101.en")

print(dataset)
