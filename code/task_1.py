import os
import pandas as pd

dataset_dir = '../dataset/unprocessed/multimodal-deep-learning-disaster-response-mouzannar/multimodal'

# Prepare lists to store data
data = {
    'category': [],
    'image_path': [],
    'text': []
}

# Traverse the directory structure
for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)
    if os.path.isdir(category_path):
        # Process images
        images_dir = os.path.join(category_path, 'images')
        texts_dir = os.path.join(category_path, 'text')

        # List images and texts
        images = os.listdir(images_dir)
        texts = os.listdir(texts_dir)

        # Find texts that have corresponding names to images
        for img_name, text_name in zip(images, texts):
            image_path = os.path.join(images_dir, img_name)
            text_path = os.path.join(texts_dir, text_name)

            # Normalize the paths to use forward slashes
            image_path = image_path.replace('\\', '/')
            text_path = text_path.replace('\\', '/')

            with open(text_path, 'r', encoding='utf-8') as file:
                text_content = file.read()

            data['category'].append(category)
            data['image_path'].append(image_path)
            data['text'].append(text_content)


df = pd.DataFrame(data)

# Save to CSV
output_path = '../dataset/processed/multimodal_dataset.csv'

df.to_csv(output_path, index=False, encoding='utf-8')
print(f"Successfully saved to CSV. Find the file at: {os.path.abspath(output_path)}")
