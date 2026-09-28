from PIL import Image
import os

# Resize image to 256x256 to reduce processing time
img = Image.open("test_image.jpg")
img_resized = img.resize((256, 256))
img_resized.save("test_image_small.jpg")
img.close()
img_resized.close()
print("Created small image: test_image_small.jpg")
print(f"Original size: {os.path.getsize('test_image.jpg')} bytes")
print(f"Small size: {os.path.getsize('test_image_small.jpg')} bytes")