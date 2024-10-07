import json
import matplotlib.pyplot as plt

# Load training history
with open('model4model4_lr.json', 'r') as f:
    history = json.load(f)

# Plotting training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Save the plot as PNG
plt.savefig('model4model4_lracc.png')
plt.show()

