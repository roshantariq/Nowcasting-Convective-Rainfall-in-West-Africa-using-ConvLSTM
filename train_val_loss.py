import json
import matplotlib.pyplot as plt

# Load training history
with open('model4model4_lr.json', 'r') as f:
    history = json.load(f)

# Plotting training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save the plot as PNG
plt.savefig('model4model4_lrloss.png')
plt.show()

