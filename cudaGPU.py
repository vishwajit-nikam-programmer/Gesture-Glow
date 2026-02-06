import tensorflow as tf

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Device:", tf.config.list_physical_devices('GPU'))

import tensorflow as tf
print(tf.__version__)
