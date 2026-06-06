# import tensorflow as tf

# print("TensorFlow Version:", tf.__version__)
# print("GPUs:", tf.config.list_physical_devices('GPU'))
# print("Built with CUDA:", tf.test.is_built_with_cuda())
# print("GPU Available:", tf.test.gpu_device_name())

# import tensorflow as tf

# a = tf.random.normal([5000, 5000])
# b = tf.random.normal([5000, 5000])

# c = tf.matmul(a, b)

# print(c.device)

import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.matmul(a, a)

print(b.numpy())