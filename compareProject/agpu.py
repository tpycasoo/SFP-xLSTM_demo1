import tensorflow as tf

# 旧的方式（已弃用）
# if tf.test.is_gpu_available():
#     print("GPU is available")

# 新的方式
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Found {len(physical_devices)} GPU(s)")
else:
    print("No GPU found.")
