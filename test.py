import tensorflow as tf

kernel_vals = [5, 5, 3, 3, 3]
feature_vals = [1, 32, 64, 128, 128, 256]
stride_vals = pool_vals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]

kernel = tf.random.truncated_normal([kernel_vals[0], kernel_vals[0], feature_vals[0], feature_vals[0 + 1]],
                               stddev=0.1)


print(kernel)