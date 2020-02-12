import tensorflow as tf

def model_v1():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 4, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dense(units = 2, activation = 'softmax', dtype = tf.float32))
    model.build()
    return model

def model_v2():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 4, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 4, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 4, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dense(units = 2, activation = 'softmax', dtype = tf.float32))
    return model

def model_v3():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 4, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 32, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 32, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 32, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 32, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dense(units = 2, activation = 'softmax', dtype = tf.float32))
    return model

def model_v4():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 4, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 4, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 4, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 4, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 4, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dense(units = 2, activation = 'softmax', dtype = tf.float32))
    return model

def model_v5():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 4, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dropout(rate = 0.2))
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 32, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dropout(rate = 0.2))
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 32, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dropout(rate = 0.2))
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 32, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dropout(rate = 0.2))
    model.add(tf.keras.layers.Dense(units = 32, input_dim = 32, activation = 'relu', dtype = tf.float32))
    model.add(tf.keras.layers.Dense(units = 2, activation = 'softmax', dtype = tf.float32))
    return model