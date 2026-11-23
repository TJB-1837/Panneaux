import tensorflow as tf

def main():
    print("Version de TensorFlow :", tf.__version__)
    print("GPU disponible :", tf.config.list_physical_devices('GPU'))

    print("Hello from rps-project!")


if __name__ == "__main__":
    main()
