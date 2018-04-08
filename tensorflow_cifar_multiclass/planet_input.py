import tensorflow as tf
import os


# data_path = '../train.tf'
height, width = 196, 196
IMAGE_SIZE = 256
NCHANNELS = 4  # RGBIr

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 400
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 40479 - 400  # number of images given minus the number held out.

batch_size = 24

MASTER_DTYPE = tf.float32  # this is the data type that will be used for the output tensors. Labels must


def read_and_decode(filename_queue):
    NLABELS = 17
    SHAPE = [256, 256, 4]
    feature = {'image': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string),
               'h': tf.FixedLenFeature([], tf.int64),
               'w': tf.FixedLenFeature([], tf.int64),
               'd': tf.FixedLenFeature([], tf.int64),
               }
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['image'], tf.uint16)
    label = tf.decode_raw(features['label'], tf.int8)

    # Cast label data into int32

    h = tf.cast(features['h'], tf.int32)
    w = tf.cast(features['w'], tf.int32)
    d = tf.cast(features['d'], tf.int32)
    # Reshape image data into the original shape
    image = tf.reshape(image, SHAPE)
    label = tf.reshape(label, [NLABELS])
    label_float = tf.cast(label, MASTER_DTYPE)
            # loss function requires label tensors and nn output tensors to be the same dtype

    return image, label_float


def distorted_inputs(datadir):
    filenames = os.path.join(datadir, 'train.tf')
    filename_queue = tf.train.string_input_producer([filenames])

    with tf.name_scope('data_augmentation'):
        imgs, labels = read_and_decode(filename_queue)
        reshaped_image = tf.cast(imgs, tf.float32)

        distorted_image = tf.random_crop(reshaped_image, [height, width, NCHANNELS])
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = tf.image.random_flip_up_down(distorted_image)
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)
        float_image = tf.image.per_image_standardization(distorted_image)

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image, labels,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    if not eval_data:
        filenames = os.path.join(data_dir, 'train.tf')
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.tf')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.name_scope('input'):
        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)

        # Read examples from files in the filename queue.
        image, label = read_and_decode(filename_queue)
        reshaped_image = tf.cast(image, tf.float32)

        # Image processing for evaluation.
        # Crop the central [height, width] of the image.
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                               height, width)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(resized_image)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, label_batch
