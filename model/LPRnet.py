import tensorflow as tf

TRAIN_EPOCHS = 300
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 2000
LEARNING_RATE_DECAY_FACTOR = 0.9
MOMENTUM = 0.9

SAVE_STEPS = 1000
VALIDATE_EPOCHS = 10

BATCH_SIZE = 64
BATCH_PER_EPOCH = 50

train_dir = 'train'
val_dir = 'valid'
test_dir = 'test'

IMG_SIZE = [128, 32]#[94, 24]##
CH_NUM = 3

CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789" # exclude I, O
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
DECODE_DICT = {i:char for i, char in enumerate(CHARS)}

NUM_CLASS = len(CHARS)+1

def small_basic_block(inputdata, out_channel, name=None):
    with tf.variable_scope(name):
        out_div4 = int(out_channel/4)
        conv1 = conv2d(inputdata, out_div4, ksize=[1,1], name='conv1')
        relu1 = tf.nn.relu(conv1)

        conv2 = conv2d(relu1, out_div4, ksize=[3,1], name='conv2')
        relu2 = tf.nn.relu(conv2)

        conv3 = conv2d(relu2, out_div4, ksize=[1,3], name='conv3')
        relu3 = tf.nn.relu(conv3)

        conv4 = conv2d(relu3, out_channel, ksize=[1,1], name='conv4')
        bn = tf.layers.batch_normalization(conv4)
        relu = tf.nn.relu(bn)
    return relu

def conv2d(inputdata, out_channel,ksize,stride=[1,1,1,1],pad = 'SAME', name=None):

    with tf.variable_scope(name):
        in_channel = inputdata.get_shape().as_list()[3]
        filter_shape = [ksize[0], ksize[1], in_channel, out_channel]
        weights = tf.get_variable('w', filter_shape, dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
        biases = tf.get_variable('b', [out_channel], dtype=tf.float32, initializer=tf.constant_initializer())
        conv = tf.nn.conv2d(inputdata, weights,
                            strides=stride,
                            padding=pad)
        add_bias = tf.nn.bias_add(conv, biases)
    return add_bias

def global_context(inputdata, ksize, strides):
    avg_pool = tf.nn.avg_pool(inputdata,
                       ksize=ksize,
                       strides=strides,
                       padding='SAME')
    sqm = tf.reduce_mean(tf.square(avg_pool))
    out = tf.div(avg_pool, sqm)
    return out


class LPRnet:

    def __init__(self, is_train):

        self.inputs = tf.placeholder(
            tf.float32,
            shape=(None, IMG_SIZE[0], IMG_SIZE[1], CH_NUM))

        self.targets = tf.sparse_placeholder(tf.int32)

        logits = self.cnn_layers(self.inputs, is_train)

        #print(logits.get_shape().as_list())

        logits_shape = tf.shape(logits)
        cur_batch_size = logits_shape[0]
        timesteps = logits_shape[1]

        seq_len = tf.fill([cur_batch_size], timesteps)

        logits = tf.transpose(logits, (1, 0, 2))
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

        self.dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1, name='decoded')

        self.edit_dis = tf.reduce_sum(tf.edit_distance(tf.cast(decoded[0], tf.int32), \
                                                self.targets, normalize=False))

        ctc_loss = tf.nn.ctc_loss(labels=self.targets, inputs=logits, sequence_length=seq_len)
        self.loss = tf.reduce_mean(ctc_loss)

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                                   global_step,
                                                   DECAY_STEPS,
                                                   LEARNING_RATE_DECAY_FACTOR,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\
            .minimize(ctc_loss, global_step=global_step)

        self.logits = logits
        self.global_step = global_step
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.init = tf.global_variables_initializer()

    def cnn_layers(self, inputs, is_train):

        ## back-bone
        conv1 = conv2d(inputs, 64, ksize=[3,3], name='conv1')
        conv1_bn = tf.layers.batch_normalization(conv1)
        conv1_relu = tf.nn.relu(conv1_bn)
        max1 = tf.nn.max_pool(conv1_relu,
                              ksize=[1, 3, 3, 1],
                              strides=[1, 1, 1, 1],
                              padding='SAME')
        sbb1 = small_basic_block(max1, 128, name='sbb1') # 128
        max2 = tf.nn.max_pool(sbb1,
                              ksize=[1, 3, 3, 1],
                              strides=[1, 2, 1, 1],
                              padding='SAME')

        sbb2 = small_basic_block(max2, 256, name='sbb2')
        sbb3 = small_basic_block(sbb2, 256, name='sbb3')
        max3 = tf.nn.max_pool(sbb3,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 1, 1],
                           padding='SAME')

        dropout1 = tf.layers.dropout(max3, training=is_train)

        conv2 = conv2d(dropout1, 256, ksize=[4, 1], name='conv_d1')
        conv2_bn = tf.layers.batch_normalization(conv2)
        conv2_relu = tf.nn.relu(conv2_bn)

        dropout2 = tf.layers.dropout(conv2_relu, training=is_train)

        conv3 = conv2d(dropout2, NUM_CLASS, ksize=[1, 13], name='conv_d2')
        conv3_bn = tf.layers.batch_normalization(conv3)
        conv3_relu = tf.nn.relu(conv3_bn)

        ## global context
        scale1 = global_context(conv1,
                             ksize=[1, 4, 1, 1],
                             strides=[1, 4, 1, 1]
                             )

        scale2 = global_context(sbb1,
                             ksize=[1, 4, 1, 1],
                             strides=[1, 4, 1, 1]
                             )

        scale3 = global_context(sbb3,
                             ksize=[1, 2, 1, 1],
                             strides=[1, 2, 1, 1]
                             )

        sqm = tf.reduce_mean(tf.square(conv3_relu))
        scale4 = tf.div(conv3_relu, sqm)

        #print(scale1.get_shape().as_list())
        #print(scale2.get_shape().as_list())
        #print(scale3.get_shape().as_list())
        #print(scale4.get_shape().as_list())

        gc_concat = tf.concat([scale1, scale2, scale3, scale4], 3)
        conv_out = conv2d(gc_concat, NUM_CLASS, ksize=(1, 1), name='conv_out')

        logits = tf.reduce_mean(conv_out, axis=2)

        return logits
