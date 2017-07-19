import tensorlayer as tl
import tensorflow as tf
import numpy as np
from LoadData import LoadData
from tensorlayer.utils import dict_to_one
import time

def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def minibatches(inputs=None, inputs2=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], inputs2[excerpt], targets[excerpt]

def fit(sess, network, train_op, cost, X_train, X_train2, y_train, x, x_2, y_, acc=None, batch_size=100,
        n_epoch=100, print_freq=5, X_val=None, X_val2=None, y_val=None, eval_train=True,
        tensorboard=False, tensorboard_epoch_freq=5, tensorboard_weight_histograms=True, tensorboard_graph_vis=True):
    assert X_train.shape[0] >= batch_size, "Number of training examples should be bigger than the batch size"

    print("Start training the network ...")
    start_time_begin = time.time()
    tensorboard_train_index, tensorboard_val_index = 0, 0
    for epoch in range(n_epoch):
        start_time = time.time()
        loss_ep = 0; n_step = 0
        for X_train_a, X_train_b, y_train_a in minibatches(X_train, X_train2, y_train,
                                                    batch_size, shuffle=True):

            feed_dict = {x: X_train_a, x_2:X_train_b, y_: y_train_a}
            feed_dict.update( network.all_drop )    # enable noise layers



            #you should fill your code
            loss, _ = ......
            # just one line



            loss_ep += loss
            n_step += 1
        loss_ep = loss_ep/ n_step
        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            if (X_val is not None) and (y_val is not None):
                print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                if eval_train is True:
                    train_loss, train_acc, n_batch = 0, 0, 0
                    for X_train_a, X_train_b, y_train_a in minibatches(
                                            X_train, X_train2, y_train, batch_size, shuffle=True):
                        dp_dict = dict_to_one( network.all_drop )    # disable noise layers
                        feed_dict = {x: X_train_a, x_2:X_train_b, y_: y_train_a}
                        feed_dict.update(dp_dict)
                        if acc is not None:



                            #you should fill your code
                            err, ac = ......
                            # just one line



                            train_acc += ac
                        else:
                            err = sess.run(cost, feed_dict=feed_dict)
                        train_loss += err
                        n_batch += 1
                    print("   train loss: %f" % (train_loss/ n_batch))
                    if acc is not None:
                        print("   train acc: %f" % (train_acc/ n_batch))
                val_loss, val_acc, n_batch = 0, 0, 0
                for X_val_a, X_val_b, y_val_a in minibatches(
                                            X_val, X_val2, y_val, batch_size, shuffle=True):
                    dp_dict = dict_to_one( network.all_drop )    # disable noise layers
                    feed_dict = {x: X_val_a, x_2:X_val_b, y_: y_val_a}
                    feed_dict.update(dp_dict)
                    if acc is not None:



                        #you should fill your code
                        err, ac = ......
                        # just one line



                        val_acc += ac
                    else:



                        #you should fill your code
                        err = ......
                        # just one line



                    val_loss += err
                    n_batch += 1
                print("   val loss: %f" % (val_loss/ n_batch))
                if acc is not None:
                    print("   val acc: %f" % (val_acc/ n_batch))
            else:
                print("Epoch %d of %d took %fs, loss %f" % (epoch + 1, n_epoch, time.time() - start_time, loss_ep))
        print("Epoch %d of %d took %fs, loss %f" % (epoch + 1, n_epoch, time.time() - start_time, loss_ep))
    print("Total training time: %fs" % (time.time() - start_time_begin))

if __name__ == "__main__":
    # load data
    data_loader = LoadData(root_path="data/rgbd-dataset_eval")
    all_train_rgb_samples, all_train_depth_samples, all_train_labels, all_test_rgb_samples, all_test_depth_samples, all_test_labels = data_loader.load_data()
    session = get_session()




    # define placeholder 
    x = tf.placeholder(tf.float32, shape=[......], name='x')
    x_depth = tf.placeholder(tf.float32, shape=[......], name='x_depth')
    y_ = tf.placeholder(tf.int64, shape=[......], name='y_')

    # define model(fill your code)

    .............................
    .............................
    .............................






    # define loss
    y = network.outputs
    cost = tl.cost.cross_entropy(y, y_, name="cost")
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    y_op = tf.argmax(tf.nn.softmax(y), 1)

    # define optimizer
    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                                      epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

    # initialize
    tl.layers.initialize_global_variables(session)

    # list model info
    # network.print_params()
    # network.print_layers()

    # train and test model
    fit(session, network, train_op, cost, np.array(all_train_rgb_samples), np.array(all_train_depth_samples), np.array(all_train_labels), x, x_depth, y_,
                 acc=acc, batch_size=100, n_epoch=500, print_freq=1,
                 X_val=np.array(all_test_rgb_samples), X_val2=np.array(all_test_depth_samples), y_val=np.array(all_test_labels), eval_train=True)

    session.close()
