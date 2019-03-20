import interpolate_spline
import numpy as np
import tensorflow as tf
from tensorflow import keras

def main():

    batchsize = 100
    num_points = 100
    dim = 3

    points_shape = (batchsize, num_points, dim)

    ph_train_points = tf.placeholder(tf.float32, shape=points_shape)
    ph_train_values = tf.placeholder(tf.float32, shape=points_shape)

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=( num_points*2, dim)),
        keras.layers.Dense(num_points*dim, activation=tf.nn.relu),
        keras.layers.Dense(int(num_points*dim//2), activation=tf.nn.relu),
        keras.layers.Dense(int(num_points*dim//2), activation=tf.nn.relu),
        keras.layers.Dense((num_points+dim+1)*dim),
        keras.layers.Reshape((num_points+dim+1, dim))
    ])

    def apply(a,b):
        double_input = tf.concat((a, b), axis=1)

        w_v = model(double_input)
        w = w_v[:, :num_points, :]
        v = w_v[:, num_points:, :]

        return interpolate_spline._apply_interpolation(ph_train_points, a, w, v, 3)

    values = apply(ph_train_points, ph_train_values)
    loss = tf.losses.mean_squared_error(values, ph_train_values)

    loss_id = tf.losses.mean_squared_error(values, ph_train_points)

    optimizer = tf.train.AdamOptimizer().minimize(loss)


    num_epochs = 30000

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter('train', sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)


        for epoch in range(num_epochs):
            v_train_points = np.random.randn(*points_shape)
            v_train_values = v_train_points + np.random.randn(*points_shape)*.1


            v_optimizer, v_loss, v_loss_id = sess.run([ optimizer, loss, loss_id], feed_dict={ph_train_points: v_train_points,
                                   ph_train_values: v_train_values})
            print("Loss {}.\t Loss with original points {}".format(v_loss, v_loss_id))
            # train_writer.add_summary(summary, epoch)


if __name__ == "__main__":
    main()