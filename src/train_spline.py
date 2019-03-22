import interpolate_spline
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops


class TrainSpline(object):
    def __init__(self,num_points=19, dim=3):
        self.num_points = num_points
        self.model = keras.models.Sequential([
#            keras.layers.Concatenate(input_shape=((num_points*2, dim), (num_points*2, dim)), axis=1),
            keras.layers.Flatten(input_shape=(num_points*2, dim)),
            keras.layers.Dense(num_points*dim, activation=tf.nn.relu),
            keras.layers.Dense(int(num_points*dim//2), activation=tf.nn.relu),
            keras.layers.Dense(int(num_points*dim//2), activation=tf.nn.relu),
            keras.layers.Dense((num_points+dim+1)*dim),
            keras.layers.Reshape((num_points+dim+1, dim))
        ])

    def __call__(self, train_points,
                       train_values,
                       query_points,
                       order,
                       regularization_weight=0.0,
                       name='interpolate_spline'):
        
        with ops.name_scope(name):
            train_points = ops.convert_to_tensor(train_points)
            train_values = ops.convert_to_tensor(train_values)
            query_points = ops.convert_to_tensor(query_points)

            # First, fit the spline to the observed data.
            with ops.name_scope('solve'):
	        double_input = tf.concat((train_points, train_values), axis=1)

                w_v = self.model(double_input)
                w = w_v[:, :self.num_points, :]
                v = w_v[:, self.num_points:, :]

            with ops.name_scope('predict'):
                query_values = interpolate_spline._apply_interpolation(query_points, train_points, w, v,
                                                      order)
         
	return query_values


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
            v_train_values = v_train_points + 2 + np.random.randn(*points_shape)*.1


            v_optimizer, v_loss, v_loss_id = sess.run([ optimizer, loss, loss_id], feed_dict={ph_train_points: v_train_points,
                                   ph_train_values: v_train_values})
            print("Loss {}.\t Loss with original points {}".format(v_loss, v_loss_id))
            # train_writer.add_summary(summary, epoch)


if __name__ == "__main__":
    main() 
