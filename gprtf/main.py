import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import sparse_ops

from gprtf.kernel_matrix import ker_matrix, ker_matrix_tf
from gprtf.common import invertA

KER_NUM_PARAMS = {'RQ': 3 ,'RBF': 2, 'PER': 3, 'MAT': 0} 

@ops.RegisterGradient("_MatrixDeterminantGrad")
def _MatrixDeterminantGrad(op, grad) :
    A = op.inputs[0]	
    C = op.outputs[0]
    Ainv = tf.matrix_inverse(A)
    return grad*C*tf.transpose(Ainv)


def train(x, y, ker_name, epoch=200):

    hyp_params  =  tf.Variable(tf.ones([KER_NUM_PARAMS[ker_name]]))
    R =  tf.Variable(tf.ones([]))

    length = x.shape[0]
    x = tf.constant(x)
    y = tf.reshape(tf.constant(y), [-1, 1])
    k = ker_matrix_tf(ker_name, hyp_params, R, x, length)

    loss = 0.5 * (tf.matmul(tf.transpose(y), tf.matmul(tf.matrix_inverse(k), y)) + tf.log(tf.matrix_determinant(k)))
    opt = tf.train.RMSPropOptimizer(learning_rate=0.005, decay=0.9, momentum=0.0).minimize(loss)

    config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in xrange(epoch):
            _, loss_ = sess.run([opt, loss])
        return sess.run([hyp_params, R])


def predict(x, y, ker_name, testx, opt_hyp, R):

    mus = [] ; sigmas = []
    xs = np.concatenate((x, testx), axis=0)
    number_of_points = xs.shape[0]
    n_train = x.shape[0]

    k = ker_matrix(ker_name, opt_hyp, R, xs, xs.shape[0])
    kstar = k[ n_train : , 0:n_train ]
    kappa = k[ n_train : , n_train : ]
    inv_k = invertA(k[0:n_train, 0:n_train])

    k = k[0:n_train, 0:n_train]

    f_mu = np.linalg.multi_dot([kstar, inv_k, y])
    f_var = kappa - np.linalg.multi_dot([kstar, inv_k, kstar.T])

    return f_mu, f_var


