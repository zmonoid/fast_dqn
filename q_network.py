"""
Code for deep Q-learning as described in:

Playing Atari with Deep Reinforcement Learning
NIPS Deep Learning Workshop 2013

and

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015


Author of Lasagne port: Nissan Pow
Modifications: Nathan Sprague
"""

import numpy as np
import mxnet as mx

class DQNInitializer(mx.initializer.Xavier):
    def _init_bias(self, _, arr):
        arr[:] = .1
    def _init_default(self, name, _):
        pass

class DeepQLearner:
    """
    Deep Q-learning network using Lasagne.
    """

    def __init__(self, input_width, input_height, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, input_scale=255.0, ctx=mx.gpu(0), K=4, lam=4):

        self.lam = lam
        self.input_width = input_width
        self.input_height = input_height
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.discount = discount
        self.K = K
        self.rho = rho
        self.lr = learning_rate
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self.rng = rng
        self.update_counter = 0
        self.ctx = ctx
        self.input_scale = input_scale

        input_shape = (batch_size, num_frames, input_width, input_height)
        net = self.build_nature_network(num_actions)

        self.loss_exe = net.simple_bind(ctx=ctx, grad_req='write',data=input_shape)
        self.target_exe = net.simple_bind(ctx=ctx, grad_req='null', data=input_shape)
        self.policy_exe = self.loss_exe.reshape(data=(1, num_frames, input_width, input_height))

        initializer = DQNInitializer(factor_type='in')
        names = net.list_arguments()
        for name in names:
            initializer(name, self.loss_exe.arg_dict[name])

        self.target_exe.copy_params_from(arg_params=self.loss_exe.arg_dict)

        #self.optimizer = mx.optimizer.create(name='rmsprop', learning_rate=self.lr, gamma2=self.momentum, clip_gradient=None)
        self.optimizer = mx.optimizer.create(name='adagrad', learning_rate=0.01, eps=0.01, wd=0.0, clip_gradient=None, rescale_grad=1.0)
        self.updater = mx.optimizer.get_updater(self.optimizer)



    @staticmethod
    def update_weights(executor, updater):
        for ind, k in enumerate(executor.arg_dict):
            if k.endswith('weight') or k.endswith('bias'):
                updater(index=ind, grad=executor.grad_dict[k], weight=executor.arg_dict[k])


    def load_weights(self, params):
        self.policy_exe.copy_params_from(params)
        self.copy_weights(self.policy_exe, self.loss_exe)
        self.copy_weights(self.policy_exe, self.policy_exe)

    @staticmethod
    def build_nature_network(num_actions=20):
        data = mx.sym.Variable("data")
        conv1 = mx.sym.Convolution(data=data, num_filter=32, stride=(4, 4),
                                   kernel=(8, 8), name="conv1")
        relu1 = mx.sym.Activation(data=conv1, act_type='relu', name="relu1")
        conv2 = mx.sym.Convolution(data=relu1, num_filter=64, stride=(2, 2),
                                   kernel=(4, 4), name="conv2")
        relu2 = mx.sym.Activation(data=conv2, act_type='relu', name="relu2")
        conv3 = mx.sym.Convolution(data=relu2, num_filter=64, stride=(1, 1),
                                   kernel=(3, 3), name="conv3")
        relu3 = mx.sym.Activation(data=conv3, act_type='relu', name="relu3")
        fc4 = mx.sym.FullyConnected(data=relu3, name="fc4", num_hidden=512)
        relu4 = mx.sym.Activation(data=fc4, act_type='relu', name="relu4")
        fc5 = mx.sym.FullyConnected(data=relu4, name="fc5", num_hidden=num_actions)
        return fc5


    def train(self, imgs, actions, rewards, terminals):
        """
        Train one batch.

        Arguments:

        imgs - b x (f + 1) x h x w numpy array, where b is batch size,
               f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """
        index = self.K + self.num_frames
        states = imgs[:, index-3:index+1, :, :]
        next_states = imgs[:, index-2:index+2:, :, :]

        st = mx.nd.array(states, ctx=self.ctx) / self.input_scale
        at = mx.nd.array(actions[:, index], ctx=self.ctx)
        rt = mx.nd.array(rewards[:, index], ctx=self.ctx)
        tt = mx.nd.array(terminals[:, index], ctx=self.ctx)
        st1 = mx.nd.array(next_states, ctx=self.ctx) / self.input_scale

        next_q_out = self.target_exe.forward(data=st1)[0]
        target_q_values = rt + mx.nd.choose_element_0index(next_q_out,
                mx.nd.argmax_channel(next_q_out)) * (1.0 - tt) * self.discount

        current_q_out = self.loss_exe.forward(is_train=True, data=st)[0]
        current_q_values = mx.nd.choose_element_0index(current_q_out, at)

        Q_matrix = np.zeros((self.batch_size, self.K*2+3))
        for k in range(0, self.K+1):
            st_ = mx.nd.array(imgs[:, k:k+self.num_frames], ctx=self.ctx) / self.input_scale
            at_ = mx.nd.array(actions[:, k+self.num_frames-1], ctx=self.ctx)
            q_out_ = self.target_exe.forward(data=st_)[0]
            Q = mx.nd.choose_element_0index(q_out_, at_)
            Q_matrix[:, k] = Q.asnumpy()

        for k in range(self.K+2, self.K*2+3):
            st_ = mx.nd.array(imgs[:, k:k+self.num_frames], ctx=self.ctx) / self.input_scale
            q_out_ = self.target_exe.forward(data=st_)[0]
            Q = mx.nd.max_axis(q_out_, axis=1)
            Q_matrix[:, k] = Q.asnumpy()

        Q_matrix[:, self.K+1] = target_q_values.asnumpy()

        L = np.zeros(self.batch_size)
        U = np.zeros(self.batch_size)

        for b in range(self.batch_size):
            index = self.num_frames + self.K

            R = 0
            Ls = [Q_matrix[b, self.K+1]]
            for k in range(1, self.K+1):
                if terminals[b, index+k]:
                    break
                R += self.discount**k * rewards[b, index+k]
                L_ = R + self.discount ** (k+1) * Q_matrix[b, self.K+2+k]
                Ls.append(L_)
            L[b] = max(Ls)

            Us = []
            R = 0
            for k in range(0, self.K+1):
                if terminals[b, index-k-self.num_frames+1]:
                    break
                R -= rewards[b, index-1-k] / self.discount ** (1+k)
                U_ = Q_matrix[b, self.K-k] / self.discount ** (1+k) + R
                Us.append(U_)
            U[b] = min(Us)

        L = mx.nd.array(L, ctx=self.ctx)
        U = mx.nd.array(U, ctx=self.ctx)

        grad_L = -self.lam * mx.nd.maximum(L - current_q_values, 0)
        grad_U = self.lam * mx.nd.maximum(current_q_values - U, 0)







        #diff = mx.nd.clip(current_q_values-target_q_values, -1.0, 1.0)
        diff = mx.nd.clip(current_q_values-target_q_values, -1.0, 1.0)  + grad_U + grad_L
        out_grad = mx.nd.zeros((self.batch_size, self.num_actions), ctx=self.ctx)
        out_grad = mx.nd.fill_element_0index(out_grad, diff, at)

        self.loss_exe.backward(out_grad)
        self.update_weights(self.loss_exe, self.updater)

        if self.freeze_interval > 0 and self.update_counter > 0 and \
                self.update_counter % self.freeze_interval == 0:
            self.target_exe.copy_params_from(arg_params=self.loss_exe.arg_dict)

        self.update_counter += 1
        return mx.nd.sum(mx.nd.abs(diff)).asnumpy()

    def q_vals(self, state):
        st = mx.nd.array([state], ctx=self.ctx) / self.input_scale
        return self.policy_exe.forward(data=st)[0].asnumpy()

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_actions)
        q_vals = self.q_vals(state)
        return np.argmax(q_vals)


def main():
    pass

if __name__ == '__main__':
    main()

