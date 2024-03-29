# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks, scope_name="RNNEncoder"):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope(scope_name):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

class CNNCharacterEncoder(object):

    def __init__(self, embed_size, filters, kernal_size, keep_prob):
        self.embed_size = embed_size
        self.filters = filters
        self.kernal_size = kernal_size
        self.keep_prob = keep_prob
        self.vocab_size = 176 # 174 + pad + unknown
        ## ADD CNN STUFF ##

    def build_graph(self, inputs, masks):

        with vs.variable_scope("CNNCharacterEncoder", reuse=tf.AUTO_REUSE):

            char_embeddings = tf.get_variable("char_embeddings", [self.vocab_size, self.embed_size])
            embed = tf.nn.embedding_lookup(char_embeddings, inputs)

            X = tf.reshape(embed, [-1, embed.get_shape()[2], embed.get_shape()[3]])

            X = Conv1D(self.filters, self.kernal_size, padding='same', activation=Activation('tanh'))(X)

            X = tf.reduce_max(X, axis=2)

            out = tf.reshape(X, [-1, embed.get_shape()[1], embed.get_shape()[2]])

            ## APPLY DROPOUT?? ##
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist

class AnsPtr(object):

    def __init__(self, hidden_size, keep_prob):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell = DropoutWrapper(self.rnn_cell, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):

        with vs.variable_scope("AnsPtr"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # inputs is (batch_size, input_len, hidden_size)

            # RNN timestep 1: feed context hidden states (inputs)
            # outputs is (batch_size, input_len, hidden_size)
            # stats is (batch_size, hidden_size)
            outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, inputs, input_lens, dtype=tf.float32)

            state = tf.expand_dims(state, axis=1)

            # RNN hidden state attends to inputs to get start distribution
            # attn_logits and attn_dist are (batch_size, 1, input_len)
            # attn_output is (batch_size, 1, hidden_size)
            attn_layer = BasicAttn(self.keep_prob, self.hidden_size*2, self.hidden_size*2)
            attn_logits, attn_dist, attn_output = attn_layer.build_graph(inputs, masks, state)

            start_logits = tf.squeeze(attn_logits)
            start_dist = tf.squeeze(attn_dist)

            # RNN timestep 2: feed attention output
            outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, attn_output, input_lens, dtype=tf.float32)

            state = tf.expand_dims(state, axis=1)

            # new RNN hidden state attends to inputs to get end distribution
            attn_layer = BasicAttn(self.keep_prob, self.hidden_size*2, self.hidden_size*2)
            attn_logits, attn_dist, attn_output = attn_layer.build_graph(inputs, masks, state)

            end_logits = tf.squeeze(attn_logits)
            end_dist = tf.squeeze(attn_dist)

            return start_logits, start_dist, end_logits, end_dist

class BiDAFOut(object):

    def __init__(self, hidden_size, keep_prob):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        # self.rnn_cell = rnn_cell.GRUCell(self.hidden_size)
        # self.rnn_cell = DropoutWrapper(self.rnn_cell, input_keep_prob=self.keep_prob)

    def build_graph(self, G, M, masks):

        with vs.variable_scope("BiDAFOut"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # G: attention output (batch_size, input_len, hidden_size*8)
            # M: modeling output (batch_size, input_len, hidden_size*2)

            w1 = tf.get_variable("w1", shape=(self.hidden_size*10), initializer=tf.contrib.layers.xavier_initializer())
            weighted_mult1 = tf.tensordot(tf.concat([G,M], axis=2), w1, axes=[[2],[0]])
            start_logits, start_dist = masked_softmax(weighted_mult1, masks, 2)

            # M2, _ = tf.nn.dynamic_rnn(self.rnn_cell, attn_output, input_lens, dtype=tf.float32)
            M2 = RNNEncoder(self.hidden_size, self.keep_prob).build_graph(M, masks, scope_name="M2")

            w2 = tf.get_variable("w2", shape=(self.hidden_size*10), initializer=tf.contrib.layers.xavier_initializer())
            weighted_mult2 = tf.tensordot(tf.concat([G,M2], axis=2), w2, axes=[[2],[0]])
            end_logits, end_dist = masked_softmax(weighted_mult2, masks, 2)

            return start_logits, start_dist, end_logits, end_dist

class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            attn_logits, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_logits, attn_dist, output


class SelfAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, hidden_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.hidden_size = hidden_size

    def build_graph(self, values, values_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("SelfAttn"):

            #Create weight matrices W1 and W2
            W1 = tf.get_variable(name="W1", shape=( self.hidden_size, self.hidden_size,))
            V = tf.get_variable(name="V", shape=( self.hidden_size,))

            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, doc_size, hidden_size)

            # Calculate attention distribution
            g = tf.tensordot(values, W1, axes=[[2],[0]]) # (batch_size, doc_size, hidden_size)
            a = tf.tensordot(V, tf.tanh(g), axes=[[0],[2]]) # shape (batch_size, 1, doc_size)
            attn_logits = tf.tensordot(values_t, a, axes=[[2], [0]]) # shape (batch_size, hidden_size, doc_size)



            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            attn_logits, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, doc_size, hidden_size). take softmax over values
            attn_logits = tf.transpose(attn_logits, perm=[0, 2, 1])
            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_logits, values_t)

            # output = tf.transpose(output, perm=[0, 2, 1])

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob) # (batch_size, doc_size, hidden_size)

            return attn_logits, attn_dist, output

class AoA(object):

    def __init__(self, keep_prob, query_vec_size, doc_vec_size, vocab_size):
        self.keep_prob = keep_prob
        self.query_vec_size = query_vec_size
        self.doc_vec_size = doc_vec_size
        self.vocab_size = vocab_size # this is len(self.word2id)

    def build_graph(self, orig_doc, queries, documents, values_mask):

        with vs.variable_scope("AoA"):

            # Calculate attention distribution
            queries = tf.transpose(queries, perm=[0, 2, 1]) # (batch_size, doc_vec_size, num_queries)
            M = tf.matmul(documents, queries) # shape (batch_size, num_docs, num_queries)

            # Column-wise softmax for Document to Query attention
            doc_attn = tf.nn.softmax(M, dim=1) # shape (batch_size, num_docs, num_queries)

            # Row-wise softmax for Query to Document attention
            q_attn = tf.nn.softmax(M, dim=2) # shape (batch_size, num_docs, num_queries)

            # Reduce q_attn by taking column-wise average
            q_attn_reduced = tf.reduce_mean(q_attn, axis=1, keep_dims=True) # shape (batch_size, 1, num_queries)
            q_attn_reduced = tf.transpose(q_attn_reduced, perm=[0, 2, 1]) # shape (batch_size, num_queries, 1)

            # Dot product between doc_attn and q_attn_reduced
            s = tf.matmul(doc_attn, q_attn_reduced) # shape (batch_size, num_docs, 1)
            # s = tf.squeeze(s, 2)

            # s_partitioned = tf.dynamic_partition(s, tf.range(tf.shape(s)[0]), 100)
            # orig_doc_partitioned = tf.dynamic_partition(orig_doc, tf.range(tf.shape(orig_doc)[0]), 100)

            # Sum-attention layer:
            # First, unstack s so that we can look at repeated instances
            # unstacked = zip(tf.unstack(s, 100), tf.unstack(orig_doc, 100))
            # unstacked = zip(s_partitioned, orig_doc_partitioned)


            # Second, we perform a segmented sum
            # unstacked = [ tf.gather(tf.unsorted_segment_sum(attn, ids, self.vocab_size), ids) for attn, ids in unstacked ]

            def sum_attention(elems):
                attns, ids = elems
                return tf.gather(tf.unsorted_segment_sum(attns, ids, self.vocab_size), ids)
            s_summed = tf.map_fn(sum_attention, elems=(s, orig_doc), dtype=tf.float32)

            # Third, we restack back into s
            # s_summed = tf.stack(unstacked)
            # s_summed = tf.dynamic_stitch(range(100), unstacked)

            # Now, reconfigure size for correct return value size

            # Finally, use logits mask to do softmax just on non-padded data
            attn_logits_mask = tf.expand_dims(values_mask, 1)
            _, attn_dist = masked_softmax(s_summed, attn_logits_mask, 2)

            # OUTPUT
            output = tf.matmul(attn_dist, tf.transpose(queries, perm=[0,2,1]))
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


class BiDAF(object):

    def __init__(self, keep_prob, query_vec_size, doc_vec_size):
        self.keep_prob = keep_prob
        self.query_vec_size = query_vec_size
        self.doc_vec_size = doc_vec_size


    def build_graph(self, documents, queries, documents_mask, queries_mask):

        with vs.variable_scope("BiDAF"):


            # Calculate similarity matrix S:

            # Tile queries and documents so that they have shape (batch_size, num_docs*num_queries, doc_vec_size)
            query_shape = queries.get_shape()
            doc_shape = documents.get_shape()

            # Repeat the whole query matrix num_docs times
            # aug_queries = tf.tile(queries, [1, doc_shape[1], 1]) # shape (batch_size, num_docs*num_queries, doc_vec_size)

            # Repeat each row of documents num_queries times
            # aug_docs = tf.reshape(documents, [-1, doc_shape[1]*doc_shape[2], 1]) # shape (batch_size, num_docs*2*hidden, 1)
            # aug_docs = tf.tile(aug_docs, [1, 1, query_shape[1]]) # shape (batch_size, num_docs*2*hidden, num_queries)
            # aug_docs = tf.reshape(tf.transpose(aug_docs, perm=[0, 2, 1]), [-1, doc_shape[1]*query_shape[1], self.doc_vec_size]) # shape (batch_size, num_docs*num_queries, doc_vec_size)

            # Perform element-wise multiplication on augmented data
            # element_mult = tf.multiply(aug_queries, aug_docs) # shape (batch_size, num_docs*num_queries, doc_vec_size)
            element_mult_test = tf.expand_dims(documents, 2) * tf.expand_dims(queries, 1) # shape (batch_size, num_docs, num_queries, doc_vec_size)
            # print element_mult.get_shape(), element_mult_test.get_shape()

            W_sim_mult = tf.get_variable("W_sim_mult", shape=(self.query_vec_size,), initializer=tf.contrib.layers.xavier_initializer())
            weighted_mult = tf.tensordot(element_mult_test, W_sim_mult, axes=[[3],[0]])


            # document_test = tf.tile(tf.expand_dims(documents, 2), [1, 1, query_shape[1], 1])
            W_sim_docs = tf.get_variable("W_sim_docs", shape=(self.doc_vec_size,1), initializer=tf.contrib.layers.xavier_initializer())
            # weighted_docs = tf.tensordot(document_test, W_sim_docs, axes=[[3],[0]])
            weighted_docs = tf.tensordot(documents, W_sim_docs, [[2],[0]])

            # queries_test = tf.tile(tf.expand_dims(queries, 1), [1, doc_shape[1], 1, 1])
            W_sim_queries = tf.get_variable("W_sim_queries", shape=(self.query_vec_size,1), initializer=tf.contrib.layers.xavier_initializer())
            # weighted_queries = tf.tensordot(queries_test, W_sim_queries, [[3],[0]])
            weighted_queries = tf.transpose(tf.tensordot(queries, W_sim_queries, [[2],[0]]), perm=[0,2,1])

            S = weighted_mult + weighted_docs + weighted_queries


            # Concatenate augmented data and element_mult for S computation
            # concat = tf.concat([aug_docs, aug_queries, element_mult], axis=2) # shape (batch_size, num_docs*num_queries, 6*doc_vec_size)

            # Weights for similarity matrix
            # W_sim = tf.get_variable("W_sim", shape=(3*self.query_vec_size,1), initializer=tf.contrib.layers.xavier_initializer())

            # Create S using weights and concat
            # S = tf.tensordot(concat, W_sim, axes=[[2], [0]]) # shape (batch_size, num_docs*num_queries, 1)
            # S = tf.reshape(S, [-1, doc_shape[1], query_shape[1]]) # shape (batch_size, num_docs, num_queries)

            # Compute softmax row-wise for Context to Query
            # C2Q = tf.nn.softmax(S, dim=2) # shape (num_docs, num_queries)
            q_mask = tf.transpose(tf.tile(tf.expand_dims(queries_mask, -1), [1, 1, doc_shape[1]]), perm=[0,2,1]) # shape (batch_size, num_docs, num_queries)
            d_mask = tf.tile(tf.expand_dims(documents_mask, -1), [1, 1, query_shape[1]])
            mask = tf.cast(tf.cast(q_mask, tf.bool) & tf.cast(d_mask, tf.bool), tf.int32) # shape (batch_size, num_docs, num_queries)

            # Create Context to Query attention matrix
            _, C2Q = masked_softmax(S, mask, 2)
            a = tf.matmul(C2Q, queries) # shape (num_docs, doc_vec_size)

            # Take max across rows (i.e. max similarity for a single context word for all query words) and compute beta
            masked_S = tf.multiply(S, tf.cast(mask, tf.float32)) # shape (batch_size, num_docs, num_queries)
            m = tf.reduce_max(masked_S, axis=2, keep_dims=True) # shape (batch_size, num_docs, 1)
            beta = tf.nn.softmax(m, dim=1) # shape (batch_size, num_docs, 1)
            # beta = masked_softmax(m, document_mask, 1) # shape (batch_size, num_docs, 1)
            cprime = tf.matmul(tf.transpose(documents, perm=[0, 2, 1]), beta) # shape (batch_size, doc_vec_size, 1)

            # Compute final output b by concatenation
            doc_a_mult, doc_cp_mult = tf.multiply(documents, a), tf.multiply(documents, tf.transpose(cprime, perm=[0, 2, 1]))
            b = tf.concat([documents, a, doc_a_mult, doc_cp_mult], axis=2) # shape (batch_size, num_docs, 4*doc_vec_size)

            return None, b


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
