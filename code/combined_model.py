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

"""This file defines the top-level model"""

from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator
from pretty_print import print_example
from modules import RNNEncoder, SimpleSoftmaxLayer, BasicAttn, BiDAF, AnsPtr, SelfAttn, BiDAFOut
from model_super import BaselineModel

logging.basicConfig(level=logging.INFO)



class CompleteModel(BaselineModel):
    """Top-level Question Answering module
        Uses Bidaf, Self attention, and end pointers based on the start pointer
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the QA model.

        Inputs:
          FLAGS: the flags passed in from main.py
          id2word: dictionary mapping word idx (int) to word (string)
          word2id: dictionary mapping word (string) to word idx (int)
          emb_matrix: numpy array shape (400002, embedding_size) containing pre-traing GloVe embeddings
        """
        print "Initializing the Combined Model..."

        super(CompleteModel, self).__init__( *args, **kwargs)



    def build_graph(self):
        """Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.

        Defines:
          self.logits_start, self.logits_end: Both tensors shape (batch_size, context_len).
            These are the logits (i.e. values that are fed into the softmax function) for the start and end distribution.
            Important: these are -large in the pad locations. Necessary for when we feed into the cross entropy function.
          self.probdist_start, self.probdist_end: Both shape (batch_size, context_len). Each row sums to 1.
            These are the result of taking (masked) softmax of logits_start and logits_end.
        """

        # Use a RNN to get hidden states for the context and the question
        # Note: here the RNNEncoder is shared (i.e. the weights are the same)
        # between the context and the question.

        ########################################
        # First bidirection GRU layer
        ########################################

        encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
        context_hiddens = encoder.build_graph(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
        question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)

        ####################
        # Bidaf Attn Layer
        ####################

        # Use context hidden states to attend to question hidden states
        attn_layer = BiDAF(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
        _, attn_output = attn_layer.build_graph(context_hiddens, question_hiddens, self.context_mask, self.qn_mask) # attn_output is shape (batch_size, context_len, hidden_size*2)
        # Concat attn_output to contexxt_hiddens to get blended_reps
        blended_reps = tf.concat([context_hiddens, attn_output], axis=2) # (batch_size, context_len, hidden_size*4)

        ####################
        # Bidaf second bidirection layer
        ####################

        # Bidaf layer after context and question attnetion is calculated. Based off oringinal BiDaf paper

        encoder2 = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
        bidaf_second_layer_hiddens = encoder2.build_graph(blended_reps, self.context_mask, scope_name="BidafEncoder") # (batch_size, question_len, hidden_size*2)

        ####################
        # Self Attn Layer
        ####################

        # if self.self_attn:
        # # Bidaf second attention layer, should eventually use self attention
        #     self_attn_layer = SelfAttn(self.keep_prob, self.FLAGS.hidden_size*2)
        #     _, self_attn_output = self_attn_layer.build_graph(bidaf_second_layer_hiddens, self.context_mask) 
        #     self_attn_reps = tf.concat([bidaf_second_layer_hiddens, self_attn_output], axis=2) 
        # else: 
        #     self_attn_reps = bidaf_second_layer_hiddens

        ####################
        # Bidaf third bidirection layer
        ####################

        # encoder3 = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
        # bidaf_third_layer = encoder3.build_graph(bidaf_second_layer_hiddens, self.context_mask, scope_name="SelfAttnBidaf") # (batch_size, question_len, hidden_size*2)
        
        # final_context_reps = tf.contrib.layers.fully_connected(bidaf_second_layer_hiddens, num_outputs=self.FLAGS.hidden_size) # final_context_reps is shape (batch_size, context_len, hidden_size)

        ####################
        # Attn_Layer
        # ansptr_layer = AnsPtr(self.FLAGS.hidden_size, self.keep_prob)

        # BiDAF Output Layer
        bidaf_out = BiDAFOut(self.FLAGS.hidden_size, self.keep_prob)
        self.logits_start, self.probdist_start, self.logits_end, self.probdist_end = bidaf_out.build_graph(attn_output, bidaf_second_layer_hiddens, self.context_mask)




def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)
