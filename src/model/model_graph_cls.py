import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy
import math

import dgl

from transformers import ElectraModel, RobertaModel, BertModel, AutoModel, AlbertModel
import transformers
if int(transformers.__version__[0]) <= 3:
    from transformers.modeling_roberta import RobertaPreTrainedModel
    from transformers.modeling_bert import BertPreTrainedModel
    from transformers.modeling_albert import AlbertPreTrainedModel
    from transformers.modeling_electra import ElectraModel, ElectraPreTrainedModel
else:
    from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
    from transformers.models.bert.modeling_bert import BertPreTrainedModel
    from transformers.models.albert.modeling_albert import AlbertPreTrainedModel
    from transformers.models.electra.modeling_electra import ElectraPreTrainedModel

from torch.nn import CrossEntropyLoss

from dgl.nn.pytorch import RelGraphConv
from src.model.utils import *

class Baseline(nn.Module):
    def __init__(self, language_model,
                 config,
                 max_sentence_length=512,
                 device="cpu",
                 d_embed=300,
                 num_relations=63 * 2 + 1,
                 graph_n_bases=-1, graph_dim=16, gcn_dep=0.0, gcn_layers=2, activation=F.relu,
                 coarse_add = False,
                 ):
        super().__init__()
        self.language_model = language_model
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.max_sentence_length = max_sentence_length
        self.device = device
        self.d_embed = d_embed

        # special token <WORD>추가
        self.config.vocab_size = self.config.vocab_size + 1
        self.language_model.resize_token_embeddings(self.config.vocab_size)

        self.reduction = nn.Linear(self.hidden_size, self.d_embed)
        classifier_dropout = (
            config.hidden_dropout_prob if config.hidden_dropout_prob is not None else 0.1
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.d_embed, self.num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            word_idxs=None,
            word_masks=None,
            real_len=None,

            g=None,

            coarse_labels=None,
            labels=None,
    ):
        # ========================================================
        # contextual Encoding
        # ========================================================
        # discriminator_hidden_states: [batch, max_seq_len, hidden_size]
        lm = self.language_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 # token_type_ids=token_type_ids,
                                 # position_ids=position_ids,
                                # output_hidden_states = True
                                 )

        discriminator_hidden_states = lm[0]
        pooler_output = lm[1]
        self.batch_size = discriminator_hidden_states.size(0)

        token_embedding = self.reduction(discriminator_hidden_states)[:, 0, :]

        hidden_states = self.classifier(token_embedding)
        logits = self.dropout(hidden_states)

        outputs = (logits, )

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            print(loss)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits,

class LMForSequenceClassification_softmax(nn.Module):
    def __init__(self, language_model,
                 config,
                 max_sentence_length=512,
                 device="cpu",
                 d_embed=300,
                 num_relations=63 * 2 + 1,
                 graph_n_bases=-1, graph_dim=16, gcn_dep=0.0, gcn_layers=2, activation=F.relu,
                 coarse_add = False,
                 ):
        super().__init__()
        self.language_model = language_model
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.max_sentence_length = max_sentence_length
        self.device = device
        self.d_embed = d_embed

        # special token <WORD>추가
        self.config.vocab_size = self.config.vocab_size + 1
        self.language_model.resize_token_embeddings(self.config.vocab_size)

        self.reduction = nn.Linear(self.hidden_size, self.d_embed)
        # self.bilstm = nn.LSTM(self.hidden_size, self.d_embed // 2, num_layers=1, batch_first=True, bidirectional=True)

        self.coarse_add = coarse_add
        if self.coarse_add:
            self.num_coarse_labels = 3

            # coarse_label prediction
            self.coarse_label_embedding = nn.Embedding(self.num_coarse_labels, self.d_embed, scale_grad_by_freq=True)
            self.coarse_label_attention = MultiHeadAttn(self.d_embed, num_heads=1, dropout=config.hidden_dropout_prob)
            self.coarse_alpha = nn.Parameter(torch.ones(1)).to("cuda")
            # self.coarse_label_reduction = nn.Linear(2*self.d_embed, self.d_embed)

        # label prediction
        self.label_embedding = nn.Embedding(self.num_labels, self.d_embed, scale_grad_by_freq=True)
        self.label_attention = MultiHeadAttn(self.d_embed, num_heads=1, dropout=config.hidden_dropout_prob)

        ## token2word
        self.token2word = Token2Word()

        # relational layer
        self.graph_dim = graph_dim
        self.emb_proj = nn.Linear(self.d_embed, self.graph_dim)
        self.num_relations = num_relations
        self.activation = activation

        def get_gnn_instance(n_layers):
            return RGCN(
                h_dim=self.graph_dim,
                num_relations=self.num_relations,
                num_hidden_layers=n_layers,
                dropout=gcn_dep,
                activation=self.activation,
                num_bases=graph_n_bases,
                eps=self.config.layer_norm_eps,
            )

        self.rgcn = get_gnn_instance(gcn_layers)

        self.graph_lstm = nn.LSTM(self.graph_dim, self.d_embed // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.graph_output_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(self.d_embed+self.hidden_size, self.num_labels)

        self.alpha = nn.Parameter(torch.ones(1)).to("cuda")
        # self.conv = nn.Conv1d(self.num_labels, self.num_labels, 1, stride=2)
        # self.classifier_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            word_idxs=None,
            word_masks=None,
            real_len=None,

            g=None,

            coarse_labels=None,
            labels=None,
    ):
        # ========================================================
        # contextual Encoding
        # ========================================================
        # discriminator_hidden_states: [batch, max_seq_len, hidden_size]
        lm = self.language_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 # token_type_ids=token_type_ids,
                                 # position_ids=position_ids,
                                # output_hidden_states = True
                                 )

        discriminator_hidden_states = lm[0]
        pooler_output = lm[1]
        self.batch_size = discriminator_hidden_states.size(0)

        token_embedding = self.reduction(discriminator_hidden_states)
        # token_embedding, (_, _) = self.bilstm(discriminator_hidden_states)
        cls_output = token_embedding[:, 0, :].unsqueeze(1) # [batch, 1, hidden_size]

        # ========================================================
        # coarse_label_embedding
        # ========================================================

        if self.coarse_add:
            # other_layer_hidden_states = lm[2][6]  # 6th layer hidden states
            # coarse_cls_output = other_layer_hidden_states[:, 0, :].unsqueeze(1)  # [batch, 1, hidden_size]

            # coarse_label_embedding: (self.num_coarse_labels, self.d_embed)
            coarse_label_embedding = self.coarse_label_embedding(
                torch.tensor([[i for i in range(0, self.num_coarse_labels)]] * self.batch_size).to("cuda"))

            word2coarse_label_attention_output = self.coarse_label_attention(cls_output, coarse_label_embedding, coarse_label_embedding, True)
            coarse_label_outputs = torch.bmm(word2coarse_label_attention_output, coarse_label_embedding)
            # [batch, 1, self.d_embed]
            # coarse_label_outputs = torch.cat([cls_output, coarse_label_outputs], dim=-1)
            # coarse_label_outputs = self.coarse_label_reduction(coarse_label_outputs)
            ## [batch, 1, self.d_embed]
            coarse_label_outputs = (1-self.coarse_alpha) * cls_output + self.coarse_alpha * coarse_label_outputs

            # [batch, self.num_coarse_label]
            coarse_logits = word2coarse_label_attention_output.squeeze(1)


        # ========================================================
        # label_embedding
        # ========================================================
        # label_embedding: (self.num_labels, self.d_embed)
        label_embedding = self.label_embedding(
            torch.tensor([[i for i in range(0, self.num_labels)]] * self.batch_size).to("cuda"))

        # [batch, self.num_label]
        label_logits = self.label_attention(coarse_label_outputs if self.coarse_add else cls_output, label_embedding,
                                                                         label_embedding, True).squeeze(1)

        # change token2word
        word_embedding = self.token2word(hidden_states=token_embedding,
                                         word_idxs=word_idxs)

        # ========================================================
        # Relational Label
        # ========================================================
        word_embedding = self.flatten_node_embeddings(word_embedding, word_masks)
        word_embedding = self.activation(self.emb_proj(word_embedding))
        graphs = dgl.batch(g)
        if len(word_embedding) != len(graphs.ndata["id"]):
            print([len(list(filter(lambda x: word_masks[i][x] == 1, range(len(word_masks[i]))))) for i in
                   range(0, len(word_masks))])
            print(len(word_embedding))
            print(graphs.ndata)
        word_embedding = self.rgcn(graphs, word_embedding)
        word_embedding = self.unflatten_node_embeddings(word_embedding, word_masks)

        _, (graph_h, _) = self.graph_lstm(word_embedding)
        rgcn_output = self.graph_output_dropout(torch.cat((graph_h[0], graph_h[1]), dim=-1))
        rgcn_logits = self.classifier(torch.cat((pooler_output, rgcn_output), dim=-1))  # graph_concat

        logits = self.alpha * F.softmax(rgcn_logits, dim=-1) + (1-self.alpha) * F.softmax(label_logits, dim=-1)
        # logits = self.conv(self.classifier_dropout(torch.cat((label_logits.unsqueeze(-1), rgcn_logits.unsqueeze(-1)), dim=-1))).squeeze(-1)

        if self.coarse_add: outputs = (logits, coarse_logits, )
        else: outputs = (logits, )

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if self.coarse_add:
                coarse_loss_fct = CrossEntropyLoss()
                coarse_loss = coarse_loss_fct(coarse_logits.view(-1, self.num_coarse_labels), coarse_labels.view(-1))
                loss = 0.1 * coarse_loss + 0.9 * loss
            print(loss)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits,


    def flatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        mask_bool_list = node_embeddings_mask.clone().detach().type(torch.BoolTensor).tolist()
        # mask_bool_list = torch.tensor(node_embeddings_mask, dtype=torch.bool).tolist()
        return node_embeddings[mask_bool_list, :]
        # return node_embeddings[node_embeddings_mask]

    def unflatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        mask_bool_list = node_embeddings_mask.clone().detach().type(torch.BoolTensor).tolist()
        # mask_bool_list = torch.tensor(node_embeddings_mask, dtype=torch.bool).tolist()
        output_node_embeddings = node_embeddings.new_zeros(
            node_embeddings_mask.shape[0], node_embeddings_mask.shape[1], node_embeddings.shape[-1]
        )
        output_node_embeddings[mask_bool_list, :] = node_embeddings
        # output_node_embeddings[node_embeddings_mask] = node_embeddings
        return output_node_embeddings


class LMForSequenceClassification(nn.Module):
    def __init__(self, language_model,
                 config,
                 max_sentence_length=512,
                 device="cpu",
                 d_embed=300,
                 num_relations=63 * 2 + 1,
                 graph_n_bases=-1, graph_dim=16, gcn_dep=0.0, gcn_layers=2, activation=F.relu,
                 coarse_add = False,
                 ):
        super().__init__()
        self.language_model = language_model
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.max_sentence_length = max_sentence_length
        self.device = device
        self.d_embed = d_embed

        # special token <WORD>추가
        self.config.vocab_size = self.config.vocab_size + 1
        self.language_model.resize_token_embeddings(self.config.vocab_size)

        self.reduction = nn.Linear(self.hidden_size, self.d_embed)
        # self.bilstm = nn.LSTM(self.hidden_size, self.d_embed // 2, num_layers=1, batch_first=True, bidirectional=True)

        self.coarse_add = coarse_add
        if self.coarse_add:
            self.num_coarse_labels = 3

            # coarse_label prediction
            self.coarse_label_embedding = nn.Embedding(self.num_coarse_labels, self.d_embed, scale_grad_by_freq=True)
            self.coarse_label_attention = MultiHeadAttn(self.d_embed, num_heads=1, dropout=config.hidden_dropout_prob)
            self.coarse_alpha = nn.Parameter(torch.ones(1)).to("cuda")
            # self.coarse_label_reduction = nn.Linear(2*self.d_embed, self.d_embed)

        # label prediction
        self.label_embedding = nn.Embedding(self.num_labels, self.d_embed, scale_grad_by_freq=True)
        self.label_attention = MultiHeadAttn(self.d_embed, num_heads=1, dropout=config.hidden_dropout_prob)

        ## token2word
        self.token2word = Token2Word()

        # relational layer
        self.graph_dim = graph_dim
        self.emb_proj = nn.Linear(self.d_embed, self.graph_dim)
        self.num_relations = num_relations
        self.activation = activation

        def get_gnn_instance(n_layers):
            return RGCN(
                h_dim=self.graph_dim,
                num_relations=self.num_relations,
                num_hidden_layers=n_layers,
                dropout=gcn_dep,
                activation=self.activation,
                num_bases=graph_n_bases,
                eps=self.config.layer_norm_eps,
            )

        self.rgcn = get_gnn_instance(gcn_layers)

        self.graph_lstm = nn.LSTM(self.graph_dim, self.d_embed // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.graph_output_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(self.d_embed+self.hidden_size, self.num_labels)

        self.alpha = nn.Parameter(torch.ones(1)).to("cuda")
        # self.conv = nn.Conv1d(self.num_labels, self.num_labels, 1, stride=2)
        # self.classifier_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            word_idxs=None,
            word_masks=None,
            real_len=None,

            g=None,

            coarse_labels=None,
            labels=None,
    ):
        # ========================================================
        # contextual Encoding
        # ========================================================
        # discriminator_hidden_states: [batch, max_seq_len, hidden_size]
        lm = self.language_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 # token_type_ids=token_type_ids,
                                 # position_ids=position_ids,
                                # output_hidden_states = True
                                 )

        discriminator_hidden_states = lm[0]
        pooler_output = lm[1]
        self.batch_size = discriminator_hidden_states.size(0)

        token_embedding = self.reduction(discriminator_hidden_states)
        # token_embedding, (_, _) = self.bilstm(discriminator_hidden_states)
        cls_output = token_embedding[:, 0, :].unsqueeze(1) # [batch, 1, hidden_size]

        # ========================================================
        # coarse_label_embedding
        # ========================================================

        if self.coarse_add:
            # other_layer_hidden_states = lm[2][6]  # 6th layer hidden states
            # coarse_cls_output = other_layer_hidden_states[:, 0, :].unsqueeze(1)  # [batch, 1, hidden_size]

            # coarse_label_embedding: (self.num_coarse_labels, self.d_embed)
            coarse_label_embedding = self.coarse_label_embedding(
                torch.tensor([[i for i in range(0, self.num_coarse_labels)]] * self.batch_size).to("cuda"))

            word2coarse_label_attention_output = self.coarse_label_attention(cls_output, coarse_label_embedding, coarse_label_embedding, True)
            coarse_label_outputs = torch.bmm(word2coarse_label_attention_output, coarse_label_embedding)
            # [batch, 1, self.d_embed]
            # coarse_label_outputs = torch.cat([cls_output, coarse_label_outputs], dim=-1)
            # coarse_label_outputs = self.coarse_label_reduction(coarse_label_outputs)
            ## [batch, 1, self.d_embed]
            coarse_label_outputs = (1-self.coarse_alpha)*cls_output + self.coarse_alpha*coarse_label_outputs

            # [batch, self.num_coarse_label]
            coarse_logits = word2coarse_label_attention_output.squeeze(1)


        # ========================================================
        # label_embedding
        # ========================================================
        # label_embedding: (self.num_labels, self.d_embed)
        label_embedding = self.label_embedding(
            torch.tensor([[i for i in range(0, self.num_labels)]] * self.batch_size).to("cuda"))

        # [batch, self.num_label]
        label_logits = self.label_attention(coarse_label_outputs if self.coarse_add else cls_output, label_embedding,
                                                                         label_embedding, True).squeeze(1)

        # change token2word
        word_embedding = self.token2word(hidden_states=token_embedding,
                                         word_idxs=word_idxs)

        # ========================================================
        # Relational Label
        # ========================================================
        word_embedding = self.flatten_node_embeddings(word_embedding, word_masks)
        word_embedding = self.activation(self.emb_proj(word_embedding))
        graphs = dgl.batch(g)
        if len(word_embedding) != len(graphs.ndata["id"]):
            print([len(list(filter(lambda x: word_masks[i][x] == 1, range(len(word_masks[i]))))) for i in
                   range(0, len(word_masks))])
            print(len(word_embedding))
            print(graphs.ndata)
        word_embedding = self.rgcn(graphs, word_embedding)
        word_embedding = self.unflatten_node_embeddings(word_embedding, word_masks)

        _, (graph_h, _) = self.graph_lstm(word_embedding)
        rgcn_output = self.graph_output_dropout(torch.cat((graph_h[0], graph_h[1]), dim=-1))
        rgcn_logits = self.classifier(torch.cat((pooler_output, rgcn_output), dim=-1))  # graph_concat

        logits = self.alpha * rgcn_logits + (1-self.alpha) * label_logits
        # logits = self.conv(self.classifier_dropout(torch.cat((label_logits.unsqueeze(-1), rgcn_logits.unsqueeze(-1)), dim=-1))).squeeze(-1)

        if self.coarse_add: outputs = (logits, coarse_logits, )
        else: outputs = (logits, )

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if self.coarse_add:
                coarse_loss_fct = CrossEntropyLoss()
                coarse_loss = coarse_loss_fct(coarse_logits.view(-1, self.num_coarse_labels), coarse_labels.view(-1))
                loss = 0.1 * coarse_loss + 0.9 * loss
            print(loss)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits,


    def flatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        mask_bool_list = node_embeddings_mask.clone().detach().type(torch.BoolTensor).tolist()
        # mask_bool_list = torch.tensor(node_embeddings_mask, dtype=torch.bool).tolist()
        return node_embeddings[mask_bool_list, :]
        # return node_embeddings[node_embeddings_mask]

    def unflatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        mask_bool_list = node_embeddings_mask.clone().detach().type(torch.BoolTensor).tolist()
        # mask_bool_list = torch.tensor(node_embeddings_mask, dtype=torch.bool).tolist()
        output_node_embeddings = node_embeddings.new_zeros(
            node_embeddings_mask.shape[0], node_embeddings_mask.shape[1], node_embeddings.shape[-1]
        )
        output_node_embeddings[mask_bool_list, :] = node_embeddings
        # output_node_embeddings[node_embeddings_mask] = node_embeddings
        return output_node_embeddings

class LMForSequenceClassification_only_HierarchicalAttn(nn.Module):
    def __init__(self, language_model,
                 config,
                 max_sentence_length=512,
                 device="cpu",
                 d_embed=300,
                 num_relations=63 * 2 + 1,
                 graph_n_bases=-1, graph_dim=16, gcn_dep=0.0, gcn_layers=2, activation=F.relu,
                 coarse_add = False,
                 ):
        super().__init__()
        self.language_model = language_model
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.max_sentence_length = max_sentence_length
        self.device = device
        self.d_embed = d_embed

        # special token <WORD>추가
        self.config.vocab_size = self.config.vocab_size + 1
        self.language_model.resize_token_embeddings(self.config.vocab_size)

        # self.reduction = nn.Linear(self.hidden_size, self.d_embed)
        self.bilstm = nn.LSTM(self.hidden_size, self.d_embed // 2, num_layers=1, batch_first=True, bidirectional=True)

        # label attention
        self.num_coarse_labels = 3
        self.coarse_label_embedding = nn.Embedding(self.num_coarse_labels, self.d_embed, scale_grad_by_freq=True)
        self.label_embedding = nn.Embedding(self.num_labels, self.d_embed, scale_grad_by_freq=True)
        self.label_attention = MultiHeadAttn(self.d_embed, num_heads=1, dropout=config.hidden_dropout_prob)

        # self.hierarchical_attention = MultiHeadAttn(self.d_embed, num_heads=1, dropout=config.hidden_dropout_prob)

        # matcher
        self.activation = activation
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1 = nn.Linear(self.d_embed, self.d_embed//2)
        self.linear2 = nn.Linear(self.d_embed//2, self.num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            word_idxs=None,
            word_masks=None,
            real_len=None,

            g=None,

            coarse_labels=None,
            labels=None,
    ):
        # ========================================================
        # contextual Encoding
        # ========================================================
        # discriminator_hidden_states: [batch, max_seq_len, hidden_size]
        lm = self.language_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 # token_type_ids=token_type_ids,
                                 # position_ids=position_ids,
                                # output_hidden_states = True
                                 )

        discriminator_hidden_states = lm[0]
        self.batch_size = discriminator_hidden_states.size(0)

        # token_embedding = self.reduction(discriminator_hidden_states)
        token_embedding, (sentence_hidden_states, _) = self.bilstm(discriminator_hidden_states)
        sentence_hidden_state = torch.cat((sentence_hidden_states[0], sentence_hidden_states[1]), dim=-1).unsqueeze(1)


        # ========================================================
        # label_attention
        # ========================================================
        # coarse_label_embedding: (self.num_coarse_labels, self.d_embed)
        coarse_label_embedding = self.coarse_label_embedding(
            torch.tensor([[i for i in range(0, self.num_coarse_labels)]] * self.batch_size).to("cuda"))

        # label_embedding: (self.num_labels, self.d_embed)
        label_embedding = self.label_embedding(
            torch.tensor([[i for i in range(0, self.num_labels)]] * self.batch_size).to("cuda"))

        # [batch, self.num_label, self.num_coarse_labels]
        label2coarse_attention_output = self.label_attention(label_embedding, coarse_label_embedding, coarse_label_embedding, True)
        coarse2label_attention_output = torch.transpose(label2coarse_attention_output, 1, 2)
        label_attention_output = torch.bmm(label2coarse_attention_output, coarse_label_embedding)
        coarse_attention_output = torch.bmm(coarse2label_attention_output, label_embedding)

        # ========================================================
        # hierarchical attention
        # ========================================================
        coarse_logits = torch.bmm(coarse_attention_output, torch.transpose(sentence_hidden_state, 1, 2)).squeeze(-1)
        hierarchical_attention = torch.bmm(torch.bmm(label_attention_output, torch.transpose(sentence_hidden_state, 1, 2)), sentence_hidden_state)
        logits = self.linear2(self.activation(self.dropout(self.linear1(self.activation(hierarchical_attention))))).squeeze(-1)

        outputs = (logits, coarse_logits, )

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            coarse_loss_fct = CrossEntropyLoss()
            coarse_loss = coarse_loss_fct(coarse_logits.view(-1, self.num_coarse_labels), coarse_labels.view(-1))
            loss = 0.1 * coarse_loss + 0.9 * loss

            print(loss)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits,

class LMForSequenceClassification_only_LabelAttn(nn.Module):
    def __init__(self, language_model,
                 config,
                 max_sentence_length=512,
                 device="cpu",
                 d_embed=300,
                 num_relations=63 * 2 + 1,
                 graph_n_bases=-1, graph_dim=16, gcn_dep=0.0, gcn_layers=2, activation=F.relu,
                 coarse_add = False,
                 ):
        super().__init__()
        self.language_model = language_model
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.max_sentence_length = max_sentence_length
        self.device = device
        self.d_embed = d_embed

        # special token <WORD>추가
        self.config.vocab_size = self.config.vocab_size + 1
        self.language_model.resize_token_embeddings(self.config.vocab_size)

        self.reduction = nn.Linear(self.hidden_size, self.d_embed)
        # self.bilstm = nn.LSTM(self.hidden_size, self.d_embed // 2, num_layers=1, batch_first=True, bidirectional=True)

        self.coarse_add = coarse_add
        if self.coarse_add:
            self.num_coarse_labels = 3

            # coarse_label prediction
            self.coarse_label_embedding = nn.Embedding(self.num_coarse_labels, self.d_embed, scale_grad_by_freq=True)
            self.coarse_label_attention_1 = MultiHeadAttn(self.d_embed, num_heads=1, dropout=config.hidden_dropout_prob)
            # self.coarse_label_concat_reduction = nn.Linear(2 * self.d_embed, self.d_embed)
            # self.coarse_label_attention_2 = MultiHeadAttn(self.d_embed, num_heads=1, dropout=0)
            self.coarse_alpha = nn.Parameter(torch.ones(1)).to("cuda")
            # self.coarse_label_reduction = nn.Linear(2*self.d_embed, self.d_embed)

        # label prediction
        self.label_embedding = nn.Embedding(self.num_labels, self.d_embed, scale_grad_by_freq=True)
        self.label_attention_1 = MultiHeadAttn(self.d_embed, num_heads=1, dropout=config.hidden_dropout_prob)
        # self.label_concat_reduction = nn.Linear(2 * self.d_embed, self.d_embed)
        # self.label_attention_2 = MultiHeadAttn(self.d_embed, num_heads=1, dropout=0)



    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            word_idxs=None,
            word_masks=None,
            real_len=None,

            g=None,

            coarse_labels=None,
            labels=None,
    ):
        # ========================================================
        # contextual Encoding
        # ========================================================
        # discriminator_hidden_states: [batch, max_seq_len, hidden_size]
        lm = self.language_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 # token_type_ids=token_type_ids,
                                 # position_ids=position_ids,
                                # output_hidden_states = True
                                 )

        discriminator_hidden_states = lm[0]
        self.batch_size = discriminator_hidden_states.size(0)

        token_embedding = self.reduction(discriminator_hidden_states)
        # token_embedding, (_, _) = self.bilstm(discriminator_hidden_states)
        cls_output = token_embedding[:, 0, :].unsqueeze(1) # [batch, 1, hidden_size]

        # ========================================================
        # coarse_label_embedding
        # ========================================================

        if self.coarse_add:
            # other_layer_hidden_states = lm[2][6]  # 6th layer hidden states
            # coarse_cls_output = other_layer_hidden_states[:, 0, :].unsqueeze(1)  # [batch, 1, hidden_size]

            # coarse_label_embedding: (self.num_coarse_labels, self.d_embed)
            coarse_label_embedding = self.coarse_label_embedding(
                torch.tensor([[i for i in range(0, self.num_coarse_labels)]] * self.batch_size).to("cuda"))

            word2coarse_label_attention_output = self.coarse_label_attention_1(cls_output, coarse_label_embedding, coarse_label_embedding, True)
            coarse_label_outputs = torch.bmm(word2coarse_label_attention_output, coarse_label_embedding)
            ## [batch, 1, self.d_embed]
            # coarse_label_outputs = torch.cat([cls_output, coarse_label_outputs], dim=-1)
            # coarse_label_outputs = self.coarse_label_reduction(coarse_label_outputs)
            # [batch, 1, self.d_embed]
            coarse_label_outputs = (1-self.coarse_alpha)*cls_output + self.coarse_alpha*coarse_label_outputs

            # [batch, self.num_coarse_label]
            coarse_logits = word2coarse_label_attention_output.squeeze(1)


        # ========================================================
        # label_embedding
        # ========================================================
        # label_embedding: (self.num_labels, self.d_embed)
        label_embedding = self.label_embedding(
            torch.tensor([[i for i in range(0, self.num_labels)]] * self.batch_size).to("cuda"))

        # [batch, self.num_label]
        logits = self.label_attention_1(coarse_label_outputs if self.coarse_add else cls_output, label_embedding, label_embedding, True).squeeze(1)

        if self.coarse_add: outputs = (logits, coarse_logits, )
        else: outputs = (logits, )

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if self.coarse_add:
                coarse_loss_fct = CrossEntropyLoss()
                coarse_loss = coarse_loss_fct(coarse_logits.view(-1, self.num_coarse_labels), coarse_labels.view(-1))
                loss = 0.1 * coarse_loss + 0.9 * loss
            print(loss)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits,

class LMForSequenceClassification_without_LabelAttn(nn.Module):
    def __init__(self, language_model,
                 config,
                 max_sentence_length=512,
                 device="cpu",
                 d_embed=300,
                 num_relations=63 * 2 + 1,
                 graph_n_bases=-1, graph_dim=16, gcn_dep=0.0, gcn_layers=2, activation=F.relu,
                 coarse_add = False,
                 ):
        super().__init__()
        self.language_model = language_model
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.max_sentence_length = max_sentence_length
        self.device = device
        self.d_embed = d_embed

        # special token <WORD>추가
        self.config.vocab_size = self.config.vocab_size + 1
        self.language_model.resize_token_embeddings(self.config.vocab_size)

        self.reduction = nn.Linear(self.hidden_size, self.d_embed)
        # self.bilstm = nn.LSTM(self.hidden_size, self.d_embed // 2, num_layers=1, batch_first=True, bidirectional=True)

        ## token2word
        self.token2word = Token2Word()

        # relational layer
        self.graph_dim = graph_dim
        self.emb_proj = nn.Linear(self.d_embed, self.graph_dim)
        self.num_relations = num_relations
        self.activation = activation

        def get_gnn_instance(n_layers):
            return RGCN(
                h_dim=self.graph_dim,
                num_relations=self.num_relations,
                num_hidden_layers=n_layers,
                dropout=gcn_dep,
                activation=self.activation,
                num_bases=graph_n_bases,
                eps=self.config.layer_norm_eps,
            )

        self.rgcn = get_gnn_instance(gcn_layers)

        self.graph_lstm = nn.LSTM(self.graph_dim, self.d_embed // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.graph_output_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(self.d_embed+self.hidden_size, self.num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            word_idxs=None,
            word_masks=None,
            real_len=None,

            g=None,

            coarse_labels=None,
            labels=None,
    ):
        # ========================================================
        # contextual Encoding
        # ========================================================
        # discriminator_hidden_states: [batch, max_seq_len, hidden_size]
        lm = self.language_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 # token_type_ids=token_type_ids,
                                 # position_ids=position_ids,
                                # output_hidden_states = True
                                 )

        discriminator_hidden_states = lm[0]
        pooler_output = lm[1]
        self.batch_size = discriminator_hidden_states.size(0)

        token_embedding = self.reduction(discriminator_hidden_states)
        # token_embedding, (_, _) = self.bilstm(discriminator_hidden_states)

        # change token2word
        word_embedding = self.token2word(hidden_states=token_embedding,
                                         word_idxs=word_idxs)

        # ========================================================
        # Relational Label
        # ========================================================
        word_embedding = self.flatten_node_embeddings(word_embedding, word_masks)
        word_embedding = self.activation(self.emb_proj(word_embedding))
        graphs = dgl.batch(g)
        if len(word_embedding) != len(graphs.ndata["id"]):
            print([len(list(filter(lambda x: word_masks[i][x] == 1, range(len(word_masks[i]))))) for i in
                   range(0, len(word_masks))])
            print(len(word_embedding))
            print(graphs.ndata)
        word_embedding = self.rgcn(graphs, word_embedding)
        word_embedding = self.unflatten_node_embeddings(word_embedding, word_masks)

        _, (graph_h, _) = self.graph_lstm(word_embedding)
        rgcn_output = self.graph_output_dropout(torch.cat((graph_h[0], graph_h[1]), dim=-1))
        logits = self.classifier(torch.cat((pooler_output, rgcn_output), dim=-1))  # graph_concat

        outputs = (logits, )

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            print(loss)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits,


    def flatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        mask_bool_list = node_embeddings_mask.clone().detach().type(torch.BoolTensor).tolist()
        # mask_bool_list = torch.tensor(node_embeddings_mask, dtype=torch.bool).tolist()
        return node_embeddings[mask_bool_list, :]
        # return node_embeddings[node_embeddings_mask]

    def unflatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        mask_bool_list = node_embeddings_mask.clone().detach().type(torch.BoolTensor).tolist()
        # mask_bool_list = torch.tensor(node_embeddings_mask, dtype=torch.bool).tolist()
        output_node_embeddings = node_embeddings.new_zeros(
            node_embeddings_mask.shape[0], node_embeddings_mask.shape[1], node_embeddings.shape[-1]
        )
        output_node_embeddings[mask_bool_list, :] = node_embeddings
        # output_node_embeddings[node_embeddings_mask] = node_embeddings
        return output_node_embeddings

class LMForSequenceClassification_maxpooling(nn.Module):
    def __init__(self, language_model,
                 config,
                 max_sentence_length=512,
                 device="cpu",
                 d_embed=300,
                 num_relations=63 * 2 + 1,
                 graph_n_bases=-1, graph_dim=16, gcn_dep=0.0, gcn_layers=2, activation=F.relu,
                 coarse_add = False,
                 ):
        super().__init__()
        self.language_model = language_model
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.max_sentence_length = max_sentence_length
        self.device = device
        self.d_embed = d_embed

        # special token <WORD>추가
        self.config.vocab_size = self.config.vocab_size + 1
        self.language_model.resize_token_embeddings(self.config.vocab_size)

        self.reduction = nn.Linear(self.hidden_size, self.d_embed)
        # self.bilstm = nn.LSTM(self.hidden_size, self.d_embed // 2, num_layers=1, batch_first=True, bidirectional=True)

        self.coarse_add = coarse_add
        if self.coarse_add:
            self.num_coarse_labels = 3

            # coarse_label prediction
            self.coarse_label_embedding = nn.Embedding(self.num_coarse_labels, self.d_embed, scale_grad_by_freq=True)
            self.coarse_label_attention = MultiHeadAttn(self.d_embed, num_heads=1, dropout=config.hidden_dropout_prob)
            self.coarse_alpha = nn.Parameter(torch.ones(1)).to("cuda")
            # self.coarse_label_reduction = nn.Linear(2*self.d_embed, self.d_embed)

        # label prediction
        self.label_embedding = nn.Embedding(self.num_labels, self.d_embed, scale_grad_by_freq=True)
        self.label_attention = MultiHeadAttn(self.d_embed, num_heads=1, dropout=config.hidden_dropout_prob)

        ## token2word
        self.token2word = Token2Word()

        # relational layer
        self.graph_dim = graph_dim
        self.emb_proj = nn.Linear(self.d_embed, self.graph_dim)
        self.num_relations = num_relations
        self.activation = activation

        def get_gnn_instance(n_layers):
            return RGCN(
                h_dim=self.graph_dim,
                num_relations=self.num_relations,
                num_hidden_layers=n_layers,
                dropout=gcn_dep,
                activation=self.activation,
                num_bases=graph_n_bases,
                eps=self.config.layer_norm_eps,
            )

        self.rgcn = get_gnn_instance(gcn_layers)
        self.graph_output_proj = nn.Linear(self.graph_dim, self.graph_dim)
        self.graph_output_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.post_combination_layernorm = nn.LayerNorm(self.d_embed+self.hidden_size,
                                                       eps=config.layer_norm_eps)
        self.classifier = nn.Linear(self.d_embed+self.hidden_size, self.num_labels)

        self.alpha = nn.Parameter(torch.ones(1)).to("cuda")

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            word_idxs=None,
            word_masks=None,
            real_len=None,

            g=None,

            coarse_labels=None,
            labels=None,
    ):
        # ========================================================
        # contextual Encoding
        # ========================================================
        # discriminator_hidden_states: [batch, max_seq_len, hidden_size]
        lm = self.language_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 # token_type_ids=token_type_ids,
                                 # position_ids=position_ids,
                                # output_hidden_states = True
                                 )

        discriminator_hidden_states = lm[0]
        pooler_output = lm[1]
        self.batch_size = discriminator_hidden_states.size(0)

        token_embedding = self.reduction(discriminator_hidden_states)
        # token_embedding, (_, _) = self.bilstm(discriminator_hidden_states)
        cls_output = token_embedding[:, 0, :].unsqueeze(1) # [batch, 1, hidden_size]

        # ========================================================
        # coarse_label_embedding
        # ========================================================

        if self.coarse_add:
            # other_layer_hidden_states = lm[2][6]  # 6th layer hidden states
            # coarse_cls_output = other_layer_hidden_states[:, 0, :].unsqueeze(1)  # [batch, 1, hidden_size]

            # coarse_label_embedding: (self.num_coarse_labels, self.d_embed)
            coarse_label_embedding = self.coarse_label_embedding(
                torch.tensor([[i for i in range(0, self.num_coarse_labels)]] * self.batch_size).to("cuda"))

            word2coarse_label_attention_output = self.coarse_label_attention(cls_output, coarse_label_embedding, coarse_label_embedding, True)
            coarse_label_outputs = torch.bmm(word2coarse_label_attention_output, coarse_label_embedding)
            # [batch, 1, self.d_embed]
            # coarse_label_outputs = torch.cat([cls_output, coarse_label_outputs], dim=-1)
            # coarse_label_outputs = self.coarse_label_reduction(coarse_label_outputs)
            ## [batch, 1, self.d_embed]
            coarse_label_outputs = (1-self.coarse_alpha)*cls_output + self.coarse_alpha*coarse_label_outputs

            # [batch, self.num_coarse_label]
            coarse_logits = word2coarse_label_attention_output.squeeze(1)


        # ========================================================
        # label_embedding
        # ========================================================
        # label_embedding: (self.num_labels, self.d_embed)
        label_embedding = self.label_embedding(
            torch.tensor([[i for i in range(0, self.num_labels)]] * self.batch_size).to("cuda"))

        # [batch, self.num_label]
        label_logits = self.label_attention(coarse_label_outputs if self.coarse_add else cls_output, label_embedding,
                                                                         label_embedding, True).squeeze(1)

        # change token2word
        word_embedding = self.token2word(hidden_states=token_embedding,
                                         word_idxs=word_idxs)

        # ========================================================
        # Relational Label
        # ========================================================
        word_embedding = self.flatten_node_embeddings(word_embedding, word_masks)
        word_embedding = self.activation(self.emb_proj(word_embedding))
        graphs = dgl.batch(g)
        if len(word_embedding) != len(graphs.ndata["id"]):
            print([len(list(filter(lambda x: word_masks[i][x] == 1, range(len(word_masks[i]))))) for i in
                   range(0, len(word_masks))])
            print(len(word_embedding))
            print(graphs.ndata)
        word_embedding = self.rgcn(graphs, word_embedding)
        word_embedding = self.unflatten_node_embeddings(word_embedding, word_masks)

        rgcn_output = self.pool_graph(word_embedding, word_masks)
        rgcn_output = self.graph_output_dropout(self.activation(self.graph_output_proj(rgcn_output)))
        rgcn_logits = self.classifier(self.post_combination_layernorm(torch.cat((pooler_output, rgcn_output), dim=-1)))  # graph_concat

        logits = self.alpha * rgcn_logits + (1-self.alpha) * label_logits

        if self.coarse_add: outputs = (logits, coarse_logits, )
        else: outputs = (logits, )

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if self.coarse_add:
                coarse_loss_fct = CrossEntropyLoss()
                coarse_loss = coarse_loss_fct(coarse_logits.view(-1, self.num_coarse_labels), coarse_labels.view(-1))
                loss = 0.1 * coarse_loss + 0.9 * loss
            print(loss)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits,


    def flatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        mask_bool_list = node_embeddings_mask.clone().detach().type(torch.BoolTensor).tolist()
        # mask_bool_list = torch.tensor(node_embeddings_mask, dtype=torch.bool).tolist()
        return node_embeddings[mask_bool_list, :]
        # return node_embeddings[node_embeddings_mask]

    def unflatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        mask_bool_list = node_embeddings_mask.clone().detach().type(torch.BoolTensor).tolist()
        # mask_bool_list = torch.tensor(node_embeddings_mask, dtype=torch.bool).tolist()
        output_node_embeddings = node_embeddings.new_zeros(
            node_embeddings_mask.shape[0], node_embeddings_mask.shape[1], node_embeddings.shape[-1]
        )
        output_node_embeddings[mask_bool_list, :] = node_embeddings
        # output_node_embeddings[node_embeddings_mask] = node_embeddings
        return output_node_embeddings

    def pool_graph(self, node_embs, node_emb_mask):
        """
        Parameters:
            node_embs: (bsz, n_nodes, graph_dim)
            node_emb_mask: (bsz, n_nodes)
        Returns:
            (bsz, graph_dim (*2))
        """
        node_emb_mask = node_emb_mask.unsqueeze(-1)
        output = masked_max(node_embs, node_emb_mask, 1)
        output = torch.where(node_emb_mask.any(1), output, torch.zeros_like(output))
        return output

class LMForSequenceClassification_more_layer(nn.Module):
    def __init__(self, language_model,
                 config,
                 max_sentence_length=512,
                 device="cpu",
                 d_embed=300,
                 num_relations=63 * 2 + 1,
                 graph_n_bases=-1, graph_dim=16, gcn_dep=0.0, gcn_layers=2, activation=F.relu,
                 coarse_add = False,
                 ):
        super().__init__()
        self.language_model = language_model
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.max_sentence_length = max_sentence_length
        self.device = device
        self.d_embed = d_embed

        # special token <WORD>추가
        self.config.vocab_size = self.config.vocab_size + 1
        self.language_model.resize_token_embeddings(self.config.vocab_size)

        self.reduction = nn.Linear(self.hidden_size, self.d_embed)
        # self.bilstm = nn.LSTM(self.hidden_size, self.d_embed // 2, num_layers=1, batch_first=True, bidirectional=True)

        self.coarse_add = coarse_add
        if self.coarse_add:
            self.num_coarse_labels = 3

            # coarse_label prediction
            self.coarse_label_embedding = nn.Embedding(self.num_coarse_labels, self.d_embed, scale_grad_by_freq=True)
            self.coarse_label_attention_1 = MultiHeadAttn(self.d_embed, num_heads=1, dropout=config.hidden_dropout_prob)
            self.coarse_label_attention_2 = MultiHeadAttn(self.d_embed, num_heads=1, dropout=0)
            self.coarse_alpha = nn.Parameter(torch.ones(1)).to("cuda")
            # self.coarse_label_reduction = nn.Linear(2*self.d_embed, self.d_embed)

        # label prediction
        self.label_embedding = nn.Embedding(self.num_labels, self.d_embed, scale_grad_by_freq=True)
        self.label_attention_1 = MultiHeadAttn(self.d_embed, num_heads=1, dropout=config.hidden_dropout_prob)
        self.label_attention_2 = MultiHeadAttn(self.d_embed, num_heads=1, dropout=0)

        ## token2word
        self.token2word = Token2Word()

        # relational layer
        self.graph_dim = graph_dim
        self.emb_proj = nn.Linear(self.d_embed, self.graph_dim)
        self.num_relations = num_relations
        self.activation = activation

        def get_gnn_instance(n_layers):
            return RGCN(
                h_dim=self.graph_dim,
                num_relations=self.num_relations,
                num_hidden_layers=n_layers,
                dropout=gcn_dep,
                activation=self.activation,
                num_bases=graph_n_bases,
                eps=self.config.layer_norm_eps,
            )

        self.rgcn = get_gnn_instance(gcn_layers)

        self.graph_lstm = nn.LSTM(self.graph_dim, self.d_embed // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.graph_output_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(self.d_embed+self.hidden_size, self.num_labels)

        self.alpha = nn.Parameter(torch.ones(1)).to("cuda")
        # self.conv = nn.Conv1d(self.num_labels, self.num_labels, 1, stride=2)
        # self.classifier_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            word_idxs=None,
            word_masks=None,
            real_len=None,

            g=None,

            coarse_labels=None,
            labels=None,
    ):
        # ========================================================
        # contextual Encoding
        # ========================================================
        # discriminator_hidden_states: [batch, max_seq_len, hidden_size]
        lm = self.language_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 # token_type_ids=token_type_ids,
                                 # position_ids=position_ids,
                                # output_hidden_states = True
                                 )

        discriminator_hidden_states = lm[0]
        pooler_output = lm[1]
        self.batch_size = discriminator_hidden_states.size(0)

        token_embedding = self.reduction(discriminator_hidden_states)
        # token_embedding, (_, _) = self.bilstm(discriminator_hidden_states)
        cls_output = token_embedding[:, 0, :].unsqueeze(1) # [batch, 1, hidden_size]

        # ========================================================
        # coarse_label_embedding
        # ========================================================

        if self.coarse_add:
            # other_layer_hidden_states = lm[2][6]  # 6th layer hidden states
            # coarse_cls_output = other_layer_hidden_states[:, 0, :].unsqueeze(1)  # [batch, 1, hidden_size]

            # coarse_label_embedding: (self.num_coarse_labels, self.d_embed)
            coarse_label_embedding = self.coarse_label_embedding(
                torch.tensor([[i for i in range(0, self.num_coarse_labels)]] * self.batch_size).to("cuda"))

            word2coarse_label_attention_output_1 = self.coarse_label_attention_1(cls_output, coarse_label_embedding, coarse_label_embedding, False)
            word2coarse_label_attention_output = self.coarse_label_attention_2(word2coarse_label_attention_output_1, coarse_label_embedding, coarse_label_embedding, True)
            coarse_label_outputs = torch.bmm(word2coarse_label_attention_output, coarse_label_embedding)
            # [batch, 1, self.d_embed]
            # coarse_label_outputs = torch.cat([cls_output, coarse_label_outputs], dim=-1)
            # coarse_label_outputs = self.coarse_label_reduction(coarse_label_outputs)
            ## [batch, 1, self.d_embed]
            coarse_label_outputs = (1-self.coarse_alpha)*cls_output + self.coarse_alpha*coarse_label_outputs

            # [batch, self.num_coarse_label]
            coarse_logits = word2coarse_label_attention_output.squeeze(1)


        # ========================================================
        # label_embedding
        # ========================================================
        # label_embedding: (self.num_labels, self.d_embed)
        label_embedding = self.label_embedding(
            torch.tensor([[i for i in range(0, self.num_labels)]] * self.batch_size).to("cuda"))

        # [batch, self.num_label]
        word2label_attention_output_1= self.label_attention_1(coarse_label_outputs if self.coarse_add else cls_output,
                                                            label_embedding, label_embedding, False)
        label_logits = self.label_attention_2(word2label_attention_output_1, label_embedding, label_embedding, True).squeeze(1)

        # change token2word
        word_embedding = self.token2word(hidden_states=token_embedding,
                                         word_idxs=word_idxs)

        # ========================================================
        # Relational Label
        # ========================================================
        word_embedding = self.flatten_node_embeddings(word_embedding, word_masks)
        word_embedding = self.activation(self.emb_proj(word_embedding))
        graphs = dgl.batch(g)
        if len(word_embedding) != len(graphs.ndata["id"]):
            print([len(list(filter(lambda x: word_masks[i][x] == 1, range(len(word_masks[i]))))) for i in
                   range(0, len(word_masks))])
            print(len(word_embedding))
            print(graphs.ndata)
        word_embedding = self.rgcn(graphs, word_embedding)
        word_embedding = self.unflatten_node_embeddings(word_embedding, word_masks)

        _, (graph_h, _) = self.graph_lstm(word_embedding)
        rgcn_output = self.graph_output_dropout(torch.cat((graph_h[0], graph_h[1]), dim=-1))
        rgcn_logits = self.classifier(torch.cat((pooler_output, rgcn_output), dim=-1))  # graph_concat

        logits = self.alpha * rgcn_logits + (1-self.alpha) * label_logits
        # logits = self.conv(self.classifier_dropout(torch.cat((label_logits.unsqueeze(-1), rgcn_logits.unsqueeze(-1)), dim=-1))).squeeze(-1)

        if self.coarse_add: outputs = (logits, coarse_logits, )
        else: outputs = (logits, )

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if self.coarse_add:
                coarse_loss_fct = CrossEntropyLoss()
                coarse_loss = coarse_loss_fct(coarse_logits.view(-1, self.num_coarse_labels), coarse_labels.view(-1))
                loss = 0.1 * coarse_loss + 0.9 * loss
            print(loss)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits,


    def flatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        mask_bool_list = node_embeddings_mask.clone().detach().type(torch.BoolTensor).tolist()
        # mask_bool_list = torch.tensor(node_embeddings_mask, dtype=torch.bool).tolist()
        return node_embeddings[mask_bool_list, :]
        # return node_embeddings[node_embeddings_mask]

    def unflatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        mask_bool_list = node_embeddings_mask.clone().detach().type(torch.BoolTensor).tolist()
        # mask_bool_list = torch.tensor(node_embeddings_mask, dtype=torch.bool).tolist()
        output_node_embeddings = node_embeddings.new_zeros(
            node_embeddings_mask.shape[0], node_embeddings_mask.shape[1], node_embeddings.shape[-1]
        )
        output_node_embeddings[mask_bool_list, :] = node_embeddings
        # output_node_embeddings[node_embeddings_mask] = node_embeddings
        return output_node_embeddings

class LMForSequenceClassification_multi_loss(nn.Module):
    def __init__(self, language_model,
                 config,
                 max_sentence_length=512,
                 device="cpu",
                 d_embed=300,
                 num_relations=63 * 2 + 1,
                 graph_n_bases=-1, graph_dim=16, gcn_dep=0.0, gcn_layers=2, activation=F.relu,
                 coarse_add=False,
                 ):
        super().__init__()
        self.language_model = language_model
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.max_sentence_length = max_sentence_length
        self.device = device
        self.d_embed = d_embed

        # special token <WORD>추가
        self.config.vocab_size = self.config.vocab_size + 1
        self.language_model.resize_token_embeddings(self.config.vocab_size)

        self.bilstm = nn.LSTM(self.hidden_size, self.d_embed // 2, num_layers=1, batch_first=True, bidirectional=True)

        self.coarse_add = coarse_add
        if self.coarse_add:
            self.num_coarse_labels = 3

            # coarse_label prediction
            self.coarse_label_embedding = nn.Embedding(self.num_coarse_labels, self.d_embed, scale_grad_by_freq=True)
            self.coarse_label_attention = MultiHeadAttn(self.d_embed, num_heads=1, dropout=config.hidden_dropout_prob)
            self.coarse_alpha = nn.Parameter(torch.ones(1)).to("cuda")
            # self.coarse_label_reduction = nn.Linear(2*self.d_embed, self.d_embed)

        # label prediction
        self.label_embedding = nn.Embedding(self.num_labels, self.d_embed, scale_grad_by_freq=True)
        self.label_attention = MultiHeadAttn(self.d_embed, num_heads=1, dropout=config.hidden_dropout_prob)

        ## token2word
        self.token2word = Token2Word()

        # relational layer
        self.graph_dim = graph_dim
        self.emb_proj = nn.Linear(self.d_embed, self.graph_dim)
        self.num_relations = num_relations
        self.activation = activation

        def get_gnn_instance(n_layers):
            return RGCN(
                h_dim=self.graph_dim,
                num_relations=self.num_relations,
                num_hidden_layers=n_layers,
                dropout=gcn_dep,
                activation=self.activation,
                num_bases=graph_n_bases,
                eps=self.config.layer_norm_eps,
            )

        self.rgcn = get_gnn_instance(gcn_layers)

        self.graph_lstm = nn.LSTM(self.graph_dim, self.d_embed // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.graph_output_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier1 = nn.Linear(self.d_embed + self.hidden_size, self.num_labels)
        self.classifier2 = nn.Linear(2 * self.d_embed + self.hidden_size, self.num_labels)

        self.alpha = nn.Parameter(torch.ones(1)).to("cuda")
        # self.conv = nn.Conv1d(self.num_labels, self.num_labels, 1, stride=2)
        # self.classifier_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            word_idxs=None,
            word_masks=None,
            real_len=None,

            g=None,

            coarse_labels=None,
            labels=None,
    ):
        # ========================================================
        # contextual Encoding
        # ========================================================
        # discriminator_hidden_states: [batch, max_seq_len, hidden_size]
        lm = self.language_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 # token_type_ids=token_type_ids,
                                 # position_ids=position_ids,
                                 # output_hidden_states = True
                                 )

        discriminator_hidden_states = lm[0]
        pooler_output = lm[1]
        self.batch_size = discriminator_hidden_states.size(0)

        discriminator_hidden_states, _ = self.bilstm(discriminator_hidden_states)
        cls_output = discriminator_hidden_states[:, 0, :].unsqueeze(1)  # [batch, 1, hidden_size]

        # ========================================================
        # coarse_label_embedding
        # ========================================================

        if self.coarse_add:
            # other_layer_hidden_states = lm[2][6]  # 6th layer hidden states
            # coarse_cls_output = other_layer_hidden_states[:, 0, :].unsqueeze(1)  # [batch, 1, hidden_size]

            # coarse_label_embedding: (self.num_coarse_labels, self.d_embed)
            coarse_label_embedding = self.coarse_label_embedding(
                torch.tensor([[i for i in range(0, self.num_coarse_labels)]] * self.batch_size).to("cuda"))

            word2coarse_label_attention_output = self.coarse_label_attention(cls_output, coarse_label_embedding,
                                                                             coarse_label_embedding, True)
            coarse_label_outputs = torch.bmm(word2coarse_label_attention_output, coarse_label_embedding)
            # [batch, 1, self.d_embed]
            # coarse_label_outputs = torch.cat([cls_output, coarse_label_outputs], dim=-1)
            # coarse_label_outputs = self.coarse_label_reduction(coarse_label_outputs)
            ## [batch, 1, self.d_embed]
            coarse_label_outputs = (1 - self.coarse_alpha) * cls_output + self.coarse_alpha * coarse_label_outputs

            # [batch, self.num_coarse_label]
            coarse_logits = word2coarse_label_attention_output.squeeze(1)

        # ========================================================
        # label_embedding
        # ========================================================
        # label_embedding: (self.num_labels, self.d_embed)
        label_embedding = self.label_embedding(
            torch.tensor([[i for i in range(0, self.num_labels)]] * self.batch_size).to("cuda"))

        # [batch, 1, self.num_label]
        word2label_attention_output = self.label_attention(coarse_label_outputs if self.coarse_add else cls_output,
                                                           label_embedding,
                                                           label_embedding, True)
        # [batch, self.d_embed]
        label_output = torch.bmm(word2label_attention_output, label_embedding).squeeze(1)
        # [batch, self.num_label]
        label_logits = word2label_attention_output.squeeze(1)

        # change token2word
        word_embedding = self.token2word(hidden_states=discriminator_hidden_states,
                                         word_idxs=word_idxs)

        # ========================================================
        # Relational Label
        # ========================================================
        word_embedding = self.flatten_node_embeddings(word_embedding, word_masks)
        word_embedding = self.activation(self.emb_proj(word_embedding))
        graphs = dgl.batch(g)
        if len(word_embedding) != len(graphs.ndata["id"]):
            print([len(list(filter(lambda x: word_masks[i][x] == 1, range(len(word_masks[i]))))) for i in
                   range(0, len(word_masks))])
            print(len(word_embedding))
            print(graphs.ndata)
        word_embedding = self.rgcn(graphs, word_embedding)
        word_embedding = self.unflatten_node_embeddings(word_embedding, word_masks)

        _, (graph_h, _) = self.graph_lstm(word_embedding)
        rgcn_output = self.graph_output_dropout(torch.cat((pooler_output, graph_h[0], graph_h[1]), dim=-1))

        rgcn_logits = self.classifier1(rgcn_output)
        logits = self.classifier2(torch.cat((label_output, rgcn_output), dim=-1))

        if self.coarse_add:
            outputs = (logits, coarse_logits,)
        else:
            outputs = (logits,)

        if labels is not None:
            rgcn_loss_fct = CrossEntropyLoss()
            label_loss_fct = CrossEntropyLoss()
            total_loss_fct = CrossEntropyLoss()

            rgcn_loss = rgcn_loss_fct(rgcn_logits.view(-1, self.num_labels), labels.view(-1))
            label_loss = label_loss_fct(label_logits.view(-1, self.num_labels), labels.view(-1))
            total_loss = total_loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = self.alpha * (total_loss) + (1 - self.alpha) * (rgcn_loss + label_loss)

            if self.coarse_add:
                coarse_loss_fct = CrossEntropyLoss()
                coarse_loss = coarse_loss_fct(coarse_logits.view(-1, self.num_coarse_labels), coarse_labels.view(-1))
                loss = 0.1 * coarse_loss + 0.9 * loss
            print(loss)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits,

    def flatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        mask_bool_list = node_embeddings_mask.clone().detach().type(torch.BoolTensor).tolist()
        # mask_bool_list = torch.tensor(node_embeddings_mask, dtype=torch.bool).tolist()
        return node_embeddings[mask_bool_list, :]
        # return node_embeddings[node_embeddings_mask]

    def unflatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        mask_bool_list = node_embeddings_mask.clone().detach().type(torch.BoolTensor).tolist()
        # mask_bool_list = torch.tensor(node_embeddings_mask, dtype=torch.bool).tolist()
        output_node_embeddings = node_embeddings.new_zeros(
            node_embeddings_mask.shape[0], node_embeddings_mask.shape[1], node_embeddings.shape[-1]
        )
        output_node_embeddings[mask_bool_list, :] = node_embeddings
        # output_node_embeddings[node_embeddings_mask] = node_embeddings
        return output_node_embeddings


class MultiHeadAttn(nn.Module):

    def __init__(self, num_units, num_heads=1, dropout=0, gpu=True, causality=False):
        '''Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        '''
        super(MultiHeadAttn, self).__init__()
        self.gpu = gpu
        self.num_units = num_units
        self.num_heads = num_heads
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, queries, keys, values, last_layer=False):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        # Linear projections
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)

        # get dim to concat
        concat_dim = len(Q.shape) - 1

        if concat_dim == 1:
            Q = Q.unsqueeze(dim=1)
            queries = queries.unsqueeze(dim=1)
            concat_dim = 2

        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=concat_dim), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=concat_dim), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=concat_dim), dim=0)  # (h*N, T_q, C/h)

        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Activation
        if not last_layer:
            outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        query_masks = query_masks.reshape([outputs.shape[0], outputs.shape[1], outputs.shape[2]])

        outputs = outputs * query_masks

        # Dropouts
        outputs = self.dropout(outputs)  # (h*N, T_q, T_k)

        if last_layer:
            return outputs

        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)

        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=concat_dim)  # (N, T_q, C)

        # Residual connection
        # outputs += queries

        return outputs

class RGCN(nn.Module):
    def __init__(self, h_dim, num_relations, num_hidden_layers=1, dropout=0, activation=F.relu, num_bases=-1, eps = 1e-8):
        super().__init__()
        self.h_dim = h_dim
        self.num_relations = num_relations
        self.dropout = dropout
        self.activation = activation
        self.num_bases = None if num_bases < 0 else num_bases
        self.layers = nn.ModuleList([self.create_graph_layer() for _ in range(num_hidden_layers)])
        self.ffn = nn.Linear(self.h_dim, self.h_dim)
        self.norm_layer = nn.LayerNorm(self.h_dim, eps=eps)

    def forward(self, graph, initial_embeddings):

        #if len(initial_embeddings) != len(graph.nodes):
        #    raise ValueError('Node embedding initialization shape mismatch')
        h = initial_embeddings
        for layer in self.layers:
            # h = self.forward_graph_layer(layer, graph, h)
            h = self.ffn(self.forward_graph_layer(layer, graph, h)) + initial_embeddings
            h = self.norm_layer(h)
        return h

    def create_graph_layer(self):
        return RelGraphConv(
            self.h_dim,
            self.h_dim,
            self.num_relations,
            "basis",
            self.num_bases,
            activation=self.activation,
            self_loop=True,
            dropout=self.dropout,
        )

    def forward_graph_layer(self, layer, graph, h):
        return layer(
            graph,
            h,
            graph.edata['type'] if 'type' in graph.edata else h.new_empty(0),
            graph.edata['norm'] if 'norm' in graph.edata else h.new_empty(0),
        )

class Token2Word(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, word_idxs):
        # (batch, max_pre_sen, seq_len) @ (batch, seq_len, hidden) = (batch, max_pre_sen, hidden)
        word_idxs = word_idxs.squeeze(1)

        word_embedding = torch.matmul(word_idxs, hidden_states)

        return word_embedding

