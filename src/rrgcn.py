import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch as th

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer, RGAT, UnionRGCNLayer2, UnionRGATLayer, CompGCNLayer, UnionRGATLayer2, CompGCNLayer2
from src.model import BaseRGCN
from src.decoder2 import ConvTransE, ConvTransR ,ConvTransRTime ,ConvTransETime
# from rgcn.time_layer import timeRNN
from rgcn.dsf import DSF_embedding, DSF_embedding2
from rgcn.vocabulary import Vocabulary
from collections import defaultdict
from torch.linalg import eigh
class MLPLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPLinear, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.act = nn.LeakyReLU(0.2)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x):
        x = self.act(F.normalize(self.linear1(x), p=2, dim=1))
        x = self.act(F.normalize(self.linear2(x), p=2, dim=1))

        return x


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == "kbat":
            return UnionRGATLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == "compgcn":
            return CompGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.opn, self.num_bases,
                            activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn" or self.encoder_name == "kbat" or self.encoder_name == "compgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class RGCNCell2(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer2(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == "kbat":
            return UnionRGATLayer2(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                                  activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc,
                                  rel_emb=self.rel_emb)
        elif self.encoder_name == "compgcn":
            return CompGCNLayer2(self.h_dim, self.h_dim, self.num_rels, self.opn, self.num_bases,
                                activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc,
                                rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):

        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.weight_mlp = nn.Linear(hidden_size, hidden_size)
        self.bias_mlp = nn.Linear(hidden_size, hidden_size)
        self.variance_epsilon = eps

    def forward(self, x, weight=None):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        if weight != None:
            return self.weight_mlp(weight) * x + self.bias_mlp(weight)
        return self.weight * x + self.bias

class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0, 
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1,pre_weight=0.7, discount=0, angle=0, use_static=False, pre_type = 'short', 
                 use_cl= False, temperature=0.007, entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, analysis=False, params=None, time_interval=None, args=None, diffu_rec=None, ent_embs=None, rel_embs=None):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.pre_weight = pre_weight
        self.discount = discount
        self.use_static = use_static
        self.pre_type = pre_type
        self.use_cl = use_cl
        self.temp =temperature
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        self.p = params
        self.time_interval = time_interval

        self.w1 = nn.Linear(self.h_dim*2, self.h_dim)
        
        self.w2 = nn.Linear(self.h_dim, self.h_dim)#W
        self.w3 = nn.Linear(self.h_dim, self.h_dim)#
        self.w4 = nn.Linear(self.h_dim*2, self.h_dim)
        self.w5 = nn.Linear(self.h_dim, self.h_dim)#
        self.w6 = nn.Linear(self.h_dim,self.h_dim)#
        self.w7 = nn.Linear(self.h_dim, self.h_dim)#
        self.w_cl = nn.Linear(self.h_dim*2, self.h_dim) #

        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))

        self.weight_1 = nn.Linear(self.h_dim*2, self.h_dim)
        self.weight_2 = nn.Linear(self.h_dim*2, self.h_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        self.weight_3 = nn.Linear(self.h_dim, 1)
        self.weight_4 = nn.Linear(self.h_dim, 1)
        self.bias_r = nn.Parameter(torch.zeros(1))

        self.seq1 = nn.Sequential(
            nn.BatchNorm1d(1024).to(self.gpu),
            nn.Linear(1024, self.h_dim).to(self.gpu),
        )
        self.emb_rel = nn.Parameter(self.seq1(rel_embs.to(self.gpu)).clone().detach().requires_grad_(True)).to(self.gpu)

        self.seq2 = nn.Sequential(
            nn.BatchNorm1d(1024).to(self.gpu),
            nn.Linear(1024, self.h_dim).to(self.gpu),
        )
        self.dynamic_emb = nn.Parameter(self.seq2(ent_embs.to(self.gpu)).clone().detach().requires_grad_(True)).to(self.gpu)

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)
        
        self.his_rgcn_layer = RGCNCell2(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.vocabulary = Vocabulary(num_ents, h_dim, num_rels)
        self.rgat_layer = RGAT(self.h_dim, self.h_dim, activation=F.rrelu, dropout=dropout, self_loop=True)
        self.projection_model = MLPLinear(self.h_dim, self.h_dim)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)   

        self.pre_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.pre_gate_weight, gain=nn.init.calculate_gain('relu'))


        self.entity_cell = nn.GRUCell(self.h_dim, self.h_dim)
        self.relation_cell = nn.GRUCell(self.h_dim, self.h_dim)
        self.time_gate = nn.Linear(self.h_dim,self.h_dim)


        if self.p.use_time_decoder:
            if decoder_name == "convtranse":
                self.decoder_ob = ConvTransETime(num_ents, num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
                self.rdecoder = ConvTransRTime(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            else:
                raise NotImplementedError
        else:
            if decoder_name == "convtranse":
                self.decoder_ob = ConvTransE(num_ents, num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
                self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            else:
                raise NotImplementedError

        self.time_max_len = args.timestamps
        self.max_len = args.diffusion_max_len
        self.time_embeddings = nn.Embedding(self.time_max_len + 1 + 1 + 1,
                                            self.h_dim)  # 1 for padding object and 1 for condition subject and 1 for cls
        self.diffu = diffu_rec
        self.LayerNorm = LayerNorm(self.h_dim, eps=1e-12)
        self.LayerNorm_static = LayerNorm(self.h_dim, eps=1e-12)
        self.embed_dropout = nn.Dropout(args.dropout)
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce2 = nn.CrossEntropyLoss()
        self.emb_dim = self.h_dim
        self.Linear = nn.Linear(3 * self.h_dim, self.h_dim)
        self.Linear2 = nn.Linear(3*self.h_dim, self.h_dim)

        self.time_embs = torch.nn.Parameter(torch.Tensor(args.timestamps, self.h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.time_embs)
        self.time_w = torch.nn.Parameter(torch.Tensor(num_ents), requires_grad=True).float()
        torch.nn.init.normal_(self.time_w)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.temporal_w = torch.nn.Parameter(torch.Tensor(self.h_dim*2, self.h_dim), requires_grad=True).float()
        self.weight4f = 0.2

        self.seq3 = nn.Sequential(
            nn.BatchNorm1d(1024).to(self.gpu),
            nn.Linear(1024, self.h_dim).to(self.gpu),
        )
        self.seq4 = nn.Sequential(
            nn.BatchNorm1d(1024).to(self.gpu),
            nn.Linear(1024, self.h_dim).to(self.gpu),
        )

    def get_composed(self,cur_output, related_emb):
        self.time_weights = []
        for i in range(len(self.inputs)):
            self.time_weights.append(self.time_gate(self.inputs[i]+related_emb))
        self.time_weights.append(torch.zeros(self.num_ents,self.h_dim).cuda())
        self.time_weights = torch.stack(self.time_weights,0)
        self.time_weights = torch.softmax(self.time_weights,0)
        output = cur_output*self.time_weights[-1]
        for i in range(len(self.inputs)):
            output += self.time_weights[i]*self.inputs[i]
        return F.normalize(output)

    def forward(self, sub_graph, T_idx, query_mask, g_list, static_graph, use_cuda, err_mat):
        if self.p.use_cd:
            dsf = DSF_embedding(self.p.sen_len, self.h_dim, self.p.cross_dropout, self.p.sen_dim).cuda()
            dsf_emb = dsf(self.dynamic_emb)
            self.dynamic_emb.data = dsf_emb
            for i in range(len(g_list)):
                if i==0:
                    self.inputs=[F.normalize(self.get_time_emb2(T_idx, i))]
                else:
                    self.inputs.append(F.normalize(self.get_time_emb2(T_idx, i)))
            f_output = F.normalize(self.fourier.forward(self.inputs))
            self.inputs.append(f_output)
            self.inputs.append(self.weight4f * self.inputs[-1] + (1-self.weight4f) * F.normalize(self.get_time_emb2(T_idx, 0)))
            related_emb = torch.spmm(err_mat, self.emb_rel)


        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        self.his_ent, subg_index = self.all_GCN(self.h, sub_graph,use_cuda)  #[7128, 200]
        his_r_emb = F.normalize(self.emb_rel)
        his_att = F.softmax(self.w5(query_mask+self.his_ent),dim=1)
        his_emb = his_att*self.his_ent
        his_emb = F.normalize(his_emb)

        history_embs = []
        att_embs = []
        his_temp_embs = []
        his_rel_embs = []
        if self.pre_type == "all":
            for i, g in enumerate(g_list):
                g = g.to(self.gpu)
                t2 = len(g_list)-i+1
                h_t = torch.sin(self.weight_t2 * t2 + self.bias_t2).repeat(self.num_ents,1)
                self.h =self.w4(torch.concat([self.h,h_t],dim=1))
                temp_e = self.h[g.r_to_e]
                x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
                for span, r_idx in zip(g.r_len, g.uniq_r):
                    x = temp_e[span[0]:span[1],:]
                    x_mean = torch.mean(x, dim=0, keepdim=True)
                    x_input[r_idx] = x_mean

                x_input = self.emb_rel + x_input

                current_h = self.rgcn.forward(g, self.h, [self.emb_rel, self.emb_rel])
                self.inputs.append(self.get_composed(current_h, related_emb))
                current_h = F.normalize(self.inputs[-1]) if self.layer_norm else self.inputs[-1]
                att_e = F.softmax(self.w2(query_mask+current_h),dim=1)

                if i == 0:
                    self.h_0 = self.entity_cell(current_h, self.h)
                    self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0

                else:
                    self.h_0 = self.entity_cell(current_h, self.h_0)
                    self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0

                time_weight = F.sigmoid(torch.mm(x_input, self.time_gate_weight) + self.time_gate_bias)
                self.hr = time_weight * x_input + (1-time_weight) * self.emb_rel
                self.hr = F.normalize(self.hr) if self.layer_norm else self.hr
                history_embs.append(self.h_0)
                his_rel_embs.append(self.hr)
                his_temp_embs.append(self.h_0)
                self.h = self.h_0
                att_emb = att_e*self.h_0
                att_embs.append(att_emb.unsqueeze(0))
            att_ent = torch.mean(torch.concat(att_embs, dim=0), dim=0)
            att_ent = F.normalize(att_ent)
            history_emb = att_ent+history_embs[-1]
            history_emb = F.normalize(history_emb) if self.layer_norm else history_emb
        else:
            self.hr = None
            history_emb = None

        # history_emb = self.inputs[-1]
        # self.hr = self.emb_rel
        return history_emb, static_emb, self.hr, his_emb, his_r_emb, his_temp_embs, his_rel_embs

    def get_time_emb2(self,t,i):
        time_emb = self.time_embs[torch.tensor(t).to(torch.int64).item() + i]
        time_relu_t=self.leaky_relu(self.time_w)
        time_relude_emb=torch.ger(time_relu_t,time_emb)
        attn = torch.cat([self.dynamic_emb, time_relude_emb], 1)
        return torch.mm(attn, self.temporal_w)

    def predict(self, que_pair, sub_graph, T_id, test_graph, num_rels, static_graph, test_triplets, time_point, err_mat, sequence, use_cuda, ent_ent_his_emb=None, ent_rel_his_emb=None):
        with torch.no_grad():
            all_triples = test_triplets
            e_e_his_emb = ent_ent_his_emb.to(self.gpu)
            e_r_his_emb = ent_rel_his_emb.to(self.gpu)

            e_e_his_emb = self.seq3(e_e_his_emb)
            e_r_his_emb = self.seq4(e_r_his_emb)

            if use_cuda:
                all_triples = all_triples.to(self.gpu)

            uniq_e = que_pair[0]
            r_len = que_pair[1]
            r_idx = que_pair[2]
            temp_r = self.emb_rel[r_idx]
            e_input = torch.zeros(self.num_ents, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_ents, self.h_dim).float()
            for span, e_idx in zip(r_len, uniq_e):
                x = temp_r[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                e_input[e_idx] = x_mean

            query_mask = torch.zeros((self.num_ents,self.h_dim)).to(self.gpu) if use_cuda else torch.zeros(1)
            e1_emb = self.dynamic_emb[uniq_e]
            rel_emb = e_input[uniq_e]
            query_emb = self.w1(torch.concat([e1_emb,rel_emb],dim=1))
            query_mask[uniq_e] = query_emb

            embedding, _, r_emb, his_emb, his_r_emb, his_temp_embs, his_rel_embs = self.forward(sub_graph, T_id, query_mask, test_graph, static_graph, use_cuda, err_mat)

            diffu_repc, score = self.diffusion(sequence, all_triples, embedding, False, embedding, e_e_his_emb, e_r_his_emb)


            if self.pre_type == "all":
                scores_ob,_,_= self.decoder_ob.forward(embedding, r_emb, all_triples, self.pre_weight, self.pre_type,None, diffu_repc)
                score_seq = F.softmax(scores_ob, dim=1)
                score_en =score_seq
            scores_en = torch.log(score_en)

        return all_triples, scores_en

    def get_loss(self, que_pair, sub_graph, T_idx, glist, triples, static_graph, time_point, d_dict_er2e, use_cuda, err_mat, sequence, train_flag = False, ent_ent_his_emb=None, ent_rel_his_emb=None):

        s = triples[:, 0]
        r = triples[:, 1]
        o = triples[:, 2]
        s_len = s.size()[0]
        distribute_er2e = torch.zeros(s_len, self.num_ents).cuda()

        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_cl = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_vocal = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_ent_diff = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        e_e_his_emb = ent_ent_his_emb.to(self.gpu)
        e_r_his_emb = ent_rel_his_emb.to(self.gpu)

        e_e_his_emb = self.seq3(e_e_his_emb)
        e_r_his_emb = self.seq4(e_r_his_emb)

        all_triples = triples

        if use_cuda:
            all_triples = all_triples.to(self.gpu)

        uniq_e = que_pair[0]
        r_len = que_pair[1]
        r_idx = que_pair[2]
        temp_r = self.emb_rel[r_idx]
        e_input = torch.zeros(self.num_ents, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_ents, self.h_dim).float()
        for span, e_idx in zip(r_len, uniq_e):
            x = temp_r[span[0]:span[1],:]
            x_mean = torch.mean(x, dim=0, keepdim=True)
            e_input[e_idx] = x_mean

        query_mask = torch.zeros((self.num_ents,self.h_dim)).to(self.gpu) if use_cuda else torch.zeros(1)
        t1 = torch.tensor(T_idx).cuda().to(self.gpu)
        q_t = torch.sin(self.weight_t2 * 0 + self.bias_t2).repeat(self.num_ents,1)
        qe_emb = self.w4(torch.concat([self.dynamic_emb,q_t],dim=1))

        e1_emb = qe_emb[uniq_e]

        rel_emb = e_input[uniq_e]
        query_emb = self.w1(torch.concat([e1_emb,rel_emb],dim=1))
        query_mask[uniq_e] = query_emb

        embedding, static_emb, r_emb, his_emb, his_r_emb, his_temp_embs, his_rel_embs = self.forward(sub_graph, T_idx, query_mask, glist, static_graph, use_cuda, err_mat)
        diffu_repc, score = self.diffusion(sequence, triples, embedding, train_flag, embedding, e_e_his_emb, e_r_his_emb)

        if self.pre_type == "all":
            scores_ob, _, score1 = self.decoder_ob.forward(embedding, r_emb, all_triples, self.pre_weight, self.pre_type, None, diffu_repc)
            score_seq = F.softmax(scores_ob, dim=1)
            score_en = score_seq

        scores_en = torch.log(score_en)
        loss_ent += F.nll_loss(scores_en, triples[:, 2])
        loss_ent_diff += self.loss_ce2(score, triples[:, 2])

        if self.relation_prediction:
            score_rel = self.rdecoder.forward(embedding,r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])
        #================================================================================================================

        if self.use_cl and self.pre_type=="all":
            for id, evolve_emb in enumerate(his_temp_embs):
                t3 = len(his_temp_embs)-id+1
                query = torch.concat([self.his_ent[all_triples[:, 0]], his_r_emb[all_triples[:, 1]]],dim=1)
                query2 = torch.concat([evolve_emb[all_triples[:, 0]], his_rel_embs[id][all_triples[:, 1]]],dim=1)
                x1 = self.w_cl(query)
                x2 = self.w_cl(query2)
                loss_cl += self.get_loss_conv(x1, x2)

        return loss_ent, loss_rel, loss_static, loss_cl, loss_vocal, loss_ent_diff

    def all_GCN(self, ent_emb, sub_graph, use_cuda):
        sub_graph = sub_graph.to(self.gpu)
        sub_graph.ndata['h'] = ent_emb
        his_emb = self.his_rgcn_layer.forward(sub_graph, ent_emb, [self.emb_rel, self.emb_rel])
        # his_emb = self.fourier.forward([ent_emb])
        subg_index = torch.masked_select(
                torch.arange(0, sub_graph.number_of_nodes(), dtype=torch.long).cuda(),
                (sub_graph.in_degrees(range(sub_graph.number_of_nodes())) > 0))
        return F.normalize(his_emb), subg_index

    def get_loss_conv(self, ent1_emb, ent2_emb):

        loss_fn = nn.CrossEntropyLoss().to(self.gpu)
        z1 = self.projection_model(ent1_emb)
        z2 = self.projection_model(ent2_emb)
        pred1 = torch.mm(z1, z2.T)
        pred2 = torch.mm(z2, z1.T)
        pred3 = torch.mm(z1, z1.T)
        pred4 = torch.mm(z2, z2.T)
        labels = torch.arange(pred1.shape[0]).to(self.gpu)

        train_cl_loss =(loss_fn(pred1 / self.temp, labels) + loss_fn(pred2 / self.temp, labels)+loss_fn(pred3 / self.temp, labels) + loss_fn(pred4 / self.temp, labels)) / 4
        return train_cl_loss

    def diffu_pre(self, item_rep, tag_emb, sr_embs, mask_seq, t, query_sub3=None):
        seq_rep_diffu, item_rep_out = self.diffu(item_rep, tag_emb, sr_embs, mask_seq, t, query_sub3)
        return seq_rep_diffu, item_rep_out

    def reverse(self, item_rep, noise_x_t, sr_embs, mask_seq, mask=None, query_sub3=None):
        reverse_pre = self.diffu.reverse_p_sample(item_rep, noise_x_t, sr_embs, mask_seq, mask, query_sub3)
        return reverse_pre

    def diffusion(self, sequence, tag, evolve_embs, train_flag, current_h, ee, er):
        object_sequence = sequence[:, -self.max_len:, 0]  ### 获取历史交互的object序列  [362, 64]
        relation_sequence = sequence[:, -self.max_len:, 1]  ### 获取历史发生的relation序列  [362, 64]
        max_time = torch.max(sequence[:, -self.max_len:, 2], dim=1, keepdim=True)[0]  ### 获取以上序列对应的 time 序列  [362, 1]
        time_sequence = max_time - sequence[:, -self.max_len:, 2] + 1 + 1  ### 获取 relative time + 1 序列

        time_sequence = torch.concat([time_sequence, torch.ones_like(tag[:, [1]]), torch.ones_like(tag[:, [1]])], dim=-1)  # [362, 65]

        object_embeddings = evolve_embs[object_sequence].clone()  # [362, 64, 200]
        relation_embeddings = self.emb_rel[relation_sequence].clone()  # [362, 64, 200]

        query_rel3 = self.emb_rel[tag[:, 1]].unsqueeze(1)  # [362, 1, 200]

        time_embeddings_re = self.time_embeddings.weight[time_sequence]  ## 获取对应的 embedding 序列
        mask_seq = (time_sequence <= max_time + 2).float()  ## mask padding part as 0
        nn = time_embeddings_re.shape[1]
        relation_embeddings = torch.concat([relation_embeddings, query_rel3, er.unsqueeze(1)], dim=1)

        if train_flag:  ### 训练过程
            ## B x H
            query_object3 = evolve_embs[tag[:, 2]].unsqueeze(1)

            t, weights = self.diffu.schedule_sampler.sample(query_object3.shape[0],
                                                            query_object3.device)  ## t is sampled from schedule_sampler

            query_object_noise = self.diffu.q_sample(query_object3, t)

            object_embeddings = torch.concat([object_embeddings, er.unsqueeze(1), query_object_noise], dim=1)
            nn = time_embeddings_re.shape[1]
            inter_embeddings = torch.cat([object_embeddings, relation_embeddings, time_embeddings_re], dim=-1)

            inter_embeddings_drop = self.LayerNorm(self.embed_dropout(self.Linear(inter_embeddings)))

            rep_diffu = self.diffu_pre(inter_embeddings_drop, inter_embeddings_drop[:, -1, :], evolve_embs, mask_seq, t)
            output = rep_diffu[0]
            score = torch.mm(rep_diffu[0], torch.tanh(current_h).transpose(0, 1))
        else:
            noise_x_t = th.randn_like(object_embeddings[:, [0, 1, -1], :])

            query_object3 = noise_x_t[:, [2], :]
            object_embeddings = torch.concat([object_embeddings, er.unsqueeze(1), query_object3], dim=1)
            nn = time_embeddings_re.shape[1]
            inter_embeddings = torch.cat([object_embeddings, relation_embeddings, time_embeddings_re], dim=-1)

            inter_embeddings_drop = self.LayerNorm(self.embed_dropout(self.Linear(inter_embeddings)))

            rep_diffu = self.reverse(inter_embeddings_drop, inter_embeddings_drop[:, -1, :], evolve_embs, mask_seq, [])
            output = rep_diffu[0]
            score = torch.mm(rep_diffu[0], torch.tanh(current_h).transpose(0, 1))

        return output, score
