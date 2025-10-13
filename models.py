# %%
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from image_patcher import ImagePatcher
import numpy as np
from calflops import calculate_flops

class Flatten(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        input_shape = x.shape
        output_shape = [input_shape[i] for i in range(self.dim)] + [-1]
        return x.view(*output_shape)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class GatedAttentionMIL(nn.Module):

    def __init__(
                self,
                num_classes=1,
                backbone='r18',
                pretrained=True,
                L=512,
                D=128,
                K=1,
                feature_dropout=0.1,
                attention_dropout=0.1,
                config=None
                ):

        super().__init__()
        self.L = L  
        self.D = D
        self.K = K
        if pretrained:
            if backbone == 'r18':
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet18(weights=weights)
            elif backbone == 'r34':
                weights = models.ResNet34_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet34(weights=weights)
            elif backbone == 'r50':
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet50(weights=weights)
        else:
            self.feature_extractor = models.resnet18()
        self.num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Dropout(attention_dropout)
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid(),
            nn.Dropout(attention_dropout)
        )
        self.attention_weights = nn.Linear(self.D, self.K)
        self.classifier = nn.Sequential(
                                        nn.Linear(self.L * self.K,
                                                  num_classes))
        self.feature_dropout = nn.Dropout(feature_dropout)

        self.patcher = ImagePatcher(patch_size=config['data']['patch_size'] if config else 128,
                                    overlap=config['data']['overlap'] if config else 0.,
                                    empty_thresh=config['data']['empty_threshold'] if config else 0.75,
                                    bag_size=config['data']['bag_size'] if config else -1)
        # torch.Size([3, 2294, 1914]) in CMMD
        self.reconstruct_attention = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        self.patcher.set_tiles(x.shape[2], x.shape[3])
        instances, instances_ids, _ = self.patcher.convert_img_to_bag(x.squeeze(0))
        instances = instances.unsqueeze(0)
        instances = self.norm_instances(instances)
        instances = instances.to(self.device)

        bs, num_instances, ch, w, h = instances.shape
        instances = instances.view(bs*num_instances, ch, w, h)
        H = self.feature_extractor(instances)
        H = self.feature_dropout(H)
        H = H.view(bs, num_instances, -1)
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(torch.mul(A_V, A_U))
        A = torch.transpose(A, 2, 1)
        A = F.softmax(A, dim=2)
        m = torch.matmul(A, H)
        Y = self.classifier(m).view(bs, -1)

        if self.reconstruct_attention:
            A = self.patcher.reconstruct_attention_map(A.unsqueeze(0), instances_ids, x.shape[2:])

        return Y, A
    
    @staticmethod
    def norm_instances(instances):
        """
        Normalize instances
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=instances.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=instances.device).view(1, 1, 3, 1, 1)
        return (instances - mean) / std


class MultiHeadGatedAttentionMIL(nn.Module):
    def __init__(
            self,
            num_classes=2,
            backbone='r18',
            pretrained=True,
            L=512,
            D=128,
            feature_dropout=0.1,
            attention_dropout=0.1,
            shared_attention=False,
            neptune_run=None,
            config=None):
        
        super().__init__()

        if neptune_run:
            self.fold_idx = None
            self.neptune_run = neptune_run
        else:
            self.fold_idx = None
            self.neptune_run = None
        # TODO: adjust L for resnet50
        self.L = L
        self.D = D
        self.num_classes = num_classes
        self.shared_attention = shared_attention

        # Feature extractor
        if pretrained:
            if backbone == 'r18':
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet18(weights=weights)
            elif backbone == 'r34':
                weights = models.ResNet34_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet34(weights=weights)
            elif backbone == 'r50':
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet50(weights=weights)
        else:
            self.feature_extractor = models.resnet18()

        self.feature_extractor.fc = Identity()

        # Attention mechanism (Shared or Separate)
        if shared_attention:
            self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        else:
            self.attention_V = nn.ModuleList([
                nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
                for _ in range(self.num_classes)
            ])
            self.attention_U = nn.ModuleList([
                nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
                for _ in range(self.num_classes)
            ])

        # Attention weight layers (Always separate per class)
        self.attention_weights = nn.ModuleList([
            nn.Linear(self.D, 1) for _ in range(self.num_classes)
        ])

        # Classifiers per class
        self.classifiers = nn.ModuleList([
            nn.Linear(self.L, 1, bias=False) for _ in range(self.num_classes)
        ])

        # Dropout layers
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.attention_dropouts = nn.ModuleList([
            nn.Dropout(attention_dropout) for _ in range(self.num_classes)
        ])

        self.patcher = ImagePatcher(patch_size=config['data']['patch_size'] if config else 128,
                                    overlap=config['data']['overlap'] if config else 0.,
                                    empty_thresh=config['data']['empty_threshold'] if config else 0.75,
                                    bag_size=config['data']['bag_size'] if config else -1)
        # torch.Size([3, 2294, 1914]) in CMMD
        self.patcher.set_tiles(2294, 1914)
        self.reconstruct_attention = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        self.patcher.set_tiles(x.shape[2], x.shape[3])
        instances, instances_ids, _ = self.patcher.convert_img_to_bag(x.squeeze(0))
        instances = instances.unsqueeze(0)
        instances = self.norm_instances(instances)
        instances = instances.to(self.device)
        bs, num_instances, ch, w, h = instances.shape
        instances = instances.view(bs*num_instances, ch, w, h)
        H = self.feature_extractor(instances)
        H = self.feature_dropout(H)
        H = H.view(bs, num_instances, -1)

        M = []
        A_all = []

        for i in range(self.num_classes):
            if self.shared_attention:
                A_V = self.attention_V(H)
                A_U = self.attention_U(H)
            else:
                A_V = self.attention_V[i](H)
                A_U = self.attention_U[i](H)

            A = self.attention_weights[i](A_V * A_U)
            A = torch.transpose(A, 2, 1)  # (bs, 1, num_instances)
            A = self.attention_dropouts[i](A)
            A = F.softmax(A, dim=2)

            A_all.append(A)
            M.append(torch.matmul(A, H))

        M = torch.cat(M, dim=1)  # (bs, num_classes, L)
        A_all = torch.cat(A_all, dim=1)  # (bs, num_classes, num_instances)

        Y = [self.classifiers[i](M[:, i, :]) for i in range(self.num_classes)]
        Y = torch.cat(Y, dim=-1)  # (bs, num_classes)
        if self.reconstruct_attention:
            A_all = self.patcher.reconstruct_attention_map(A_all.unsqueeze(0), instances_ids, x.shape[2:])
        return Y, A_all
    
    @staticmethod
    def norm_instances(instances):
        """
        Normalize instances (ImageNet normalization)
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=instances.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=instances.device).view(1, 1, 3, 1, 1)
        return (instances - mean) / std
    

# %%
# based on https://github.com/binli123/dsmil-wsi/blob/master/dsmil.py
class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class IClassifier(nn.Module):
    def __init__(self,
                 output_class=1,
                 backbone='r18',
                 pretrained=True,
                 ):
        super(IClassifier, self).__init__()
        
        if pretrained:
            if backbone == 'r18':
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet18(weights=weights)
            elif backbone == 'r34':
                weights = models.ResNet34_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet34(weights=weights)
            elif backbone == 'r50':
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet50(weights=weights)
        else:
            self.feature_extractor = models.resnet18()
        self.num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()    
        self.fc = nn.Linear(self.num_features, output_class)

    def forward(self, x):
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B
    
class DSMIL(nn.Module):
    def __init__(self,
                 num_classes=1,
                 backbone='r18',
                 pretrained=True,
                 nonlinear=True,
                 dropout_v=0.1,
                 passing_v=False,
                 config=None
                 ):
        super(DSMIL, self).__init__()
        self.i_classifier = IClassifier(
            backbone=backbone,
            pretrained=pretrained,
            output_class=num_classes
        )
        self.b_classifier = BClassifier(
            input_size=self.i_classifier.num_features,
            output_class=num_classes,
            dropout_v=dropout_v,
            nonlinear=nonlinear,
            passing_v=passing_v
        )
        self.patcher = ImagePatcher(patch_size=config['data']['patch_size'] if config else 128,
                                    overlap=config['data']['overlap'] if config else 0.,
                                    empty_thresh=config['data']['empty_threshold'] if config else 0.75,
                                    bag_size=config['data']['bag_size'] if config else -1)
        # torch.Size([3, 2294, 1914]) in CMMD
        self.patcher.set_tiles(2294, 1914)
        self.reconstruct_attention = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        self.patcher.set_tiles(x.shape[2], x.shape[3])
        instances, instances_ids, _ = self.patcher.convert_img_to_bag(x.squeeze(0))
        instances = instances.unsqueeze(0)
        instances = self.norm_instances(instances)
        instances = instances.to(self.device)
        # print(f"{instances.shape=}")
        feats, classes = self.i_classifier(instances.squeeze(0)) # in: NCHW
        # print(f"{feats.shape=}, {classes.shape=}")
        prediction_bag, A, B = self.b_classifier(feats, classes)
        
        if self.reconstruct_attention:
            A = self.patcher.reconstruct_attention_map(A.permute(1, 0).unsqueeze(0).unsqueeze(0), instances_ids, x.shape[2:])
        else:
            A = A.permute(1, 0).unsqueeze(0)  # 1x1x(N or HxW)

        return (classes, prediction_bag), (A, B)
        
    @staticmethod
    def norm_instances(instances):
        """
        Normalize instances (ImageNet normalization)
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=instances.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=instances.device).view(1, 1, 3, 1, 1)
        return (instances - mean) / std


# based on https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 512, D = 128, dropout = True, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L=512, D=128, dropout=0.1, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.1))
            self.attention_b.append(nn.Dropout(0.1))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.1)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=0.1, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=512,
        backbone='r18', pretrained=True, output_class=1, config=None):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 128], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        
        if pretrained:
            if backbone == 'r18':
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet18(weights=weights)
            elif backbone == 'r34':
                weights = models.ResNet34_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet34(weights=weights)
            elif backbone == 'r50':
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet50(weights=weights)
        else:
            self.feature_extractor = models.resnet18()
        self.num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()    
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.patcher = ImagePatcher(patch_size=config['data']['patch_size'] if config else 128,
                                    overlap=config['data']['overlap'] if config else 0.,
                                    empty_thresh=config['data']['empty_threshold'] if config else 0.75,
                                    bag_size=config['data']['bag_size'] if config else -1)
        # torch.Size([3, 2294, 1914]) in CMMD
        self.patcher.set_tiles(2294, 1914)
        self.reconstruct_attention = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, x, label=None, instance_eval=False, return_features=False, attention_only=False):
        self.patcher.set_tiles(x.shape[2], x.shape[3])
        instances, instances_ids, _ = self.patcher.convert_img_to_bag(x.squeeze(0))
        instances = instances.unsqueeze(0)
        instances = self.norm_instances(instances)
        instances = instances.to(self.device)
        h = self.feature_extractor(instances.squeeze(0)) # in: NCHW
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
    
        if self.reconstruct_attention:
            A = self.patcher.reconstruct_attention_map(A.unsqueeze(0).unsqueeze(0), instances_ids, x.shape[2:])
        else:
            A = A.unsqueeze(0).unsqueeze(0)  # 1x1x(N or HxW)
            pass
        return (logits, Y_prob, Y_hat, results_dict), A  # 1x1x(N or HxW)

    @staticmethod
    def norm_instances(instances):
        """
        Normalize instances (ImageNet normalization)
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=instances.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=instances.device).view(1, 1, 3, 1, 1)
        return (instances - mean) / std



def get_FLOPS(model, input):
    flops, macs, params = calculate_flops(model=model,
                                    output_as_string=True,
                                    output_precision=4,
                                    args=[input],
                                    print_results=False,
                                    print_detailed=False)
    print(f'Model: {model.__class__.__name__}  FLOPS: {flops},  MACs: {macs},  Params: {params}')
    return flops, macs, params


# input_shape = (120, 1024)
# flops, macs, params = calculate_flops(model=model, 
#                                       input_shape=input_shape,
#                                       output_as_string=True,
#                                       output_precision=4)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# input_tensor = torch.ones(1,3,3000,3000).to(device)

# clam = CLAM(gate=True, size_arg="small", dropout=0.1, k_sample=8, n_classes=2,
#         instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=512,
#         backbone='r18', pretrained=True, output_class=1, config=None).to(device)
# clam.reconstruct_attention = True
# y, aux = clam(input_tensor,
#             label=torch.tensor([1]).to(device),
#             instance_eval=True)
# print(f"{y[0].shape=}, {y[1].shape=}, {y[2].shape=}, {y[3]=}, {aux.shape=}")

# dsmil = DSMIL(num_classes=1, backbone='r18', pretrained=True).to(device)
# dsmil.reconstruct_attention = True
# a, b = dsmil(input_tensor)
# classes, prediction_bag = a
# A, B = b
# print(f"{classes.shape=}, {prediction_bag.shape=}, {A.shape=}, {B.shape=}")

# gamil = GatedAttentionMIL(backbone='r18', pretrained=True).to(device)
# gamil.reconstruct_attention = True
# Y, A = gamil(input_tensor)
# print(f"{Y.shape=}, {A.shape=}")


# mhgamil = MultiHeadGatedAttentionMIL(backbone='r18', pretrained=True).to(device)
# mhgamil.reconstruct_attention = True
# Y, A = mhgamil(input_tensor)
# print(f"{Y.shape=}, {A.shape=}")



# get_FLOPS(clam, input_tensor)
# get_FLOPS(mhgamil, input_tensor)
# get_FLOPS(dsmil, input_tensor)
# get_FLOPS(gamil, input_tensor)

# instances.shape=torch.Size([1, 270, 3, 128, 128])
# feats.shape=torch.Size([270, 512]), classes.shape=torch.Size([270, 1])
# classes.shape=torch.Size([270, 1]), prediction_bag.shape=torch.Size([1, 1]), A.shape=torch.Size([1, 1, 1, 3000, 3000]), B.shape=torch.Size([1, 1, 512])
# Y.shape=torch.Size([1, 1]), A.shape=torch.Size([1, 1, 1, 3000, 3000])
# Y.shape=torch.Size([1, 2]), A.shape=torch.Size([1, 2, 1, 3000, 3000])

# %%