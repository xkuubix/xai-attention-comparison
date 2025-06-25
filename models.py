import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from image_patcher import ImagePatcher

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
                attention_dropout=0.1):

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

        self.patcher = ImagePatcher(patch_size=128,
                                    overlap=0.75,
                                    empty_thresh=0.75,
                                    bag_size=-1)
        # torch.Size([3, 2294, 1914]) in CMMD
        self.patcher.get_tiles(2294, 1914)
        self.reconstruct_attention = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
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
        Y = self.classifier(m)

        if self.reconstruct_attention:
            A = self.patcher.reconstruct_attention_map(A.unsqueeze(0), instances_ids, x.shape[2:])

        return Y, A


    def mc_inference(self, x, n=30):
        self.eval()
        def enable_dropout(m):
            if isinstance(m, torch.nn.Dropout):
                m.train()

        self.apply(enable_dropout)
        
        instances, instances_ids, _ = self.patcher.convert_img_to_bag(x.squeeze(0))
        instances = instances.unsqueeze(0)
        instances = self.norm_instances(instances)
        instances = instances.to(self.device)

        bs, num_instances, ch, w, h = instances.shape
        instances = instances.view(bs*num_instances, ch, w, h)
        # Feature extraction only once
        with torch.no_grad():
            H = self.feature_extractor(instances)
            H = H.view(bs, num_instances, -1)

        predictions = []
        attention_weights = []
        
        with torch.no_grad():
            for _ in range(n):
                H_dropout = self.feature_dropout(H)  # Apply dropout in each pass

                A_V = self.attention_V(H_dropout)
                A_U = self.attention_U(H_dropout)
                A = self.attention_weights(torch.mul(A_V, A_U))
                A = torch.transpose(A, 2, 1)
                A = F.softmax(A, dim=2)

                m = torch.matmul(A, H_dropout)
                outputs = self.classifier(m)
                outputs = F.sigmoid(outputs)

                predictions.append(outputs)
                attention_weights.append(A)

                torch.cuda.empty_cache()
                gc.collect()

        predictions = torch.stack(predictions)  # Shape: (n, batch_size, num_classes)
        attention_weights = torch.stack(attention_weights)  # Shape: (n, batch_size, num_instances, K)
            
    
        if self.reconstruct_attention:
            attention_weights = self.patcher.reconstruct_attention_map(attention_weights, instances_ids, x.shape[2:])
            attention_weights = attention_weights.mean(dim=0).unsqueeze(0)  # Average over MC samples
        return predictions, attention_weights
    
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
            neptune_run=None):
        
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

        self.patcher = ImagePatcher(patch_size=128,
                                    overlap=0.75,
                                    empty_thresh=0.75,
                                    bag_size=-1)
        # torch.Size([3, 2294, 1914]) in CMMD
        self.patcher.get_tiles(2294, 1914)
        self.reconstruct_attention = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):

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
    

    def mc_inference(self, x, N=30):
        """
        Optimized MC dropout inference that:
        1. Extracts features ONCE
        2. Expands features for parallel MC sampling
        3. Computes attention and predictions in parallel
        """
        self.eval()
        def enable_dropout(m):
            if isinstance(m, torch.nn.Dropout):
                m.train()

        self.apply(enable_dropout)
        
        instances, instances_ids, _ = self.patcher.convert_img_to_bag(x.squeeze(0))
        instances = instances.unsqueeze(0)
        instances = self.norm_instances(instances)
        instances = instances.to(self.device)

        bs, num_instances, ch, w, h = instances.shape
        instances = instances.view(bs*num_instances, ch, w, h)
        # Feature extraction only once
        with torch.no_grad():
            H = self.feature_extractor(instances)
            H = H.view(bs, num_instances, -1)

        with torch.no_grad():
            # 1. Feature extraction (single pass)
            H = self.feature_extractor(instances.view(-1, *instances.shape[-3:]))
            H = H.view(bs, num_instances, -1)  # (bs, num_instances, L)
            
            # 2. Expand features for MC sampling
            H_expanded = H.unsqueeze(0).expand(N, -1, -1, -1)  # (N, bs, num_instances, L)
            H_drop = self.feature_dropout(H_expanded)  # Apply dropout to all samples
            
            # 3. Parallel attention computation
            if self.shared_attention:
                A_V = self.attention_V(H_drop)  # (N, bs, num_instances, D)
                A_U = self.attention_U(H_drop)  # (N, bs, num_instances, D)
                A = A_V * A_U                   # (N, bs, num_instances, D)

                A = torch.stack([self.attention_weights[i](A) for i in range(self.num_classes)], dim=2)  # (N, bs, num_classes, num_instances, 1)
                A = A.squeeze(-1)  # (N, bs, num_classes, num_instances)
                A = torch.stack([self.attention_dropouts[i](A[:, :, i, :]) for i in range(self.num_classes)], dim=2)
                # (N, bs, num_classes, num_instances)
            else:
                # For separate attention, we need to loop through classes
                A = []
                for i in range(self.num_classes):
                    A_V = self.attention_V[i](H_drop)
                    A_U = self.attention_U[i](H_drop)
                    A_i = self.attention_weights[i](A_V * A_U)  # (N, bs, num_instances, 1)
                    A_i = A_i.transpose(-1, -2)  # (N, bs, 1, num_instances)
                    A_i = self.attention_dropouts[i](A_i)
                    A.append(A_i)
                A = torch.cat(A, dim=2)  # (N, bs, num_classes, num_instances)
            
            A = F.softmax(A, dim=-1)
            
            # 4. Parallel prediction computation
            M = torch.stack([
                torch.bmm(A[:, :, i, :], H_drop.squeeze(1))
                for i in range(A.size(2))
            ], dim=2)
            # Classify all samples in parallel
            Y = torch.stack([
                self.classifiers[i](M[:, :, i]) for i in range(self.num_classes)
            ], dim=-1)  # (N, bs, num_classes)
        Y = Y.squeeze(-2)
        if self.reconstruct_attention:
            A = self.patcher.reconstruct_attention_map(A, instances_ids, x.shape[2:])
            A = A.mean(dim=0).unsqueeze(0)  # Average over MC samples
        return Y, A
    
    @staticmethod
    def norm_instances(instances):
        """
        Normalize instances (ImageNet normalization)
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=instances.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=instances.device).view(1, 1, 3, 1, 1)
        return (instances - mean) / std