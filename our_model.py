import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
import torch.distributed as dist
from src.attention.attentions import *
import torchvision.models as models


NUM_SAMPLES=1

np.random.seed(3483)
torch.manual_seed(3483)
torch.cuda.manual_seed(3483)
torch.cuda.manual_seed_all(3483)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SequentialMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SequentialMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)

    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)
        seq_len_v = v.size(1)
        
        q = self.positional_encoding(q)  # Add positional encoding to q
        k = self.positional_encoding(k)  # Add positional encoding to k
        v = self.positional_encoding(v)  # Add positional encoding to v

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Scaled Dot-Product Attention
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, num_heads, seq_len_q, seq_len_k)

        dk = k.size()[-1]
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  # Add the mask to the scaled tensor.

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        output = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth)

        output = output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len_q, num_heads, depth)
        output = output.view(batch_size, -1, self.d_model)  # (batch_size, seq_len_q, d_model)

        output = self.dense(output)  # (batch_size, seq_len_q, d_model)

        return output


class SpatialMultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_heads, d_k):
        super(SpatialMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.in_channels = in_channels

        self.query = nn.Conv2d(in_channels, num_heads * d_k, kernel_size=1)
        self.key = nn.Conv2d(in_channels, num_heads * d_k, kernel_size=1)
        self.value = nn.Conv2d(in_channels, num_heads * d_k, kernel_size=1)
        self.fc = nn.Conv2d(num_heads * d_k, in_channels, kernel_size=1)
        
        self.attention = None

    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        # Generate queries, keys, and values
        q = self.query(x).view(batch_size, self.num_heads, self.d_k, height * width)
        k = self.key(x).view(batch_size, self.num_heads, self.d_k, height * width)
        v = self.value(x).view(batch_size, self.num_heads, self.d_k, height * width)
        
        # Transpose for dot product attention
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attention, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, -1, height, width)
        
        # Final linear layer
        out = self.fc(out)
        
        return out

class SequentialAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(SequentialAttentionBlock, self).__init__()
        self.multi_head_attention = SequentialMultiHeadAttention(in_channels, num_heads)

    def forward(self, x):
        x = self.multi_head_attention(x,x,x)
        return x

class SequentialCrossAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(SequentialCrossAttentionBlock, self).__init__()
        self.multi_head_attention = SequentialMultiHeadAttention(in_channels, num_heads)

    def forward(self, x,y):
        x = self.multi_head_attention(x,y,x)
        return x


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        def ConvBlock(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def ConvBlock3(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def layer1transpose(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )


        self.decoder = nn.Sequential(
            #ConvBlock 1
            ConvBlock(256, 128),
            layer1transpose(128, 128),
            #ConvBlock 2
            ConvBlock(128, 64),
            layer1transpose(64, 64),
            ConvBlock3(64, 3, 32),
        )

        self.de_dense = nn.Sequential(
            nn.Linear(8192,768),
            nn.ELU(),
            nn.Linear(768, 256 * 16 * 16), #To be 768 later
            nn.ELU(),
        )

    def forward(self, x):
        output = self.de_dense(x)
        output = output.view(x.size(0), 256, 16, 16)
        output = self.decoder(output)
        return output


class WeightedAverageLayer(nn.Module):
    def __init__(self, dim_size):
        super(WeightedAverageLayer, self).__init__()
        self.dim_size = dim_size  # Size of the dimension to average over
        self.weights = torch.randn(self.dim_size,device='cuda')

    def forward(self, x):
        # Step 2: Generate positive random weights
        weights = torch.abs(self.weights)
        
        # Step 3: Normalize the weights
        weights = weights / weights.sum()
        
        # Step 4: Reshape the weights to be broadcastable
        weights = weights.view(1, self.dim_size, 1)
        
        # Apply the weighted average
        weighted_avg = (x * weights).sum(dim=1)
        
        return weighted_avg

class DADFSM(nn.Module):
    def __init__(self,args):
        super(DADFSM,self).__init__()
        
        self.args = args 
        
        # Using ResNet Backbone
        if self.args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif self.args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)

        last_layer_idx = -2
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        self.num_patches = 16
        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))

        # Linear-based patch-level attention over the 16 patches
        self.attn_pat = Self_Attn_Bot(self.args.trans_linear_in_dim, self.num_patches)

        # MLP-mixing frame-level enrichment over the 4 frames.
        self.fr_enrich = MLP_Mix_Enrich(self.args.trans_linear_in_dim, self.args.num_frames)
        self.spatial_average = WeightedAverageLayer(self.num_patches)
        self.early_attention = SequentialCrossAttentionBlock(self.args.trans_linear_in_dim,4)
        self.temp_average = WeightedAverageLayer(self.args.num_frames)


        self.decoder = decoder() 

    def forward(self,x,y):
        x = x.float().cuda()
        y = y.float().cuda()

        # Extract features 
        x = x.reshape(-1,self.args.in_channels,self.args.img_size[0],self.args.img_size[1])
        y = y.reshape(-1,self.args.in_channels,self.args.img_size[0],self.args.img_size[1])

        x = self.resnet(x)
        y = self.resnet(y)

        # decrease to 4 X 4 = 16 patches 
        x = self.adap_max(x)
        x = x.reshape(-1,self.args.trans_linear_in_dim,self.num_patches)
        x = x.permute(0,2,1)

        y = self.adap_max(y)
        y = y.reshape(-1,self.args.trans_linear_in_dim,self.num_patches)
        y = y.permute(0,2,1)

        #Spatial attention
        x = self.attn_pat(x)
        y = self.attn_pat(y)

        #Average across patches 
        x = self.spatial_average(x) #torch.mean(x,dim=1)
        y = self.spatial_average(y) #torch.mean(y,dim=1)

        #reshape before heading into temporal attention 
        x = x.reshape(-1,self.args.num_frames,self.args.trans_linear_in_dim)        
        y = y.reshape(-1,self.args.num_frames,self.args.trans_linear_in_dim)        

        #Spatial attention
        x = self.fr_enrich(x)
        y = self.fr_enrich(y)

        z = self.early_attention(x,y)

        z = torch.flatten(z,start_dim=1)
        out = self.decoder(z)

        return out