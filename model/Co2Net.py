import torch
import torch.nn as nn

import torch.nn.functional as F


from einops import rearrange

from com import  DConv, MLP,Mlp_conv,LayerNorm

    
class  LAB(nn.Module):
    def __init__(self, f_number, k_size=7,padding_mode='reflect') -> None:
        super().__init__()
        # self.channel_independent =DConv7(f_number,k_size,padding_mode)
        self.channel_independent =DConv(f_number,k_size,padding_mode)
        self.channel_dependent = MLP(f_number, excitation_factor=2)

    def forward(self, x):
        return self.channel_dependent(self.channel_independent(x))
  


##########################################################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class ICoB_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        """ input: 2dim->[dim,L],[dim,L]
            output: 2dim 
        """
        super(ICoB_Attention, self).__init__()
        self.num_heads = num_heads
        dim=dim//2
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv2 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv1 = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.qkv_dwconv2= nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
     
        
    def forward(self, x):
       
        b, c, h, w = x.shape
        x1,x2=torch.chunk(x,2,dim=1)

        qkv1 = self.qkv_dwconv1(self.qkv1(x1))
        qkv2 = self.qkv_dwconv2(self.qkv2(x2))
        
        q1, k1, v1 = qkv1.chunk(3, dim=1)
        q2, k2, v2 = qkv2.chunk(3, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)


        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)


        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature1
        attn1 = attn1.softmax(dim=-1)
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature2
        attn2 = attn2.softmax(dim=-1)
        out1= (attn1@v2)
        out2= (attn2@v1)
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out1 = self.project_out1(out1)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out2 = self.project_out2(out2)

        return torch.cat([out1,out2],dim=1)


##########################################################################
class GCoB_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,):
        super(GCoB_TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
       
        self.ffn= Mlp_conv(dim,ffn_expansion_factor,bias)
    def forward(self, x,training=False):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x),training)

        return x




class ICoB_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(ICoB_TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = ICoB_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
      
        self.ffn= Mlp_conv(dim,ffn_expansion_factor,bias)
    def forward(self, x,training=False):

        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x),training)

        return x




class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x








class Co2M(nn.Module):
    def __init__(self,
                 dim=64,
                 heads=[8, 8, 8], 
                 k_size=7,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Co2M, self).__init__()
    

        self.LAB1_layer =LAB(f_number=dim,k_size=k_size)
        self.LAB2_layer =LAB(f_number=dim,k_size=k_size)
    
      
        self.ICoB_layer = ICoB_TransformerBlock(dim=dim*2, num_heads=heads[0], ffn_expansion_factor=1,bias=bias, LayerNorm_type=LayerNorm_type)
            
        self.GCoB_layer = GCoB_TransformerBlock(dim=dim*2, num_heads=heads[0], ffn_expansion_factor=1,bias=bias, LayerNorm_type=LayerNorm_type)
  
    def forward(self, A_in,A_in_T,x_in,x_in_T,training=False):
        #m=n-1
        A_out = self.LAB1_layer(A_in)
        A_out_T = self.LAB2_layer(A_in_T)

        B_out,B_out_T = torch.chunk(self.ICoB_layer(torch.cat([A_out+x_in,A_out_T+x_in_T],dim=1),training),2,dim=1)
        x_out,x_out_T = torch.chunk(self.GCoB_layer(torch.cat([B_out,B_out_T],dim=1),training),2,dim=1)
        return  A_out,A_out_T,x_out,x_out_T



class Co2Net(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 dim=64,
                 num_blocks=[5, 5],
                 heads=[8, 8, 8], 
                 k_size=7,
                 bias=False,
                 LayerNorm_type='WithBias',
               
                 ):

        super(Co2Net, self).__init__()
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim)
        self.patch_embed1 = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim)

        self.Co2M_layers = nn.ModuleList()
        for i  in range(num_blocks[0]):
           self.Co2M_layers.append(Co2M(dim=dim, heads=heads, k_size=k_size, bias=bias, LayerNorm_type=LayerNorm_type))
    

        self.T_tial =  nn.Sequential(nn.Conv2d(dim*2,dim*2,kernel_size=3,padding='same',groups=dim,bias=False),
                              nn.Conv2d(dim*2,6,kernel_size=1,stride=1,padding=0,groups=1))
        self.CNN_tial1=nn.Conv2d(dim,3,1,bias=False)

             
    def forward(self, inp_img1,inp_img2,training=False):
        A= self.patch_embed(inp_img1)
        AT= self.patch_embed1(inp_img2)
        x=torch.zeros_like(A)
        xT=torch.zeros_like(AT)
        for indx,Co2M_layer in enumerate(self.Co2M_layers):
            A,AT,x,xT = Co2M_layer(A,AT,x,xT,training)

        out1= self.CNN_tial1(0.5*(x+xT))

        return out1





if __name__ == '__main__':
    print("==start==")
    img_size = 256
    patch_size = 4
    inp_channels = 3
    batch_size = 2
    num_blocks = [5, 5]
   
    dim = 32
    k_size=7
    bias = False
    model = Co2Net(dim=dim, num_blocks=num_blocks, k_size=7).cuda()

    img = torch.randn(batch_size, inp_channels, img_size, img_size).cuda()
    out = model(img,img,True)
    print(out.shape)





