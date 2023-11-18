import torch
import torch.nn as nn 
from torch import Tensor
from joeynmt.encoders import TransformerEncoder
from reinforces.reinforce_utils import *
from torch.distributions import Normal

# create sinosoidal position indicator for sequence
def position_embedding(src_emb):
    """create sinosoidal position embeddings for encoding by given embedded src
    this indicates the perturbation position
    :param src_emb: [batch, len, dim]
    :return pe: [1, len, dim], for auto broadcast
    """
    _, len, dim = src_emb.shape
    if dim % 2 != 0:
        raise ValueError(f"Cannot use sin/cos positional encoding with "
                            f"odd dim (got dim={dim})")
    pe = src_emb.new_zeros(len, dim)
    position = torch.arange(0, len).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                            -(math.log(10000.0) / dim)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    pe = pe.unsqueeze(0)  # shape: [1, len, dim] then broadcast to embeddings
    return pe.detach()


class Agent(nn.Module):
    """
    a one-step reconstruction and a one-step mean-preserving gaussian adversarial.
    the Agent must learn the a mean-preserving noise scale apt to specific tokens.
    the agent takes in the textual representation(matrix) for policy. 
    agent consists of a actor & critic with shared feature layers.

    noising p(x_t|x_t-1) takes the src embedding and noise scale as inputs with only std prediction.
    alpha = \pi(emb_t,t);
    x_t = alpha * x_{t-1} + \sqrt{1-alpha**2}  this is a single-step noising
    
    critic(V(s_t) = Net(s_t)) takes the src embedding and purturbation 
    actor and critic shares the input and feature modules, but different output layers
    the value estimates the quality of a representation. 

    the denoising will directly denoise by that noise scale with posterior mean estimation

    states: the current representation of the src 
    actions: a perturbation matrix on the current representation
    rewards: BLEU scores or discriminator values
    """
    def __init__(self,
            embedding_dim:int,
            action_range,
            action_roll_steps:int=5, 
            max_roll_steps:int=40,
            d_model:int=256, dropout:float=0.0, 
            **kwargs):
        super(Agent, self).__init__()
        self.input_dim = embedding_dim
        self.action_dim = self.input_dim # the output dimension
        self.action_roll_steps = action_roll_steps  # for on policy value estimation training
        self.d_model = d_model
        self.dropout = dropout
        self.action_range = action_range  # perturbation range
        # extract std range by given action range
        self.min_scale = action_range[0]
        self.max_scale = action_range[1]
        print("normalized_std_range within (%f,%f)"%(self.min_scale, self.max_scale))

        # diffusion schedule by karras methods
        self.sigma_data=0.5
        self.rho = kwargs["rho"] if "rho" in kwargs else 7  
        indices = torch.linspace(0,1, max_roll_steps-1)
        self.noise_scales = (self.min_scale**(1/self.rho) + indices*(self.max_scale**(1/self.rho)-self.min_scale**(1/self.rho)))**self.rho

        # input processing
        self.input_layer_norm = nn.LayerNorm(self.d_model, elementwise_affine=True)

        # relevant features from input states
        self.feature_model = TransformerEncoder(
            hidden_size=d_model, ff_size=2*d_model, 
            num_layers=kwargs["num_layers"], num_heads=kwargs["num_heads"], 
            dropout=self.dropout,emb_dropout=self.dropout,
            freeze=False)
        
        # Continuous diffusion policy,
        # estimated V value for encoded states, the meaning-preserving 
        # noise scale within a limited variance
        self.critic = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, 1)
        )

        # the diffusion reconstruction
        self.denoise_linear = nn.Sequential(  # reconstructed normed x_0 given normed_x_t
            nn.Linear(self.d_model*2, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.action_dim)
        )
        self.denoise_scale_linear = nn.Sequential( 
            nn.Linear(self.action_dim, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.action_dim)
        )
        self.denoise_reconstruct = nn.Sequential(  # trained on local and soft updates the global
            nn.Linear(self.d_model, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.action_dim)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for _, p in self.named_parameters():
            default_init(p)
    
    def sync_from(self, agent):
        self.load_state_dict(agent.state_dict())

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.min_scale) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.min_scale)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def forward(self, 
        normed_src_emb:Tensor, src_mask:Tensor, src_length:Tensor, 
        ):
        """ 
        default noising forward with diffusion policy: \pi(s_t, t)
        :param src_emb: batched padded src tokens [batch, len, dim]
        :param src_mask: indicate valid token [batch, 1, len]
        :param src_length: [batch]
        :param noise_scale: the noise scale within (0,1) from original embedding [batch, len, dim]
        generate purturbation by given src embedding
        """
        input = normed_src_emb + position_embedding(normed_src_emb)  # indicates the token position
        # input +=  self.noise_scale_linear(noise_scale)  # expand position dimension
        input_feature = self.input_layer_norm(input)
        encoded_feature, _ = self.feature_model(input_feature, src_length, src_mask)
        encoded_feature = self.input_layer_norm(encoded_feature)
        return encoded_feature
    
    def denoiser_model(self, 
        normed_src_emb:Tensor, src_mask:Tensor, src_length:Tensor, 
        noise_scale:Tensor, 
        ):
        """ 
        the Denoiser net D_\theta given corresponding noise scale
        :param src_emb: batched padded src tokens [batch, len, dim]
        :param src_mask: indicate valid token [batch, 1, len]
        :param src_length: [batch]
        :param noise_scale: the noise scale within (0,1) from original embedding [batch, len, dim]
        generate purturbation by given src embedding
        """
        input = normed_src_emb + position_embedding(normed_src_emb)  # indicates the token position
        input += self.denoise_scale_linear(noise_scale)
        input_feature = self.input_layer_norm(input)
        encoded_feature, _ = self.feature_model(input_feature, src_length, src_mask)
        encoded_feature = self.input_layer_norm(encoded_feature)
        model_out = self.denoise_linear(encoded_feature)
        return model_out
    

    def reconstruct(self,
        next_normed_src_emb:Tensor, src_mask:Tensor, src_length:Tensor,
        noise_scale:Tensor
        ):
        """
        reconstruct the initial normed_src_emb, given x_t
        the F_\theta
        """
        encoded_feature = self.forward(
            next_normed_src_emb, src_mask, src_length, 
            noise_scale)
        reconstructed_normed_src_emb= self.denoise_reconstruct(encoded_feature)
        return  reconstructed_normed_src_emb


    def get_V(self, 
        normed_src_emb:Tensor, src_mask:Tensor, src_length:Tensor, 
        noise_scale:Tensor
        ) -> Tensor:
        """
        critic: V(s_t), estimate the expected future rewards (quality) of the state
        guide the noise flow.
        :param normed_src_emb: batched padded src tokens [batch, len, dim]
        :param src_mask: indicate valid token [batch, 1, len]
        :param src_length: [batch]
        :param noise scale: noise scale wrt the origin states [batch, len, dim]
        return: the estimate state-value on each token [batch, len, 1]
        """
        input = normed_src_emb + position_embedding(normed_src_emb)  # indicates the token position
        input += self.denoise_scale_linear(noise_scale)
        # input += make_timestep_vec(timestep, dim=input.shape[-1]).unsqueeze(dim=1) # expand position dimension
        input_feature = self.input_layer_norm(input)
        encoded_feature, _ = self.feature_model(input_feature, src_length, src_mask)
        encoded_feature = self.input_layer_norm(encoded_feature)

        v_estimate=self.critic(encoded_feature)
        return v_estimate


    def noise_by_scale(self, 
        normed_src_emb:Tensor, src_mask:Tensor, src_length:Tensor,
        noise_scale:Tensor):
        """
        noise_scale: the standard deviation of noising
        return: one-step noising by given standard deviation (noise scale). 
        """
        pert_normed_src_emb = torch.sqrt(1-noise_scale**2) * normed_src_emb \
            + noise_scale * torch.randn_like(normed_src_emb)
        
        return pert_normed_src_emb
    
    def get_score_function():
        return 

