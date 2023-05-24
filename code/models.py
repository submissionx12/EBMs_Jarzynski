
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np

class GMM_teacher(nn.Module): # teacher model 

    def __init__(self, dim, device, mix_weigths = [0.2, 0.2, 0.3, 0.3], stds = [5,5,8,8]):
        super().__init__()

        self.device = device
        self.dim = dim
        self.mean = torch.randn(4, dim, device=device)*5
        self.stds = torch.tensor([stds], device=device)

        if dim==2:
          dist = 40.0
        else:
          dist=20.0

        self.mean[0,:2] = self.mean[0,:2] + torch.tensor([-dist,dist],device=device)
        self.mean[1,:2] = self.mean[1,:2] + torch.tensor([dist,dist],device=device)
        self.mean[2,:2] = self.mean[2,:2] + torch.tensor([dist,-dist],device=device)
        self.mean[3,:2] = self.mean[3,:2] + torch.tensor([-dist,-dist],device=device)

        # log of the standard deviation (diagonal)
        self.log_std = torch.zeros(4, dim,device=device)
        self.log_std[0,:] = np.log(np.sqrt(stds[0]))
        self.log_std[1,:] = np.log(np.sqrt(stds[1]))
        self.log_std[2,:] = np.log(np.sqrt(stds[2]))
        self.log_std[3,:] = np.log(np.sqrt(stds[3]))

#         self.log_std = nn.Parameter(self.log_std)
        self.mix_logits = torch.tensor(mix_weigths).log() # log of the weights 
        self.cov_inv = torch.eye(dim,device=device)
        self.cov_inv[0:int(dim/2)]*=2
        self.cov_inv[-3:]*=7
        self.cov_inv[0:int(dim/4)]*=4.5

        # sampler for all the modes 
        self.gmm1 = MultivariateNormal(self.mean[0,:],((self.log_std[0,0].exp())**2)*self.cov_inv) 
        self.gmm2 = MultivariateNormal(self.mean[1,:],((self.log_std[1,0].exp())**2)*self.cov_inv)
        self.gmm3 = MultivariateNormal(self.mean[2,:],((self.log_std[2,0].exp())**2)*self.cov_inv)
        self.gmm4 = MultivariateNormal(self.mean[3,:],((self.log_std[3,0].exp())**2)*self.cov_inv)


    def forward(self, X): # evaluate the score
        energy = (X.unsqueeze(1) - self.mean) ** 2 / (2 * (2 * self.log_std).exp()) + np.log(
            2 * np.pi) / 2. + self.log_std
        log_prob = -energy.sum(dim=-1)
        mix_probs = F.log_softmax(self.mix_logits,dim=0)
        log_prob += mix_probs
        log_prob = torch.logsumexp(log_prob, dim=-1)
        return -log_prob

    def sample(self, n): # sampler for all the modes 
        s = torch.randperm(int(n))
        N = torch.multinomial(self.mix_logits.exp(), int(n), replacement=True).bincount()
        sample_gmm1 = self.gmm1.sample((N[0],))
        sample_gmm2 = self.gmm2.sample((N[1],))
        sample_gmm3 = self.gmm3.sample((N[2],))
        sample_gmm4 = self.gmm4.sample((N[3],))
        return torch.cat((sample_gmm1,sample_gmm2,sample_gmm3,sample_gmm4),0)[s].to(self.device)

    def requires_grad(self, bool):
        for p in self.parameters():
            p.requires_grad = bool




class Abstract_teacher(nn.Module): # teacher model 

    def __init__(self, 
                 device, 
                 weights=torch.tensor([0.2, 0.2, 0.3, 0.3]), 
                 means = torch.zeros(4, 2), 
                 stds = torch.ones(4, 2)
                 ):
        super().__init__()

        self.device = device
        self.means = means
        self.stds = stds 
        self.weights = weights

        assert self.means.shape == self.stds.shape, "Mean and Std shapes don't match"
        assert self.means.shape[0] == self.weights.shape[0], "Number of modes and weigths don't match"

        self.dim = self.means.shape[-1]

        # sampler for all the modes 
        self.gmm1 = MultivariateNormal(self.means[0,:],torch.diag(self.stds[0,:]**2)) 
        self.gmm2 = MultivariateNormal(self.means[1,:],torch.diag(self.stds[1,:]**2)) 
        self.gmm3 = MultivariateNormal(self.means[2,:],torch.diag(self.stds[2,:]**2)) 
        self.gmm4 = MultivariateNormal(self.means[3,:],torch.diag(self.stds[3,:]**2)) 


    def forward(self, X): # evaluate the score

        assert X.shape[-1]==self.dim, f"Input dimension {X.shape[-1]} doesnt match Teacher dimension {self.dim}."
        
        energy = 0.5 * (X[None,...] - self.means[:,None,:])**2 / self.stds[:, None, :] + 0.5 * torch.log( 2*np.pi * self.stds[:,None,:]**2)#(4,N,dim)
        likelihood = ((-energy.sum(dim=-1)).exp() * self.weights[:,None]).sum(dim=0)
        return - likelihood.log()

    def sample(self, n): # sampler for all the modes 
        s = torch.randperm(int(n))
        N = torch.multinomial(self.weigths, int(n), replacement=True).bincount()
        sample_gmm1 = self.gmm1.sample((N[0],))
        sample_gmm2 = self.gmm2.sample((N[1],))
        sample_gmm3 = self.gmm3.sample((N[2],))
        sample_gmm4 = self.gmm4.sample((N[3],))
        return torch.cat((sample_gmm1,sample_gmm2,sample_gmm3,sample_gmm4),0)[s].to(self.device)

    def requires_grad(self, bool):
        for p in self.parameters():
            p.requires_grad = bool


class SimpleTeacher(GMM_teacher): 
    def __init__(self, dim, device):
        super().__init__(dim, device, mix_weigths=[2/10,2/10,3/10,3/10], )

class HardTeacher(GMM_teacher):
    pass

class GMM(nn.Module): # MLP for the energy

    def __init__(self,dim,hidden_dim, device):
        super().__init__()
        self.dim = dim
        self.device = device
        self.neuralnet = nn.Sequential(
        nn.Linear(dim,hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim,hidden_dim), nn.SiLU(),
#         nn.Dropout(p=0.5),
#         nn.Linear(hidden_dim*2,hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim,hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim,1))
        self.var=8.0
        self.A=0.
        self.A=torch.tensor(self.A,device=self.device)
        self.A=nn.Parameter(self.A)

    def forward(self, X):
        #initially a gaussian in zero to ensure that walkers are sampled from \rho_0
        energy = self.neuralnet(X) + 2*(self.A.exp())/(1+self.A.exp())*(X**2/2/self.var).sum(dim=1,keepdim = True)
        return energy
    
    def log_weight_update(self, x, y, step_size):

        x.requires_grad = True
        out_x = self.forward(x)
        out_x.sum().backward()
        dU_x = x.grad.data
        dU_x = dU_x.detach()
        x.grad = None
        x.requires_grad = False
        loga = (-((y-x+step_size*dU_x)**2)/(4*step_size)).sum(dim=tuple(range(1,len(x.shape)))) - self.forward(x).detach().squeeze()
        return loga
        
    def requires_grad(self, bool):
        for p in self.parameters():
            p.requires_grad = bool

    def sample(self,n):
        self.dis = MultivariateNormal(torch.zeros(self.dim),self.var*torch.eye(self.dim))
        return self.dis.sample((int(n),)).to(self.device)



class Walkers():

    def __init__(self, init_data, hx, device, clip_lim):
        self.hx = hx
        self.clip_lim = clip_lim
        self.sigma = torch.sqrt(torch.tensor([2*hx])).to(device) # 'diff' in original code
        self.device = device
        self.walkers = init_data
        self.old_walkers = self.walkers.data.clone().detach()

    def requires_grad(self, bool):
        self.old_walkers.requires_grad = bool
        self.walkers.requires_grad = bool

    def clean_autograd(self):
        self.walkers.grad.detach_()
        self.walkers.grad.zero_()
        self.requires_grad(False)

    def memorize(self):
        self.old_walkers = self.walkers.data.clone().detach()
        self.old_walkers.requires_grad = False

    def clip(self):
        self.walkers[self.walkers>self.clip_lim] = self.old_walkers[self.walkers>self.clip_lim]
        self.walkers[self.walkers<-self.clip_lim] = self.old_walkers[self.walkers<-self.clip_lim]

    def Langevin_step(self, model: GMM):
        model.requires_grad(False)
        self.requires_grad(True)
        energy = model(self.walkers)
        energy.sum().backward()
        self.memorize() # right before the Langevin

        noise = self.sigma * torch.randn_like(self.walkers, device=self.device)
        self.walkers.data.add_(
            -self.hx * self.walkers.grad.data + noise
        )

        self.clean_autograd()
        self.clip()



class WeightedWalkers(Walkers):

    def __init__(self, 
                 init_data, 
                 hx, 
                 device, 
                 clip_lim, 
                 use_resampling,
                 max_var,
                 ):
        super().__init__(init_data, hx, device, clip_lim)
        self.use_resampling = use_resampling
        self.max_var = max_var
        self.log_weights = torch.zeros(self.walkers.shape[0], device=device, requires_grad=False)

    def update_weights(self, delta):
        self.log_weights += delta

    def compute_delta(self, model):
        #def log_weight_update(self, x, y, step_size):

        self.requires_grad(True)
        out_x = model(self.old_walkers)
        out_x.sum().backward()
        dU_x = self.old_walkers.grad.data
        dU_x = dU_x.detach()
        self.old_walkers.grad = None
        self.old_walkers.requires_grad = False

        loga = (-((self.walkers-self.old_walkers+self.hx*dU_x)**2)/(4*self.hx)).sum(dim=tuple(range(1,len(self.old_walkers.shape)))) - model(self.old_walkers).detach().squeeze()
        self.requires_grad(False)
        return loga
    
    def weight_update_2(self, model):
        raise NotImplementedError

    def get_normalized_weigths(self):
        """
        Returns the L1-normalized weigths associated to the walkers.
        """
        return F.normalize(
            self.log_weights.clone().detach().exp(), 
            p=1.0, 
            dim=-1
        )

    def resample(self):
        if not self.use_resampling:
            pass 
        else:
            if self.log_weights.std()>self.max_var:
                self.walkers = self.walkers[
                    torch.multinomial(
                        self.log_weights.exp(), 
                        self.n_walkers, 
                        replacement=True
                        )
                ]
                self.log_weights = torch.zeros_like(self.log_weights)