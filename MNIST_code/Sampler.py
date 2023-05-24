## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class Sampler:

    def __init__(self, model, img_shape, sample_size):
        """
        Inputs:
            model - Neural network to use for modeling the energy function U_theta
            img_shape - Shape of the images to model
            sample_size - Batch size of the samples
        """
        super().__init__()
        self.model = model.to(device)
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = sample_size
        self.langevin_steps = torch.tensor(0).float()
        self.examples = torch.rand((sample_size,)+img_shape,device=device)*2-1
        self.weights = torch.ones((sample_size,),device=device)
        self.log_weights = torch.zeros((sample_size,),device=device)
        self.normalization = 0
        self.normal_0 = 0
        self.ce = 0
        self.old_energy = model(self.examples).clone().detach()

        
    def sample_new_exmps(self, steps, step_size,noise_level,batch_size,num_batch):
        """
        Function for getting a new batch of the walkers.
        Inputs:
            steps - Number of iterations in ULA (Unadjusted Langevin Dynamics)
            step_size - step size for ULA
        """
        for i in range(num_batch):
            
            # Construct a batch 
            batch = torch.multinomial(torch.ones((self.sample_size,)),batch_size,replacement = False).to(device)
            inp_imgs = self.examples.detach()[batch]
            weights = self.weights[batch]
        
            # Perform ULA and update weights
        
            self.examples[batch], new_log_weights = self.generate_samples(self.model, inp_imgs, steps=steps, step_size=step_size,noise_level = noise_level)
            self.langevin_steps = self.langevin_steps + steps

            # Update the weights of walkers that are not in the batch

            new_energy = self.model(self.examples).clone().detach()
            self.log_weights = self.log_weights+(-new_energy + self.old_energy)
            self.log_weights[batch] = self.log_weights[batch] + new_log_weights
            self.old_energy = new_energy
        
            with torch.no_grad():
                self.weights = self.log_weights.exp()
                self.normalization = self.normal_0*(self.weights.mean()) # Update the normalization constant
        
        return inp_imgs

    def generate_samples(self, model, inp_imgs, steps, step_size, noise_level, return_img_per_step=False):
        """
        Function for sampling images for a given model. 
        Inputs:
            model - Neural network to use for modeling the energy function U_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in ULA (Unadjusted Langevin Dynamics)
            step_size - step size for ULA
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before ULA: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input. 
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        
        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)
        
        # List for storing generations at each step (for later analysis)
        imgs_per_step = []
        log_weights = torch.zeros((inp_imgs.shape[0],),device=inp_imgs.device)
        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            inp_imgs.requires_grad = True
            noise.normal_(0, noise_level)
            # inp_imgs.data.add_(noise.data)
            # inp_imgs.data.clamp_(min=-1.0, max=1.0)
            
            # Part 2: calculate gradients for the current input.
            out_imgs = model(inp_imgs)
            out_imgs.sum().backward()
            # inp_imgs.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            old_inp_imgs_data = inp_imgs.data.clone().detach()
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data + noise.data)
            inp_imgs.grad = None
            inp_imgs.requires_grad = False
            
            # Make sure that our samples are within the domain [-1,1] in each dimension
            inp_imgs[inp_imgs>1] = old_inp_imgs_data[inp_imgs>1]
            inp_imgs[inp_imgs<-1] = old_inp_imgs_data[inp_imgs<-1]

            # Update the log of the weights A_k
            inp_imgs_copy = inp_imgs.clone().detach()
            old_inp_imgs_copy = old_inp_imgs_data.clone().detach()
            log_weights += self.log_weight_update(model, inp_imgs_copy, old_inp_imgs_copy, step_size) - self.log_weight_update(model, old_inp_imgs_copy, inp_imgs_copy, step_size)
            
            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())
        
        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs, log_weights
        
    def pure_generate_samples(self, model, inp_imgs, steps, step_size, noise_level, return_img_per_step=False):
        """
        Function for sampling images for a given model without updating their Jarzynski weights.
        Inputs:
            model - Neural network to use for modeling the energy function U_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in ULA (Unadjusted Langevin Dynamics)
            step_size - step size for ULA
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before ULA: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input. 
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        
        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)
        
        # List for storing generations at each step (for later analysis)
        imgs_per_step = []
        
        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            inp_imgs.requires_grad = True
            noise.normal_(0, noise_level)
            
            # Part 2: calculate gradients for the current input.
            out_imgs = model(inp_imgs)
            out_imgs.sum().backward()

            # Apply gradients to our current samples
            old_inp_imgs_data = inp_imgs.data.clone().detach()
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data + noise.data)
            inp_imgs.grad = None
            inp_imgs.requires_grad = False
            
            inp_imgs[inp_imgs>1] = old_inp_imgs_data[inp_imgs>1]
            inp_imgs[inp_imgs<-1] = old_inp_imgs_data[inp_imgs<-1]
            
            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())
        
        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs
        
        
    def log_weight_update(self,model, x, y, step_size):

        """
        Function for updating the log of the weights
        """
        
        x.requires_grad = True
        out_x = model(x)
        out_x.sum().backward()
        dU_x = x.grad.data
        dU_x = dU_x.detach()
        x.grad = None
        x.requires_grad = False
        loga = (-((y-x+step_size*dU_x)**2)/(4*step_size)).sum(dim=tuple(range(1,len(x.shape))))
        return loga.detach()
    
    def resample_multinomial(self):

        """
        With given weights and samples x, resample with the multinomial resampling scheme
        """

        new_index = torch.multinomial(self.weights,self.sample_size,replacement = True).to(device)
        self.examples = self.examples[new_index]
        self.weights = torch.ones((self.sample_size,),device=device)/self.sample_size
        self.log_weights = torch.zeros((self.sample_size,),device=device)
        self.normal_0 = self.normalization
        