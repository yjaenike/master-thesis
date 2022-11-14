'''A wrapper class for scheduled optimizer '''
import numpy as np

class ScheduledOptim():
    """A simple wrapper class for learning rate scheduling
    
    Attributes
    ----------
    _optimizer (torch.Optim): The Optimizer used during training
    lr_mul (flaot): learning rate multiplicator
    d_model (int): dimensionality of the model
    n_warmup_steps (int) Number of warmup steps 
    n_steps (int): total number of steps taken
    
    Methods
    -------
    step_and_update_lr(self): performs one step in the optimizer and updates the learning rate
    zero_grad(self): Zero out the gradients with the inner optimizer
    _get_lr_scale(self): 
    _update_learning_rate(self): updates the learning step with each step
    """

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        """ Step with the inner optimizer """
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        """ Zero out the gradients with the inner optimizer """
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        """ Calculates factor to scale the learning rate """
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        """ Learning rate scheduling per step """

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

