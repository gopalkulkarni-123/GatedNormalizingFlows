from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from .layers import SharedDot, SharedDotPrior, Swish


class CondRealNVPFlow3D(nn.Module):
    def __init__(self, f_n_features, g_n_features,
                 weight_std=0.01, warp_inds=[0],
                 centered_translation=False, eps=1e-6):
        super(CondRealNVPFlow3D, self).__init__()
        self.f_n_features = 64 #f_n_features
        self.g_n_features = g_n_features
        self.weight_std = weight_std
        self.warp_inds = warp_inds
        self.keep_inds = [0, 1, 2, 3, 4, 5]
        self.centered_translation = centered_translation
        self.register_buffer('eps', torch.from_numpy(np.array([eps], dtype=np.float32)))
        for ind in self.warp_inds:
            self.keep_inds.remove(ind)

        self.T_mu_0 = nn.Sequential(OrderedDict([
            ('mu_sd0', SharedDot(len(self.keep_inds), self.f_n_features, 1)),
            ('mu_sd0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('mu_sd0_relu', nn.ReLU(inplace=True)),
            ('mu_sd1', SharedDot(self.f_n_features, self.f_n_features, 1)),
            ('mu_sd1_bn', nn.BatchNorm1d(self.f_n_features, affine=False))
        ]))

        self.T_mu_0_cond_w = nn.Sequential(OrderedDict([
            ('mu_sd1_film_w0', nn.Linear(self.g_n_features, self.f_n_features, bias=False)),
            ('mu_sd1_film_w0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('mu_sd1_film_w0_swish', Swish()),
            ('mu_sd1_film_w1', nn.Linear(self.f_n_features, self.f_n_features, bias=True))
        ]))

        self.T_mu_0_cond_b = nn.Sequential(OrderedDict([
            ('mu_sd1_film_b0', nn.Linear(self.g_n_features, self.f_n_features, bias=False)),
            ('mu_sd1_film_b0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('mu_sd1_film_b0_swish', Swish()),
            ('mu_sd1_film_b1', nn.Linear(self.f_n_features, self.f_n_features, bias=True))
        ]))

        self.T_mu_1 = nn.Sequential(OrderedDict([
            ('mu_sd1_relu', nn.ReLU(inplace=True)),
            ('mu_sd2', SharedDot(self.f_n_features, len(self.warp_inds), 1, bias=True))
        ]))

        with torch.no_grad():
            self.T_mu_0_cond_w[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_0_cond_w[-1].bias.data, 0.0)
            self.T_mu_0_cond_b[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_0_cond_b[-1].bias.data, 0.0)
            self.T_mu_1[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_1[-1].bias.data, 0.0)

        self.T_logvar_0 = nn.Sequential(OrderedDict([
            ('logvar_sd0', SharedDot(len(self.keep_inds), self.f_n_features, 1)),
            ('logvar_sd0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('logvar_sd0_relu', nn.ReLU(inplace=True)),
            ('logvar_sd1', SharedDot(self.f_n_features, self.f_n_features, 1)),
            ('logvar_sd1_bn', nn.BatchNorm1d(self.f_n_features, affine=False))
        ]))

        self.T_logvar_0_cond_w = nn.Sequential(OrderedDict([
            ('logvar_sd1_film_w0', nn.Linear(self.g_n_features, self.f_n_features, bias=False)),
            ('logvar_sd1_film_w0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('logvar_sd1_film_w0_swish', Swish()),
            ('logvar_sd1_film_w1', nn.Linear(self.f_n_features, self.f_n_features, bias=True))
        ]))

        self.T_logvar_0_cond_b = nn.Sequential(OrderedDict([
            ('logvar_sd1_film_b0', nn.Linear(self.g_n_features, self.f_n_features, bias=False)),
            ('logvar_sd1_film_b0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('logvar_sd1_film_b0_swish', Swish()),
            ('logvar_sd1_film_b1', nn.Linear(self.f_n_features, self.f_n_features, bias=True))
        ]))

        self.T_logvar_1 = nn.Sequential(OrderedDict([
            ('logvar_sd1_relu', nn.ReLU(inplace=True)),
            ('logvar_sd2', SharedDot(self.f_n_features, len(self.warp_inds), 1, bias=True))
        ]))

        with torch.no_grad():
            self.T_logvar_0_cond_w[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_0_cond_w[-1].bias.data, 0.0)
            self.T_logvar_0_cond_b[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_0_cond_b[-1].bias.data, 0.0)
            self.T_logvar_1[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_1[-1].bias.data, 0.0)

    def forward(self, p, g, mode='direct'):
        logvar = torch.zeros_like(p)
        mu = torch.zeros_like(p)

        logvar_0_cond_b = self.T_logvar_0_cond_b(g).unsqueeze(2)
        logvar_0 = self.T_logvar_0(p[:, self.keep_inds, :].contiguous())
        logvar_0_cond_w = self.T_logvar_0_cond_w(g).unsqueeze(2)

        epsilon_plus_exp_of_logvar_0_cond_w = torch.add(self.eps, torch.exp(logvar_0_cond_w))
        temp_logvar = epsilon_plus_exp_of_logvar_0_cond_w * logvar_0

        logvar[:, self.warp_inds, :] = nn.functional.softsign(self.T_logvar_1(temp_logvar + logvar_0_cond_b))

        #logvar[:, self.warp_inds, :] = nn.functional.softsign(self.T_logvar_1(
        #    torch.add(self.eps, torch.exp(self.T_logvar_0_cond_w(g).unsqueeze(2))) * self.T_logvar_0(p[:, self.keep_inds, :].contiguous()) + self.T_logvar_0_cond_b(g).unsqueeze(2)
        # ))

        mu_0_cond_b = self.T_mu_0_cond_b(g).unsqueeze(2)
        mu_0 = self.T_mu_0(p[:, self.keep_inds, :].contiguous())
        mu_0_cond_w = self.T_mu_0_cond_w(g).unsqueeze(2)
        
        epsilon_plus_exp_of_mu_0_cond_w = torch.add(self.eps, torch.exp(mu_0_cond_w))
        temp_mu = epsilon_plus_exp_of_mu_0_cond_w * mu_0
        
        mu[:, self.warp_inds, :] = self.T_mu_1(temp_mu + mu_0_cond_b)
        
        
        
        #mu[:, self.warp_inds, :] = self.T_mu_1(
        #    torch.add(self.eps, torch.exp(self.T_mu_0_cond_w(g).unsqueeze(2))) *
        #    self.T_mu_0(p[:, self.keep_inds, :].contiguous()) + self.T_mu_0_cond_b(g).unsqueeze(2)
        #)

        logvar = logvar.contiguous()
        mu = mu.contiguous()

        if mode == 'direct':
            p_out = torch.sqrt(torch.add(self.eps, torch.exp(logvar))) * p + mu
        elif mode == 'inverse':
            p_out = (p - mu) / torch.sqrt(torch.add(self.eps, torch.exp(logvar)))

        return p_out, mu, logvar

class CondRealNVPFlow3DTriple(nn.Module):
    def __init__(self, f_n_features, g_n_features, weight_std=0.02, pattern=0, centered_translation=False):
        super(CondRealNVPFlow3DTriple, self).__init__()
        self.f_n_features = f_n_features
        self.g_n_features = g_n_features
        self.weight_std = weight_std
        self.pattern = pattern
        self.centered_translation = centered_translation

        if pattern == 0:
            self.nvp1 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[0, 1, 2],
                                          centered_translation=centered_translation)
            self.nvp2 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[3, 4, 5],
                                          centered_translation=centered_translation)
            """ 
            self.nvp3 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[2],
                                          centered_translation=centered_translation)
            self.nvp4 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[3],
                                          centered_translation=centered_translation)
            
        elif pattern == 1:
            self.nvp1 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[0, 1],
                                          centered_translation=centered_translation)
            self.nvp2 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[0, 2],
                                          centered_translation=centered_translation)
            self.nvp3 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[1, 2],
                                          centered_translation=centered_translation)
            """
    def forward(self, p, g, mode='direct'):
        if mode == 'direct':
            p1, mu1, logvar1 = self.nvp1(p, g, mode=mode)
            p2, mu2, logvar2 = self.nvp2(p1, g, mode=mode)
            #p3, mu3, logvar3 = self.nvp3(p2, g, mode=mode)
        elif mode == 'inverse':
            p2, mu2, logvar2 = self.nvp2(p, g, mode=mode)
            p1, mu1, logvar1 = self.nvp1(p2, g, mode=mode)
            #p1, mu1, logvar1 = self.nvp1(p2, g, mode=mode)

        return [p1, p2], [mu1, mu2], [logvar1, logvar2]

class RealNVPFlow(nn.Module):
    def __init__(self, n_features, g_n_features, weight_std=0.01, warp_inds=[0], eps=1e-6):
        super(RealNVPFlow, self).__init__()
        self.n_features = n_features
        self.g_n_features = g_n_features
        self.weight_std = weight_std
        self.warp_inds = warp_inds
        self.keep_inds = list(np.arange(g_n_features))
        self.register_buffer('eps', torch.from_numpy(np.array([eps], dtype=np.float32)))
        for ind in self.warp_inds:
            self.keep_inds.remove(ind)

        self.T_mu_0 = nn.Sequential(OrderedDict([
            ('mu_mlp0', nn.Linear(len(self.keep_inds), self.n_features, bias=False)),
            ('mu_mlp0_bn', nn.BatchNorm1d(self.n_features)),
            ('mu_mlp0_swish', Swish()),
            ('mu_mlp1', nn.Linear(self.n_features, len(self.warp_inds), bias=True))
        ]))
        with torch.no_grad():
            self.T_mu_0[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_0[-1].bias.data, 0.0)

        self.T_logvar_0 = nn.Sequential(OrderedDict([
            ('logvar_mlp0', nn.Linear(len(self.keep_inds), self.n_features, bias=False)),
            ('logvar_mlp0_bn', nn.BatchNorm1d(self.n_features)),
            ('logvar_mlp0_swish', Swish()),
            ('logvar_mlp1', nn.Linear(self.n_features, len(self.warp_inds), bias=True))
        ]))
        with torch.no_grad():
            self.T_logvar_0[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_0[-1].bias.data, 0.0)

    def forward(self, g, mode='direct'):
        logvar = torch.zeros_like(g)
        mu = torch.zeros_like(g)

        logvar[:, self.warp_inds] = torch.log(torch.add(
            self.eps,
            torch.exp(self.T_logvar_0(g[:, self.keep_inds].contiguous()))
        ))
        mu[:, self.warp_inds] = self.T_mu_0(g[:, self.keep_inds].contiguous())

        logvar = logvar.contiguous()
        mu = mu.contiguous()

        if mode == 'direct':
            g_out = torch.exp(0.5 * logvar) * g + mu
        elif mode == 'inverse':
            g_out = torch.exp(-0.5 * logvar) * (g - mu)

        return g_out, mu, logvar

class RealNVPFlowCouple(nn.Module):
    def __init__(self, n_features, g_n_features, weight_std=0.01, pattern=0):
        super(RealNVPFlowCouple, self).__init__()
        self.n_features = n_features
        self.g_n_features = g_n_features
        self.weight_std = weight_std
        self.pattern = pattern

        if pattern == 0:
            self.nvp1 = RealNVPFlow(n_features, g_n_features,
                                    weight_std=weight_std, warp_inds=list(np.arange(g_n_features)[::2]))
            self.nvp2 = RealNVPFlow(n_features, g_n_features,
                                    weight_std=weight_std, warp_inds=list(np.arange(g_n_features)[1::2]))
        elif pattern == 1:
            self.nvp1 = RealNVPFlow(n_features, g_n_features,
                                    weight_std=weight_std, warp_inds=list(np.arange(g_n_features)[:g_n_features // 2]))
            self.nvp2 = RealNVPFlow(n_features, g_n_features,
                                    weight_std=weight_std, warp_inds=list(np.arange(g_n_features)[g_n_features // 2:]))

    def forward(self, g, mode='direct'):
        if mode == 'direct':
            g1, mu1, logvar1 = self.nvp1(g, mode=mode)
            g2, mu2, logvar2 = self.nvp2(g1, mode=mode)
        elif mode == 'inverse':
            g2, mu2, logvar2 = self.nvp2(g, mode=mode)
            g1, mu1, logvar1 = self.nvp1(g2, mode=mode)

        return [g1, g2], [mu1, mu2], [logvar1, logvar2]

class ConditionalPrior(nn.Module):
    def __init__(self, g_n_features_prior, c_n_features_prior,
                 weight_std=0.01, warp_inds=[0],
                 centered_translation=False, eps=1e-6):
        super(ConditionalPrior, self).__init__()
        self.g_n_features_prior = g_n_features_prior
        self.c_n_features_prior = c_n_features_prior
        self.weight_std = weight_std
        self.warp_inds = warp_inds
        self.keep_inds = list(np.arange(32))
        self.centered_translation = centered_translation
        self.register_buffer('eps', torch.from_numpy(np.array([eps], dtype=np.float32)))
        for ind in self.warp_inds:
            self.keep_inds.remove(ind)

        self.T_mu_0 = nn.Sequential(OrderedDict([
            ('mu_sd0', SharedDotPrior(len(self.keep_inds), self.g_n_features_prior)),
            ('mu_sd0_bn', nn.BatchNorm1d(self.g_n_features_prior)),
            ('mu_sd0_relu', nn.ReLU(inplace=True)),
            ('mu_sd1', SharedDotPrior(self.g_n_features_prior, self.g_n_features_prior)),
            ('mu_sd1_bn', nn.BatchNorm1d(self.g_n_features_prior, affine=False))
        ]))

        self.T_mu_0_cond_w = nn.Sequential(OrderedDict([
            ('mu_sd1_film_w0', nn.Linear(self.c_n_features_prior, self.g_n_features_prior, bias=False)),
            ('mu_sd1_film_w0_bn', nn.BatchNorm1d(self.g_n_features_prior)),
            ('mu_sd1_film_w0_swish', Swish()),
            ('mu_sd1_film_w1', nn.Linear(self.g_n_features_prior, self.g_n_features_prior, bias=True))
        ]))

        self.T_mu_0_cond_b = nn.Sequential(OrderedDict([
            ('mu_sd1_film_b0', nn.Linear(self.c_n_features_prior, self.g_n_features_prior, bias=False)),
            ('mu_sd1_film_b0_bn', nn.BatchNorm1d(self.g_n_features_prior)),
            ('mu_sd1_film_b0_swish', Swish()),
            ('mu_sd1_film_b1', nn.Linear(self.g_n_features_prior, self.g_n_features_prior, bias=True))
        ]))

        self.T_mu_1 = nn.Sequential(OrderedDict([
            ('mu_sd1_relu', nn.ReLU(inplace=True)),
            ('mu_sd2', SharedDotPrior(self.g_n_features_prior, len(self.warp_inds), bias=True))
        ]))

        with torch.no_grad():
            self.T_mu_0_cond_w[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_0_cond_w[-1].bias.data, 0.0)
            self.T_mu_0_cond_b[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_0_cond_b[-1].bias.data, 0.0)
            self.T_mu_1[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_1[-1].bias.data, 0.0)

        self.T_logvar_0 = nn.Sequential(OrderedDict([
            ('logvar_sd0', SharedDotPrior(len(self.keep_inds), self.g_n_features_prior)),
            ('logvar_sd0_bn', nn.BatchNorm1d(self.g_n_features_prior)),
            ('logvar_sd0_relu', nn.ReLU(inplace=True)),
            ('logvar_sd1', SharedDotPrior(self.g_n_features_prior, self.g_n_features_prior)),
            ('logvar_sd1_bn', nn.BatchNorm1d(self.g_n_features_prior, affine=False))
        ]))

        self.T_logvar_0_cond_w = nn.Sequential(OrderedDict([
            ('logvar_sd1_film_w0', nn.Linear(self.c_n_features_prior, self.g_n_features_prior, bias=False)),
            ('logvar_sd1_film_w0_bn', nn.BatchNorm1d(self.g_n_features_prior)),
            ('logvar_sd1_film_w0_swish', Swish()),
            ('logvar_sd1_film_w1', nn.Linear(self.g_n_features_prior, self.g_n_features_prior, bias=True))
        ]))

        self.T_logvar_0_cond_b = nn.Sequential(OrderedDict([
            ('logvar_sd1_film_b0', nn.Linear(self.c_n_features_prior, self.g_n_features_prior, bias=False)),
            ('logvar_sd1_film_b0_bn', nn.BatchNorm1d(self.g_n_features_prior)),
            ('logvar_sd1_film_b0_swish', Swish()),
            ('logvar_sd1_film_b1', nn.Linear(self.g_n_features_prior, self.g_n_features_prior, bias=True))
        ]))

        self.T_logvar_1 = nn.Sequential(OrderedDict([
            ('logvar_sd1_relu', nn.ReLU(inplace=True)),
            ('logvar_sd2', SharedDotPrior(self.g_n_features_prior, len(self.warp_inds), bias=True))
        ]))

        with torch.no_grad():
            self.T_logvar_0_cond_w[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_0_cond_w[-1].bias.data, 0.0)
            self.T_logvar_0_cond_b[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_0_cond_b[-1].bias.data, 0.0)
            self.T_logvar_1[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_1[-1].bias.data, 0.0)

    def forward(self, g, c, mode='direct'):
        logvar = torch.zeros_like(g)
        mu = torch.zeros_like(g)

        logvar_0_cond_b = self.T_logvar_0_cond_b(c)
        logvar_0 = self.T_logvar_0(g[:, self.keep_inds].contiguous())
        logvar_0_cond_w = self.T_logvar_0_cond_w(c)

        epsilon_plus_exp_of_logvar_0_cond_w = torch.add(self.eps, torch.exp(logvar_0_cond_w))
        temp_logvar = epsilon_plus_exp_of_logvar_0_cond_w * logvar_0

        logvar[:, self.warp_inds] = nn.functional.softsign(self.T_logvar_1(temp_logvar + logvar_0_cond_b))

        #logvar[:, self.warp_inds] = nn.functional.softsign(self.T_logvar_1(
        #    torch.add(self.eps, torch.exp(self.T_logvar_0_cond_w(c).unsqueeze(1))) *
        #    self.T_logvar_0(g[:, self.keep_inds].contiguous()) + self.T_logvar_0_cond_b(c).unsqueeze(1)
        #))

        mu_0_cond_b = self.T_mu_0_cond_b(c)
        mu_0 = self.T_mu_0(g[:, self.keep_inds].contiguous())
        mu_0_cond_w = self.T_mu_0_cond_w(c)
        
        epsilon_plus_exp_of_mu_0_cond_w = torch.add(self.eps, torch.exp(mu_0_cond_w))
        temp_mu = epsilon_plus_exp_of_mu_0_cond_w * mu_0
        
        mu[:, self.warp_inds] = self.T_mu_1(temp_mu + mu_0_cond_b)
        
        
        #mu[:, self.warp_inds] = self.T_mu_1(
        #    torch.add(self.eps, torch.exp(self.T_mu_0_cond_w(c).unsqueeze(1))) *
        #    self.T_mu_0(g[:, self.keep_inds].contiguous()) + self.T_mu_0_cond_b(c).unsqueeze(1)
        #)

        logvar = logvar.contiguous()
        mu = mu.contiguous()

        if mode == 'direct':
            p_out = torch.sqrt(torch.add(self.eps, torch.exp(logvar))) * g + mu
        elif mode == 'inverse':
            p_out = (g - mu) / torch.sqrt(torch.add(self.eps, torch.exp(logvar)))

        return p_out, mu, logvar
    
class ConditionalPriorCouple(nn.Module):
    def __init__(self, g_n_features_prior, c_n_features_prior, latent_space_sample, weight_std=0.01, pattern=0):
        super(ConditionalPriorCouple, self).__init__()
        self.g_n_features_prior = g_n_features_prior
        self.c_n_features = c_n_features_prior
        self.latent_space_sample = latent_space_sample
        self.weight_std = weight_std
        self.pattern = pattern

        if pattern == 0:
            self.nvp1 = ConditionalPrior(g_n_features_prior, c_n_features_prior,
                                    weight_std=weight_std, warp_inds=list(np.arange(self.latent_space_sample)[::2]))
            self.nvp2 = ConditionalPrior(g_n_features_prior, c_n_features_prior,
                                    weight_std=weight_std, warp_inds=list(np.arange(self.latent_space_sample)[1::2]))
        elif pattern == 1:
            self.nvp1 = ConditionalPrior(g_n_features_prior, c_n_features_prior,
                                    weight_std=weight_std, warp_inds=list(np.arange(self.latent_space_sample)[:self.latent_space_sample // 2]))
            self.nvp2 = ConditionalPrior(g_n_features_prior, c_n_features_prior,
                                    weight_std=weight_std, warp_inds=list(np.arange(self.latent_space_sample)[self.latent_space_sample // 2:]))

    def forward(self, g, c, mode='direct'):
        if mode == 'direct':
            g1, mu1, logvar1 = self.nvp1(g, c, mode=mode)
            g2, mu2, logvar2 = self.nvp2(g1, c, mode=mode)
        elif mode == 'inverse':
            g2, mu2, logvar2 = self.nvp2(g, c, mode=mode)
            g1, mu1, logvar1 = self.nvp1(g2, c, mode=mode)

        return [g1, g2], [mu1, mu2], [logvar1, logvar2]