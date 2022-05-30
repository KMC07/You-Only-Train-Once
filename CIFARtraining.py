import torch
import torch.nn.functional as F
from itertools import chain
from tqdm.autonotebook import tqdm
import numpy as np
from torch import optim
from device import CUDADEVICE
import pandas as pd
from scipy.stats import loguniform


def sample(mu, logvar):
    std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
    eps = torch.randn_like(std)  # random ~ N(0, 1)
    return eps.mul(std).add_(mu)


def trainBVAEFixedCIFAR(trainloader, encoder, decoder, beta=1, epochs=100, training_dir='./trainingResults/CIFAR10FixedWeights.csv', enc_dir='./weights/encCIFAR10FixedWeights.weights', dec_dir='./weights/decCIFAR10FixedWeights.weights'):
    # parameters
    beta = beta
    nEpoch = epochs

    KL_loss = []
    recon_loss = []
    full_loss = []

    # construct the encoder, decoder and optimiser
    enc = encoder.to(CUDADEVICE)
    dec = decoder.to(CUDADEVICE)
    optimizer = optim.Adam(chain(enc.parameters(), dec.parameters()), lr=1e-4, weight_decay=1e-5)

    for epoch in range(nEpoch):
        losses = []
        trainloader = tqdm(trainloader)

        for i, data in enumerate(trainloader, 0):
            inputs, _ = data
            inputs, _ = inputs.to(CUDADEVICE), _.to(CUDADEVICE)
            optimizer.zero_grad()

            mu, log_sigma2 = enc(inputs)
            z = sample(mu, log_sigma2)
            outputs = dec(z)

            recon = F.binary_cross_entropy(outputs, inputs, reduction='sum') / inputs.shape[0]

            kl_diverge = 0.5 * torch.mean(
                torch.pow(mu, 2) + torch.pow(log_sigma2, 2) - torch.log(torch.pow(log_sigma2, 2)) - 1)

            loss = recon + beta * kl_diverge
            loss.backward()
            optimizer.step()

            # keep track of the loss and update the stats
            losses.append(loss.item())
            trainloader.set_postfix(loss=np.mean(losses), epoch=epoch)
        KL_loss.append(kl_diverge.data)
        recon_loss.append(recon.data)
        full_loss.append(loss.data)

    KL_numpy = []
    recon_np = []
    full_np = []
    for i in range(len(KL_loss)):
        KL_numpy.append(KL_loss[i].data.cpu().detach().numpy())
        recon_np.append(recon_loss[i].data.cpu().detach().numpy())
        full_np.append(full_loss[i].data.cpu().detach().numpy())

    data = {'KL_numpy': KL_numpy, 'recon_np': recon_np, 'full_np': full_np}

    # save the training loss values
    df = pd.DataFrame.from_dict(data)
    df.to_csv(training_dir, index=False)

    # save the model weights
    torch.save(enc.state_dict(), enc_dir)
    torch.save(dec.state_dict(), dec_dir)

    return KL_numpy, recon_np, full_np


def trainBVAEYotoCIFAR(trainloader, encoder, decoder, epochs=100, training_dir='./trainingResults/CIFAR10Yoto.csv', enc_dir='./weights/encCIFAR10Yoto.weights', dec_dir='./weights/decCIFAR10Yoto.weights'):
    # parameters
    nEpoch = epochs

    # construct the encoder, decoder and optimiser
    enc = encoder.to(CUDADEVICE)
    dec = decoder.to(CUDADEVICE)
    optimizer = optim.Adam(chain(enc.parameters(), dec.parameters()), lr=1e-4, weight_decay=1e-5)

    KL_loss = []
    recon_loss = []
    full_loss = []

    for epoch in range(nEpoch):
        losses = []
        trainloader = tqdm(trainloader)

        for i, data in enumerate(trainloader, 0):
            inputs, _ = data
            inputs, _ = inputs.to(CUDADEVICE), _.to(CUDADEVICE)
            optimizer.zero_grad()
            beta_initial = loguniform.rvs(0.125, 512, size=1)

            beta = np.float32(beta_initial[0]).tolist()

            beta2 = torch.tensor([1 * beta], requires_grad=False)
            beta2 = torch.broadcast_to(beta2, (1, 256)).to(CUDADEVICE)
            mu, log_sigma2 = enc(inputs, beta2)
            z = sample(mu, log_sigma2)
            outputs = dec(z, beta2)

            recon = F.binary_cross_entropy(outputs, inputs, reduction='sum') / inputs.shape[0]

            kl_diverge = 0.5 * torch.mean(
                torch.pow(mu, 2) + torch.pow(log_sigma2, 2) - torch.log(torch.pow(log_sigma2, 2)) - 1)

            loss = recon + beta * kl_diverge
            loss.backward()
            optimizer.step()

            # keep track of the loss and update the stats
            losses.append(loss.item())
            trainloader.set_postfix(loss=np.mean(losses), epoch=epoch)
        KL_loss.append(kl_diverge.data)
        recon_loss.append(recon.data)
        full_loss.append(loss.data)

    KL_numpy = []
    recon_np = []
    full_np = []
    for i in range(len(KL_loss)):
        KL_numpy.append(KL_loss[i].data.cpu().detach().numpy())
        recon_np.append(recon_loss[i].data.cpu().detach().numpy())
        full_np.append(full_loss[i].data.cpu().detach().numpy())

    data = {'KL_numpy': KL_numpy, 'recon_np': recon_np, 'full_np': full_np}

    #save the training loss values
    df = pd.DataFrame.from_dict(data)
    df.to_csv(training_dir, index=False)

    #save the model weights
    torch.save(enc.state_dict(), enc_dir)
    torch.save(dec.state_dict(), dec_dir)

    return KL_numpy, recon_np, full_np

