import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMBlock(nn.Module):
    def __init__(self, out_channels=None):
        super(FiLMBlock, self).__init__()
        self.out_channels = out_channels
        self.mu = nn.Linear(256, out_channels)
        self.sigma = nn.Linear(256, out_channels)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        mu = self.mu(x)
        sigma = self.sigma(x)
        return self.activation(mu), self.activation(sigma)


class FiLMImplement(nn.Module):
    def __init__(self, window):
        super(FiLMImplement, self).__init__()
        self.window = window

    def broadcast_2d(self, x):
        return torch.broadcast_to(x, (self.window, self.window))

    def forward(self, x, mu, sigma):
        mu_broadcast = torch.stack(tuple(map(self.broadcast_2d, mu.squeeze(0))), dim=0)
        sigma_broadcast = torch.stack(tuple(map(self.broadcast_2d, sigma.squeeze(0))), dim=0)
        x = sigma_broadcast * x + mu_broadcast
        return x


################################# THE ENCODERS

class Encoder(nn.Module):
    '''
    simple encoder with a single hidden dense layer (ReLU activation)
    and linear projections to the diag-Gauss parameters
    '''

    def __init__(self, latent_size=256, nc=3, width=1):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.nc = nc
        self.width = width

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8 * self.width, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8 * self.width, out_channels=8 * self.width, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8 * self.width, out_channels=16 * self.width, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(16 * self.width, out_channels=16 * self.width, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(16 * self.width, out_channels=32 * self.width, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(32 * self.width, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.enc_linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU())

        self.fc_mu = nn.Linear(512, latent_size)
        self.fc_var = nn.Linear(512, latent_size)

    def forward(self, x, helo=None):
        # block 1
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)

        # block 2
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)
        x = F.leaky_relu(x)

        # block 3
        x = self.conv5(x)
        x = F.leaky_relu(x)
        x = self.conv6(x)
        x = F.leaky_relu(x)
        x = x.view(x.shape[0], -1)
        x = self.enc_linear(x)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

class YotoEncoder(nn.Module):
    '''
    simple encoder with a single hidden dense layer (ReLU activation)
    and linear projections to the diag-Gauss parameters
    '''

    def __init__(self, latent_size=256, nc=3, width=1):
        super(YotoEncoder, self).__init__()
        self.latent_size = latent_size
        self.nc = nc
        self.width = width

        self.film1 = FiLMBlock(8 * self.width)
        self.film2 = FiLMBlock(16 * self.width)
        self.film3 = FiLMBlock(32)
        self.filmimp1 = FiLMImplement(16)
        self.filmimp2 = FiLMImplement(8)
        self.filmimp3 = FiLMImplement(4)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8 * self.width, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8 * self.width, out_channels=8 * self.width, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8 * self.width, out_channels=16 * self.width, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(16 * self.width, out_channels=16 * self.width, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(16 * self.width, out_channels=32 * self.width, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(32 * self.width, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.enc_linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU())

        self.fc_mu = nn.Linear(512, latent_size)
        self.fc_var = nn.Linear(512, latent_size)

    def forward(self, x, helo=None):
        # block 1
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        mu, sigma = self.film1(helo)
        x = self.filmimp1(x, mu, sigma)
        x = F.leaky_relu(x)

        # block 2
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)
        mu2, sigma2 = self.film2(helo)
        x = self.filmimp2(x, mu2, sigma2)
        x = F.leaky_relu(x)

        # block 3
        x = self.conv5(x)
        x = F.leaky_relu(x)
        x = self.conv6(x)
        mu3, sigma3 = self.film3(helo)
        x = self.filmimp3(x, mu3, sigma3)
        x = F.leaky_relu(x)
        x = x.view(x.shape[0], -1)
        x = self.enc_linear(x)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

def SHAPESEncoder(width=1):
    return Encoder(256, 3, width)

def SHAPESYotoEncoder(width=1):
    return YotoEncoder(256, 3, width)

######################### DECODER

class Decoder(nn.Module):
    '''
    simple decoder: single dense hidden layer (ReLU activation) followed by
    output layer with a sigmoid to squish values
    '''

    def __init__(self, latent_size=256, nc=3, width=1):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.nc = nc
        self.width = width
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
        )

        self.activation = nn.Sigmoid()

        self.conv1U = nn.ConvTranspose2d(32, out_channels=32 * self.width, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2U = nn.ConvTranspose2d(32 * self.width, out_channels=16 * self.width, kernel_size=3, stride=1, padding=1,
                                         output_padding=0)
        self.conv3U = nn.ConvTranspose2d(16 * self.width, out_channels=16 * self.width, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)
        self.conv4U = nn.ConvTranspose2d(16 * self.width, out_channels=8 * self.width, kernel_size=3, stride=1, padding=1,
                                         output_padding=0)
        self.conv5U = nn.ConvTranspose2d(8 * self.width, out_channels=8 * self.width, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)
        self.conv6U = nn.ConvTranspose2d(8 * self.width, out_channels=3, kernel_size=3, stride=1, padding=1, output_padding=0)

    def forward(self, x, helo=None):
        x = self.decoder_input(x)
        x = x.view(-1, 32, 4, 4)

        # block 1
        x = self.conv1U(x)
        x = F.leaky_relu(x)
        x = self.conv2U(x)
        x = F.leaky_relu(x)

        # block 2
        x = self.conv3U(x)
        x = F.leaky_relu(x)
        x = self.conv4U(x)
        x = F.leaky_relu(x)

        # block 3
        x = self.conv5U(x)
        x = F.leaky_relu(x)
        x = self.conv6U(x)
        return self.activation(x)


class YotoDecoder(nn.Module):
    '''
    simple decoder: single dense hidden layer (ReLU activation) followed by
    output layer with a sigmoid to squish values
    '''

    def __init__(self, latent_size=256, nc=3, width=1):
        super(YotoDecoder, self).__init__()
        self.latent_size = latent_size
        self.nc = nc
        self.width = width
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
        )

        self.activation = nn.Sigmoid()

        self.film1U = FiLMBlock(16 * self.width)
        self.film2U = FiLMBlock(8 * self.width)
        self.film3U = FiLMBlock(3)
        self.filmimp1U = FiLMImplement(8)
        self.filmimp2U = FiLMImplement(16)
        self.filmimp3U = FiLMImplement(32)

        self.conv1U = nn.ConvTranspose2d(32, out_channels=32 * self.width, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2U = nn.ConvTranspose2d(32 * self.width, out_channels=16 * self.width, kernel_size=3, stride=1, padding=1,
                                         output_padding=0)
        self.conv3U = nn.ConvTranspose2d(16 * self.width, out_channels=16 * self.width, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)
        self.conv4U = nn.ConvTranspose2d(16 * self.width, out_channels=8 * self.width, kernel_size=3, stride=1, padding=1,
                                         output_padding=0)
        self.conv5U = nn.ConvTranspose2d(8 * self.width, out_channels=8 * self.width, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)
        self.conv6U = nn.ConvTranspose2d(8 * self.width, out_channels=3, kernel_size=3, stride=1, padding=1, output_padding=0)

    def forward(self, x, helo=None):
        x = self.decoder_input(x)
        x = x.view(-1, 32, 4, 4)

        # block 1
        x = self.conv1U(x)
        x = F.leaky_relu(x)
        x = self.conv2U(x)
        mu11, sigma11 = self.film1U(helo)
        x = self.filmimp1U(x, mu11, sigma11)
        x = F.leaky_relu(x)

        # block 2
        x = self.conv3U(x)
        x = F.leaky_relu(x)
        x = self.conv4U(x)
        mu22, sigma22 = self.film2U(helo)
        x = self.filmimp2U(x, mu22, sigma22)
        x = F.leaky_relu(x)

        # block 3
        x = self.conv5U(x)
        x = F.leaky_relu(x)
        x = self.conv6U(x)
        mu33, sigma33 = self.film3U(helo)
        x = self.filmimp3U(x, mu33, sigma33)
        return self.activation(x)

def SHAPESDecoder(width=1):
    return Decoder(256, 3, width)

def SHAPESYotoDecoder(width=1):
    return YotoDecoder(256, 3, width)