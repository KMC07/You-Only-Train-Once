from CIFAR10Models import CIFAR10Encoder, CIFAR10Decoder, CIFAR10YotoEncoder, CIFAR10YotoDecoder
from SHAPESModels import SHAPESEncoder, SHAPESDecoder, SHAPESYotoEncoder, SHAPESYotoDecoder
from CIFARtraining import trainBVAEFixedCIFAR, trainBVAEYotoCIFAR
from SHAPESTraining import trainBVAEFixedSHAPES, trainBVAEYotoSHAPES

EPOCHS = 1
training_dir = './trainingResults'
weights_dir = './weights'

def train_Yoto_CIFAR(width, train):
    # train yoto BVAE on the CIFAR10 data set with specified width

    kL_loss, recon_loss, full_loss = [], [], []

    #create the models based on the width
    enc = CIFAR10YotoEncoder(width)
    dec = CIFAR10YotoDecoder(width)

    #save the cifar yoto training data
    trainingDir = "{}/CIFAR10Yoto_x{}.csv".format(training_dir, width)

    #directory to save encoder weights
    encDir = "{}/encCIFAR10Yoto_x{}.weights".format(weights_dir, width)

    # directory to save decoder weights
    decDir = "{}/decCIFAR10Yoto_x{}.weights".format(weights_dir, width)

    kL_loss, recon_loss, full_loss = trainBVAEYotoCIFAR(train, enc, dec, EPOCHS, trainingDir, encDir, decDir)
    return kL_loss, recon_loss, full_loss

def train_BVAE_CIFAR(width, train, beta=1):
    # train fixed bVAE on the CIFAR10 data set with specified width
    kL_loss, recon_loss, full_loss = [], [], []

    # create the models based on the width
    enc = CIFAR10Encoder(width)
    dec = CIFAR10Decoder(width)

    # save the cifar yoto training data
    trainingDir = "{}/CIFAR10BVAE{}_x{}.csv".format(training_dir, beta, width)

    # directory to save encoder weights
    encDir = "{}/encCIFAR10BVAE{}_x{}.weights".format(weights_dir, beta, width)

    # directory to save decoder weights
    decDir = "{}/decCIFAR10BVAE{}_x{}.weights".format(weights_dir, beta, width)

    kL_loss, recon_loss, full_loss = trainBVAEFixedCIFAR(train, enc, dec, beta, EPOCHS, trainingDir, encDir, decDir)
    return kL_loss, recon_loss, full_loss


def train_Yoto_SHAPES(width, train):
    # train yoto BVAE on the SHAPES data set with specified width

    kL_loss, recon_loss, full_loss = [], [], []

    #create the models based on the width
    enc = SHAPESYotoEncoder(width)
    dec = SHAPESYotoDecoder(width)

    #save the cifar yoto training data
    trainingDir = "{}/SHAPESYoto_x{}.csv".format(training_dir, width)

    #directory to save encoder weights
    encDir = "{}/encSHAPESYoto_x{}.weights".format(weights_dir, width)

    # directory to save decoder weights
    decDir = "{}/decSHAPESYoto_x{}.weights".format(weights_dir, width)

    kL_loss, recon_loss, full_loss = trainBVAEYotoSHAPES(train, enc, dec, EPOCHS, trainingDir, encDir, decDir)
    return kL_loss, recon_loss, full_loss

def train_BVAE_SHAPES(width, train, beta=1):
    # train fixed bVAE on the SHAPES data set with specified width
    kL_loss, recon_loss, full_loss = [], [], []

    # create the models based on the width
    enc = SHAPESEncoder(width)
    dec = SHAPESDecoder(width)

    # save the cifar yoto training data
    trainingDir = "{}/SHAPESBVAE{}_x{}.csv".format(training_dir, beta, width)

    # directory to save encoder weights
    encDir = "{}/encSHAPESBVAE{}_x{}.weights".format(weights_dir, beta, width)

    # directory to save decoder weights
    decDir = "{}/decSHAPESBVAE{}_x{}.weights".format(weights_dir, beta, width)

    kL_loss, recon_loss, full_loss = trainBVAEFixedSHAPES(train, enc, dec, beta, EPOCHS, trainingDir, encDir, decDir)
    return kL_loss, recon_loss, full_loss