from dataset import prepare_CIFAR_dataset, prepare_SHAPES_dataset
from trainingUtils import train_BVAE_CIFAR, train_BVAE_SHAPES, train_Yoto_CIFAR, train_Yoto_SHAPES

def main():
    CIFARtrainloader, CIFARtestloader = prepare_CIFAR_dataset()
    SHAPEStrainloader, SHAPEStestloader = prepare_SHAPES_dataset()

    #train yoto with width 1
    train_Yoto_CIFAR(width=1, train=CIFARtrainloader)

    # train bvae with width 1 and beta
    train_BVAE_CIFAR(width=1, beta=1, train=CIFARtrainloader)
    
    #train yoto with width 1
    #train_Yoto_SHAPES(width=1, train=SHAPEStrainloader)

    # train bvae with width 1 and beta
    #train_BVAE_SHAPES(width=1, beta=1, train=SHAPEStrainloader)

    '''
    this is the training for the CIFAR 10 dataset
    We changed width to 1, 2, 4 and the beta was rerun with beta 1, 16, 64, 256
    
    # train bvae with width 1 and beta
    for i in [1, 2, 4]:
        train_Yoto_CIFAR(width=i)
        for j in [1, 16, 64, 256]:
            train_BVAE_CIFAR(width=i, beta=j)
    '''

    '''
    this is the training for the SHAPES dataset (this takes a lot longer than CIFAR - BEWARE)
    We changed width to 1, 2, 4 and the beta was rerun with beta 1, 128, 512

        # train bvae with width 1 and beta
        for i in [1, 2, 4]:
            train_Yoto_SHAPES(width=i)
            for j in [1, 128, 512]:
                train_BVAE_SHAPES(width=i, beta=j)
        '''


if __name__ == '__main__':
    main()