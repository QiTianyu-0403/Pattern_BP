import argparse
from train import train


def main():
    parser = argparse.ArgumentParser(description='BP')
    
    parser.add_argument("-m", "--model", help="BP or CNN", type=str, default='BP')
    parser.add_argument("-bs", "--batchsize", help="the batch size of each epoch", type=int, default=32)
    parser.add_argument("-e", "--EPOCH", help="the number of epochs", type=int, default=1500)
    parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=0.001)
    args = parser.parse_args()
    
    train(args)
        
        
if __name__ == "__main__":
    main()