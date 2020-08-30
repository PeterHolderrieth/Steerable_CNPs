import argparse

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.set_defaults(
    BATCH_SIZE=30,
    N_EPOCHS=3,
    PRINT_OUTPUT=True,
    N_ITERAT_PER_EPOCH=1,
    LEARNING_RATE=1e-4, 
    FILENAME=None)

# Add the arguments to the parser
ap.add_argument("-batch", "--BATCH_SIZE", type=int, required=False,
   help="Batch size.")
ap.add_argument("-lr", "--LEARNING_RATE", type=float,required=False,
   help="Learning rate.")
ap.add_argument("-G", "--Group", type=str, required=True,help="Group")
ap.add_argument("-print", "--PRINT_OUTPUT", type=bool, required=False,help="Print output?")
ap.add_argument("-epochs", "--N_EPOCHS", type=int, required=False,help="Number of epochs.")
ap.add_argument("-it", "--N_ITERAT_PER_EPOCH", type=int, required=False,help="Number of iterations per epoch.")
ap.add_argument("-file", "--FILENAME", type=str, required=False,help="Number of iterations per epoch.")

args = vars(ap.parse_args())
print(args)