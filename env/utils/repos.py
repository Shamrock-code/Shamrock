import os

def clone_acpp(folder):
    if os.path.isdir(folder):
        print("-- skipping git clone folder does already exist")
    else:
        print("-- clonning https://github.com/AdaptiveCpp/AdaptiveCpp.git")
        os.system("git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git "+folder)
