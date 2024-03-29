import argparse

NAME = "Debian generic"
PATH = "machine/debian-generic"

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog=PATH,description= NAME+' env for Shamrock')

    parser.add_argument("--backend", action='store', help="build directory to use")
    parser.add_argument("--sycl-impl", action='store', help="build directory to use")
    parser.add_argument("--sycl-backend", action='store', help="build directory to use")
    parser.add_argument("--arch", action='store', help="machine assumed for the environment")

    args = parser.parse_args()

    print("test")