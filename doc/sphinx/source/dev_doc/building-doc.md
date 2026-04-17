# Building the documentation

## Repo setup

We assume in this guide that you have cloned and built Shamrock from source.

In other word something like this:
```bash
git clone --recurse-submodules git@github.com:Shamrock-code/Shamrock.git
cd Shamrock
./env/new-env --builddir build --machine debian-generic.acpp -- --backend omp
cd build
source ./activate
shamconfigure
shammake
```

## Building the Sphinx documentation

The Sphinx documentation is now the main documentation of the code. In order to build it i provide some utilites that come with the main dev environments rather than giving you the endless list of commands to get it working because this is such a pain ...

The Sphinx documentation include several components, the user and developer documentation as well as the Python API and Examples. All of them are mandatory to build the doc exept from running all the examples. In general if you have a standard laptop building all the exemples take a very long time so i advise against it. In that case you can, provided that the environment is active do

### Without running the examples

```bash
# I assume you already sources the env like so
source ./activate

# Generate the sphinx doc but do not run the examples
generate_sphinx_doc_no_examples
```

Once it is built just open the resulting page like so if you use firefox:
```bash
firefox ../doc/sphinx/build/html/index.html
```

### With a single example

Especially if you are writing new examples you may want to test them. In that case the command is 
```bash
generate_sphinx_doc_single_example examples/sph/run_orzag_tang.py
```
You can replace the path by anything in the `examples/` directory.

### All the examples (very long...)

In general i'd say do not bother and let the Github CI do that for you and download the result. If you are brave though here is the command:
```bash
generate_sphinx_doc_with_examples
```

## Building the doxygen doc

Just do
```bash
(cd ../doc/doxygen && doxygen dox.conf)
```

And then to open it
```bash
firefox ../doc/doxygen/html/index.html
```