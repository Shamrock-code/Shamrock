# Shamrock install (Spack)

With spack things are pretty easy:

```bash
git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git
source spack/share/spack/setup-env.sh 
spack spec shamrock
spack install shamrock
```

## Using Shamrock

Simply do:
```bash
spack load shamrock
```

After this shamrock will be available both as a programm and as a python lib.