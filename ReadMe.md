# Quickstart guide

install conda - [Anaconda Distribution](https://www.anaconda.com/download)
Don't forget to include it in path. Otherwise its gonna be very annoying.


run:

``` cmd
conda env create --file environment.yml
conda activate iGEM
```

to rebuild after modifying the yml file (to include new variables):

``` cmd
conda env update -f environment.yml
conda activate iGEM
```

If you installed something you shouldn't, use this 
```
conda env update -f environment.yml --prune
conda activate iGEM
```

once you run pymol install a plugin from
```
https://github.com/APAJanssen/Alphafold2import
```