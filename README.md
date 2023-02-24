Testing updates to README file!


[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/NovembreLab/feems/main)

# feems

**F**ast **E**stimation of **E**ffective **M**igration **S**urfaces (`feems`) is a python package 
implementing a statistical method for inferring and visualizing gene-flow in 
spatial population genetic data.

The `feems` method and software was developed by Joe Marcus and Wooseok Ha and 
advised by Rina Foygel Barber and John Novembre. We also used code from Benjamin M. Peter 
to help construct the spatial graphs. 

For details on the method see our [pre-print](https://www.biorxiv.org/content/10.1101/2020.08.07.242214v1). Note that `feems` is in review so the method could be subject to change.  

Note: MS Windows users will struggle to install feems directly in a 
Windows environment because at least one of the dependencies does not
have a Windows port.  A virtual Linux machine should be preferable if 
you are on a Windows machine. 

# Quick start using bioconda

Typically the simplest way to get started with feems is to install 
[Anaconda][anaconda] or [Miniconda][miniconda], 
then install feems using the [Bioconda recipe][bioconda-recipe]:

```bash
conda install -c bioconda feems -c conda-forge
```

See the next section for alternative ways to install feems, or if 
"conda install" worked for you, skip ahead to "Running feems". 

# Alternative installation instructions (Python 3.8)

As an alternative way to get started, setup a `conda` 
environment:

```
conda create -n=feems_e python=3.8.3 
conda activate feems_e
```

Some of the plotting utilities in the `feems` package require `geos` as a 
dependency which can be installed on mac with brew as follows:

```
brew install geos
```

Unfortunately some of the other dependencies for `feems` are not easily 
installed by pip so we recommend getting started using `conda`:

```
conda install -c conda-forge suitesparse=5.7.2 scikit-sparse=0.4.4 cartopy=0.18.0 jupyter=1.0.0 jupyterlab=2.1.5 sphinx=3.1.2 sphinx_rtd_theme=0.5.0 nbsphinx=0.7.1 pandas-plink sphinx-autodoc-typehints
```

We added jupyter and jupyterlab to explore some example notebooks but these 
are not necessary for the `feems` package. Once the `conda` environment has 
been setup with these tricky dependencies we can install `feems`:

```
pip install git+https://github.com/NovembreLab/feems
```

You can also install `feems` locally by:

```
git clone https://github.com/NovembreLab/feems
cd feems/
pip install .
```
NOTE: Some users have reported a compatibility error arising at this step with the installation of shapely v1.7.1 (specificed in requirements.txt).  If this arises, recreate the `feems_e` conda environment, and run `pip install shapely --no-binary shapely==1.7.1` before the `pip install` feems command above. 

# Running feems

To help get your analysis started, we provide an example workflow in the [getting-started.ipynb](https://github.com/NovembreLab/feems/blob/main/docsrc/notebooks/getting-started.ipynb) notebook. The notebook analyzes empirical data from North American gray wolves populations published in [Schweizer et al. 2015](https://onlinelibrary.wiley.com/doi/full/10.1111/mec.13364?casa_token=idW0quVPOU0AAAAA:o_ll85b8rDbnW3GtgVeeBUB4oDepm9hQW3Y445HI84LC5itXsiH9dGO-QYGPMsuz0b_7eNkRp8Mf6tlW). 

An example workflow using a λ value estimated from a cross-validation procedure is highlighted in [cross-validation.ipynb](https://github.com/NovembreLab/feems/blob/main/docsrc/notebooks/cross-validation.ipynb). We recommend using this procedure in choosing an appropriate λ value for the fit. 

[anaconda]: https://www.anaconda.com/products/distribution
[miniconda]: https://docs.conda.io
[bioconda-recipe]: https://anaconda.org/bioconda/feems
