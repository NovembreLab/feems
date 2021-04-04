# feems

**F**ast **E**stimation of **E**ffective **M**igration **S**urfaces (`feems`) is a python package 
implementing a statistical method for inferring and visualizing gene-flow in 
spatial population genetic data.

The `feems` method and software was developed by Joe Marcus and Wooseok Ha and 
advised by Rina Foygel Barber and John Novembre. We also used code from Benjamin M. Peter 
to help construct the spatial graphs. 

# Setup

We've found that the easiest way to get started is to setup a `conda` 
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
conda install -c conda-forge suitesparse=5.7.2 scikit-sparse=0.4.4 cartopy=0.18.0 jupyter=1.0.0 jupyterlab=2.1.5 sphinx=3.1.2 sphinx_rtd_theme=0.5.0 nbsphinx=0.7.1 sphinx-autodoc-typehints
```

We added jupyter and jupyterlab to explore some example notebooks but these 
are not necessary for the `feems` package. Once the `conda` environment has 
been setup with these tricky dependencies we can install `feems`:

```
pip install git+https://github.com/jhmarcus/feems
```

You can also install `feems` locally by:

```
git clone https://github.com/jhmarcus/feems
cd feems/
pip install .
```
