[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/NovembreLab/feems/main)

# FEEMS

**F**ast **E**stimation of **E**ffective **M**igration **S**urfaces (`FEEMS`) is a python package 
implementing a statistical method for inferring and visualizing gene-flow in 
spatial population genetic data, requiring only genotype data and locations of samples.

The `FEEMS` method and software was developed by Joe Marcus and Wooseok Ha and 
advised by Rina Foygel Barber and John Novembre. We also used code from Benjamin M. Peter 
to help construct the spatial graphs. 

For details on the method see our [publication](https://doi.org/10.7554/eLife.61927). 

# FEEMSmix

**F**ast **E**stimation of **E**ffective **M**igration **S**urfaces + ad**mix**ture (`FEEMSmix`) is built on top of `FEEMS`, and is a method for 
representing long-range gene flow on a background migration surface estimated by `FEEMS`. 

The `FEEMSmix` method and software was developed by Vivaswat Shastry and John Novembre. 

#### Jump to [RUNNING `FEEMS`](https://github.com/VivaswatS/feems/blob/main/README.md#running-feems) to get started right away!  

# IMPORTANT UPDATES (since v1.0)

### Change in the default mode of `FEEMS`

In the original publication above, `FEEMS` only fit the edge weights on the graph and kept the node-specific parameters (proportional to heterozygosities) fixed at a constant value. However, in the latest version, we change the default functionality to be one in which we fit node-specific parameters under the model. This will increase both model fit and runtime, however, we've found this to be a worthy trade-off, especially when looking for long-range gene flow events on a baseline `FEEMS` grid. The old functionality can still be used by setting `sp_graph.fit(..., optimize_q=None)`. 

This new map of heterozygosity values (roughly proportional to effective population size) can also be visualized across the grid. See [getting-started.ipynb](https://github.com/VivaswatS/feems/blob/main/docsrc/notebooks/getting-started.ipynb) and [cross-validation.ipynb](https://github.com/VivaswatS/feems/blob/main/docsrc/notebooks/cross-validation.ipynb) for more details on this mode. 

### Inclusion of a shape file with coarser resolution (`grid_500.sh*`)

If you have samples spanning continental scales, then a coarser grid may be of use to you, so we provide three resolutions (spanning from fine to coarse): 

1. `grid_100` (corresponds to `res = 6` in `dgconstruct` from package `dggridR` with cell area of approx. $6{,}200$ sq km and a cell spacing of $110$ km)
2. `grid_250` (corresponds to `res = 5` in `dgconstruct` with a cell area of approx. $25{,}000$ sq km and a cell spacing of $220$ km)
3. `grid_500` (corresponds to `res = 4` in `dgconstruct` and a cell area of $100{,}000$ sq km with a cell spacing of $440$ km)

You can also create your own shapefile using this command in R:
```r
dggs <- dgconstruct(
    res = 4,
    precision = 30,
    projection = "ISEA",
    aperture = 4,
    topology = "TRIANGLE"
)

dgearthgrid(dggs, savegrid="PATH/world_triangle_res4.shp")
```

_Note: For finer resolutions (`res > 8`), this command will take a really long time to save the grid for the entire earth. In these cases, you can subset a region of the world and intersect this region with the triangular grid, see an example [R script](https://github.com/karolisr/pitcairnia-dr-nrv/blob/36430941db8762b703ef58d94764b77a33763798/data/dggs/generate-dgg-files.R) (thanks to @karolisr)_

**Rule of thumb**: For the density of the grid, it is a balance between finer resolutions and runtime: a good place to start is a resolution in which most individuals sampled as part of a single sampling population/unit get assigned to a unique deme. 

The `outer` extent/boundary of the grid can be constructed using this tool: [https://www.keene.edu/campus/maps/tool/](https://www.keene.edu/campus/maps/tool/). 

### Inclusion of a flag to avoid wrapping locations around America

Previously, `FEEMS` employed a trick to place points in Western Alaska on the 'other side' of the International Date Line, i.e., on the 'same side' of the map as the North American landmass (only an issue with some projections). This threshold longitude was chosen as $-40$. 

Now, we provide an option in `prepare_graph_inputs(..., wrap_america=False)` to turn the flag off for flexibility. 

### Inclusion of a plotting script using `ggplot2` 

We also now have a script that can plot the `FEEMS` map in R using `ggplot2` (thanks to @LukeAndersonTrocme). First, you will need to export the relevant graph attributes in python (after `FEEMS` has been fit) using these commands:
```python
# write the relevant edge weights out into a csv file
np.savetxt('edgew.csv', np.vstack((np.array(sp_graph.edges).T, sp_graph.w)).T, delimiter=',')

# write the deme coordinates + sample size (node attributes) out into a csv file
np.savetxt('nodepos.csv', np.vstack((sp_graph.node_pos.T, [sp_graph.nodes[n]['n_samples'] for n in range(len(sp_graph.nodes))])).T, delimiter=',')
```
and then you can use this script [ggplot_feems.R](https://github.com/VivaswatS/feems/blob/main/feems/data/ggplot_feems.R) to plot the baseline `FEEMS` figure in R. With the figure in `ggplot2` format, you will also be able to add any extra features as you see fit (e.g., sample locations, ecological gradients, etc.)

### Inclusion of spatial prediction as a feature

With `FEEMSmix`, we also provide the functionality of predicting the location of samples on a migration surface estimated with `FEEMS`. We observe comparable performance to another state-of-the-art deep learning method called `Locator` ([Battey _et al_ 2020](https://doi.org/10.7554/eLife.54507)) and believe it could be a useful tool for the spatial population genetics community. This functionality can be found in [miscellaneous-functions.ipynb](https://github.com/VivaswatS/feems/blob/main/docsrc/notebooks/miscellaneous-functions.ipynb).

### Visualization of admixture pies on `FEEMS` map

We provide a basic function to overlay admixture proportions from an `admixture`/`STRUCTURE`-like model as pie charts on an underlying `FEEMS` map. This has proven useful as a visualization tool when interpreting both `FEEMS` and `FEEMSmix` results (especially, the latter). See example code in [miscellaneous-functions.ipynb](https://github.com/VivaswatS/feems/blob/main/docsrc/notebooks/miscellaneous-functions.ipynb).

### Visualization of model fits with PCA and `admixture` ([Alexander _et al_ 2009](https://genome.cshlp.org/content/19/9/1655.long))

We provide functions to visualize fits of two widely-used models (Principal Components Analysis and `admixture`) to the observed genetic data and plot the outliers on a geographic map (akin to a `FEEMSmix` analysis). For PCA, we compute the principal components in-house, whereas for `admixture`, we ask you provide the .P & .Q matrices for different $K$ values. This functionality can be found in [miscellaneous-functions.ipynb](https://github.com/VivaswatS/feems/blob/main/docsrc/notebooks/miscellaneous-functions.ipynb).

# INSTALLATION

Note: MS Windows users will struggle to install `FEEMS` directly in a 
Windows environment because at least one of the dependencies does not
have a Windows port. A virtual Linux machine should be preferable if 
you are on a Windows machine. 

## Quick start using bioconda

Typically, the simplest way to get started with `FEEMS` (or `FEEMSmix`) is to install 
[Anaconda][anaconda] or [Miniconda][miniconda] or [mamba][mamba], 
then install `FEEMS` using the [Bioconda recipe][bioconda-recipe]:

```bash
conda install -c bioconda feems -c conda-forge
```

See the next section for alternative ways to install FEEMS, or if 
"conda install" worked for you, skip ahead to "Running `FEEMS`". 

## Alternative installation instructions (Python >=3.8)

As an alternative way to get started, setup a `conda` 
environment:

```bash
conda create -n=feems_e python=3.12
conda activate feems_e
```

Some of the plotting utilities in the `FEEMS` package require `geos` as a 
dependency which can be installed on a Mac with brew as follows:

```bash
brew install geos
```

If you are on a Windows machine, you can install `geos` using: 

```bash
conda install -c conda-forge geos
```

Then, you can install further dependencies using:

```bash
conda install -c conda-forge scikit-sparse suitesparse 
conda install -c conda-forge cartopy
```

Jupyter and jupyterlab are also needed to explore the example notebooks but 
are *not* required for the functioning of the `FEEMS`/`FEEMSmix` package (you could also use `brew`, if you want it to be available outside the environment):

```bash
pip install notebook
```

Once the `conda` environment has 
been setup with these dependencies, we can install `FEEMS`:

```bash
pip install git+https://github.com/VivaswatS/feems
```

You can also install `FEEMS` locally by:

```bash
git clone https://github.com/VivaswatS/feems
cd feems/
pip install .
```

NOTE: Some users have reported a compatibility error arising at this step with the installation of shapely v1.7.1 (specificed in requirements.txt).  If this arises, recreate the `feems_e` conda environment, and run `pip install shapely --no-binary shapely` before the `pip install feems` command above. 

## Another alternative using the `.yml` file

Another easy option is also to use the `feems.yml` file from this repo as a blueprint for the installation and setup of the appropriate conda environment. 

You can download the file onto your computer and simply run:

```bash
conda env create -f feems.yml
conda activate feems_e
```

This will create an environment called `feems_e` which will contain all the dependencies and `FEEMS`/`FEEMSmix`. 

# RUNNING `FEEMS`

To help get your analysis started (and to verify appropriate installation), we provide an example workflow in the [getting-started.ipynb](https://github.com/VivaswatS/feems/blob/main/docsrc/notebooks/getting-started.ipynb) notebook. The notebook analyzes empirical data from North American gray wolves populations published in [Schweizer et al. 2015](https://onlinelibrary.wiley.com/doi/full/10.1111/mec.13364?casa_token=idW0quVPOU0AAAAA:o_ll85b8rDbnW3GtgVeeBUB4oDepm9hQW3Y445HI84LC5itXsiH9dGO-QYGPMsuz0b_7eNkRp8Mf6tlW). 

An example workflow using a λ value estimated from a cross-validation procedure is highlighted in [cross-validation.ipynb](https://github.com/VivaswatS/feems/blob/main/docsrc/notebooks/cross-validation.ipynb). We recommend using this procedure in choosing an appropriate λ value for the fit. 

# RUNNING `FEEMSmix`

Since `FEEMSmix` is built on top of `FEEMS`, this analysis will start where the previous section left off (i.e., after the initial `FEEMS` fit). We will also use the data from North American gray wolves to illustrate the working of this method, scroll down the [getting-started.ipynb](https://github.com/VivaswatS/feems/blob/main/docsrc/notebooks/getting-started.ipynb) to see example analyses. 

[anaconda]: https://www.anaconda.com/products/distribution
[miniconda]: https://docs.conda.io
[mamba]: https://mamba.readthedocs.io/en/latest/
[bioconda-recipe]: https://anaconda.org/bioconda/feems
