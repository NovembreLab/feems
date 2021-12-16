## commands to run treemix and qpadm
# formatting input data
~/Downloads/plink_mac_20210606/plink --bfile c1global1nfd --freq --within c1global1nfd.pops --out c1global1nfdnew
python2 plink2treemix.py c1global1nfd.frq.strat.gz c1global1nfd.tm
gzip c1global1nfd.tm
# running treemix
for i in {1..10}
> do
> ~/Documents/treemix-1.13/src/treemix -i c1global1nfd.tm.gz -o tmoutfiles/out.$i -k 1000 -m $i > tm_${i}_log
> done

# converting input data (PACKEDPED to PACKEDANCESTRYMAPGENO)
# use convertf as in the manual (all input files the same - .fam becomes .ped but with pop labels now)
# running qpadm
~/Documents/AdmixTools/bin/qpDstat -p pardstatfile 

/Users/vivaswat/Documents/AdmixTools/bin/qpDstat: parameter file: pardstatfile
### THE INPUT PARAMETERS
##PARAMETER NAME: VALUE
genotypename: c1global1nfd.packedancestrymapgeno
snpname: c1global1nfd.snp
indivname: c1global1nfd.ind
popfilename: f4popfile
f4mode: YES