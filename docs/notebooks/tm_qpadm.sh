## commands to run treemix and qpadm
# formatting input data
~/Downloads/plink_mac_20210606/plink --bfile c1global1nfd --freq --within c1global1nfd.pops --out c1global1nfdnew
python2 plink2treemix.py c1global1nfd c1global1nfd.tm.gz
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

## subsetting entire AEA dataset
# converting from bed to ped for subsetting
~/Downloads/plink_mac_20210606/plink --bfile ../c1global1nfd --recode --tab --out subc1global1nfd

# vim commands
:%s/\[//g
# :g/ /d
:%s/\]//g

# start R
indspops<-read.csv("../c1global1nfd.indiv_meta",header=T,stringsAsFactors=F)
pop2keep<-read.csv("pops2keep.txt",header=F)
write.table(file="inds2keep.txt",indspops[indspops$popId %in% pop2keep[,1],c(1,1)],row.names=F,quote=F,col.names=F,sep="\t")
write.table(file="keepc1global1nfd.pops",indspops[indspops$popId %in% pop2keep[,1],c(1,1,6)],row.names=F,quote=F,col.names=c("sampleId","sampleId","popId"))

~/Downloads/plink_mac_20210606/plink --file subc1global1nfd --keep inds2keep.txt --make-bed --out keepc1global1nfd --double-id
