## R script to plot flashpca results using bpeter scripts
# feb 7, 2022

library(ggplot2)
setwd("/project2/jnovembre/old_project/bpeter/eems_tib/pca")

# flash_c1global1nfd_dim100
load.pc<-read.table('flash_global0sf_dim20.load',header=F,sep="",stringsAsFactors=F,)
# 19954 x 100
pc.pc<-read.table('flash_global0sf_dim20.pc',header=F,sep="",stringsAsFactors=F,)
# 4697 x 100
pve.pc<-round(read.table('flash_global0sf_dim20.pve',header=F,)[,1]*100,2)
# 100

# basic plots with black dots
plot(pc.pc[,1],pc.pc[,2],pch=20,cex=0.25,xlab=paste0('PC1 (',pve.pc[1],'%)'),ylab=paste0('PC2 (',pve.pc[2],'%)'))
plot(pc.pc[,2],pc.pc[,3],pch=20,cex=0.25,xlab=paste0('PC2 (',pve.pc[2],'%)'),ylab=paste0('PC3 (',pve.pc[3],'%)'))
plot(pc.pc[,1],pc.pc[,3],pch=20,cex=0.25,xlab=paste0('PC1 (',pve.pc[1],'%)'),ylab=paste0('PC3 (',pve.pc[3],'%)'))

# get median location for pops
med.pc<-read.table("median_global0sf_dim20.pc",header=T,sep=',')
pop.pc<-med.pc[,1]
med.pc<-med.pc[,-1]

plot(med.pc[,1],med.pc[,2],pch=20,col='grey40',xlab=paste0('PC1 (',pve.pc[1],'%)'),ylab=paste0('PC2 (',pve.pc[2],'%)'))
# throw a ggrepel on this binch
ggplot(med.pc,aes(x=PC1_M,y=PC2_M))+geom_point(color='grey40')+xlab(paste0('PC1 (',pve.pc[1],'%)'))+ylab(paste0('PC2 (',pve.pc[2],'%)'))+theme_bw()

# file with meta info about pops (had to add in line for pop 753 AX-ME from gvar2.pop_display...)
pop.meta<-read.csv('../gvar2.pop_display',header=T,sep=',',stringsAsFactors=F,fill=T)
pop.meta<-pop.meta[pop.meta$popId %in% pop.pc,]

## ggrepel with abbrev
ggplot()+geom_point(data=med.pc,aes(x=PC1_M,y=PC2_M),color='grey40')+xlab(paste0('PC1 (',pve.pc[1],'%)'))+ylab(paste0('PC2 (',pve.pc[2],'%)'))+theme_bw()+geom_text_repel(data=med.pc,aes(x=PC1_M,y=PC2_M),label=pop.meta$abbrev,size=2.5,max.overlaps=15)
p12<-ggplot()+geom_point(data=med.pc,aes(x=PC1_M,y=PC2_M),color='grey40')+xlab(paste0('PC1 (',pve.pc[1],'%)'))+ylab(paste0('PC2 (',pve.pc[2],'%)'))+theme_bw()

ggplot()+geom_point(data=med.pc,aes(x=PC1_M,y=PC3_M),color='grey40')+xlab(paste0('PC1 (',pve.pc[1],'%)'))+ylab(paste0('PC3 (',pve.pc[3],'%)'))+theme_bw()+geom_text_repel(data=med.pc,aes(x=PC1_M,y=PC3_M),label=pop.meta$abbrev,size=2.5,max.overlaps=15)
p13<-ggplot()+geom_point(data=med.pc,aes(x=PC1_M,y=PC3_M),color='grey40')+xlab(paste0('PC1 (',pve.pc[1],'%)'))+ylab(paste0('PC3 (',pve.pc[3],'%)'))+theme_bw()

ggplot()+geom_point(data=med.pc,aes(x=PC2_M,y=PC3_M),color='grey40')+xlab(paste0('PC2 (',pve.pc[2],'%)'))+ylab(paste0('PC3 (',pve.pc[3],'%)'))+theme_bw()+geom_text_repel(data=med.pc,aes(x=PC2_M,y=PC3_M),label=pop.meta$abbrev,size=2.5,max.overlaps=15)
p23<-ggplot()+geom_point(data=med.pc,aes(x=PC2_M,y=PC3_M),color='grey40')+xlab(paste0('PC2 (',pve.pc[2],'%)'))+ylab(paste0('PC3 (',pve.pc[3],'%)'))+theme_bw()

# long range edges with joint optim (lamb=2.5)
lre.pop1<-c(219,217,218,493,93,493,217,219,791,218,327,100,402,264,264,566,351,402,107,327)
lre.pop2<-c(452,452,452,402,402,264,766,766,327,766,344,402,219,219,217,402,327,217,402,452)
            753,753,753,402,402,402,753,327,766,219,217,766,402,264,418,402,218,264,766,418
            753,753,753,402,264,402,766,766,327,766,219,219,217,402,402,217,218,327,344,418
# long range edges with default optim 
lre.pop1<-c(217,219,218,217,218,219,493,766,538,93,493,758,463,463,107,402,130,93,538,418)
lre.pop2<-c(452,452,452,453,453,453,402,453,402,402,418,453,453,452,402,452,452,418,418,452)
lre<-data.frame(PC1 = med.pc$PC1_M[c(match(lre.pop1,pop.meta$popId),match(lre.pop2,pop.meta$popId))],
                PC2 = med.pc$PC2_M[c(match(lre.pop1,pop.meta$popId),match(lre.pop2,pop.meta$popId))],
                PC3 = med.pc$PC3_M[c(match(lre.pop1,pop.meta$popId),match(lre.pop2,pop.meta$popId))],
                labs = pop.meta$abbrev[c(match(lre.pop1,pop.meta$popId),match(lre.pop2,pop.meta$popId))])
lre$labs[duplicated(lre$labs)] <- NA
p12 + geom_segment(aes(x=lre[1:20,1], y=lre[1:20,2], xend=lre[21:40,1], yend=lre[21:40,2]), lintype=3, size=1.3, color='grey80', alpha=0.8) + geom_text_repel(data=lre,aes(x=PC1,y=PC2,label=labs),size=3,max.overlaps=15)
p13 + geom_segment(aes(x=lre[1:20,1], y=lre[1:20,3], xend=lre[21:40,1], yend=lre[21:40,3]), lintype=3, size=1.3, color='grey80', alpha=0.8) + geom_text_repel(data=lre,aes(x=PC1,y=PC3,label=labs),size=3,max.overlaps=15)
p23 + geom_segment(aes(x=lre[1:20,2], y=lre[1:20,3], xend=lre[21:40,2], yend=lre[21:40,3]), lintype=3, size=1.3, color='grey80', alpha=0.8) + geom_text_repel(data=lre,aes(x=PC2,y=PC3,label=labs),size=3,max.overlaps=15)

## finding average within cluster vs between cluster differences as a way of seeing which PCs best separate out the pops with
## 'spurious' edges to see if we can find alleles/SNPs that are associated with potential batch effects
# write a function that takes two pop ids as input and gives the distribution amongst all PCs
get.dist.clust.med<-function(id1, id2, pop.meta, med.pc){
    x1<-which(pop.meta$popId==id1)
    x2<-which(pop.meta$popId==id2)

    dist.pc<-matrix(NA,nrow=20,ncol=2)
    for(i in 1:20){
        dist.pc[i,1]<-abs(med.pc[x1,i]-med.pc[x2,i])
        dist.pc[i,2]<-mean(dist(med.pc[,i]))
    }
    return(dist.pc)
}

ggplot()+geom_point(data=med.pc,aes(x=pop.meta$popId,y=PC20_M),color='grey40')+xlab(paste0('PC20 (',pve.pc[20],'%)'))+theme_bw()+geom_text_repel(data=med.pc,aes(x=pop.meta$popId,y=PC20_M),label=pop.meta$abbrev,size=2.5,max.overlaps=15)

ggplot()+geom_point(data=med.pc,aes(x=pop.meta$popId,y=PC13_M),color='grey40')+xlab(paste0('PC13 (',pve.pc[13],'%)'))+theme_bw()+geom_text_repel(data=med.pc,aes(x=pop.meta$popId,y=PC13_M),label=pop.meta$abbrev,size=2.5,max.overlaps=15)

ggplot()+geom_point(data=med.pc,aes(x=pop.meta$popId,y=PC3_M),color='grey40')+xlab(paste0('PC3 (',pve.pc[3],'%)'))+theme_bw()+geom_text_repel(data=med.pc,aes(x=pop.meta$popId,y=PC3_M),label=pop.meta$abbrev,size=2.5,max.overlaps=15)

ggplot()+geom_point(data=med.pc,aes(x=pop.meta$popId,y=PC16_M),color='grey40')+xlab(paste0('PC16 (',pve.pc[16],'%)'))+theme_bw()+geom_text_repel(data=med.pc,aes(x=pop.meta$popId,y=PC16_M),label=pop.meta$abbrev,size=2.5,max.overlaps=15)