############################################################################
#Principal Components Analysis
#Dataset: Pen Based Handwritten Digits - 0 to 9
#
#k-NN with PCA's that make upto 80% variation
#https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/

#Modified by: Anand
#Modified Date: 12/16/2020
############################################################################

rm(list=ls())

set.seed(235)
library("ggfortify")
library("class")
library("pca3d")
library("rgl")

#loading dataset
pen.tra = read.table("pendigits.tra", sep = ",")
pen.tes = read.table("pendigits.tes", sep = ",")

pendigits = rbind(pen.tra, pen.tes)
names(pendigits) = c(paste0(c("X", "Y"), rep(1:8, each = 2)), "digit")
pendigits$digit = factor(pendigits$digit)

rm(pen.tra, pen.tes)

dim(pendigits)
names(pendigits)

table(pendigits$digit)

####################################################################################
#Variance-of 16 variables
covart<-var(pendigits[,-17])
sort(diag(covart),decreasing = T)

#to find eigen values of covariance matrix to understand how many PCA's contribute
#80-90% variance in data
eigenval<-eigen(covart)
eigval<-eigenval$values

round(eigval*100/sum(eigval),2)

#Inorder to use data with with 80 to 90% variance we have to 
for (i in seq(1,length(eigval))){
  cat("Number of PCs to be included:",i," for percentage variation:",
      sum(eigval[1:i])*100/sum(eigval),"%","\n")
}

###################################################################################
pca.res<- prcomp(pendigits[,-17], scale. = F)

x11()
autoplot(pca.res,x=1,y=2,data = pendigits, colour = 'digit')

x11()
autoplot(pca.res,x=3,y=4,data = pendigits, colour = 'digit')

x11()
autoplot(pca.res,x=5,y=6,data = pendigits, colour = 'digit')

x11()
autoplot(pca.res,x=7,y=8,data = pendigits, colour = 'digit')

###################################################################################
###3D plot 
groups<-factor(pendigits$digit)
summary(groups)

x11()
pca3d(pca.res, group=groups,show.plane=TRUE, legend = "right")


pca.res <- princomp(pendigits[,-17], cor=TRUE, scores=TRUE)
plot3d(pca.res$scores[,1:3],xlab="PCA1", ylab="PCA2", zlab="PCA3",main="3D plot of PCA Scores",col=rainbow(1000))

################K-Nearest Neighbors for raw and PCA##################
dim(pendigits)

trainidx<-sample(1:nrow(pendigits),nrow(pendigits)*0.7,replace = FALSE)

rtraindat<-pendigits[trainidx,]
rtestdat<-pendigits[-trainidx,]
dim(rtraindat)
dim(rtestdat)

pca.res<-princomp(pendigits[,-17],cor = T)
pca.comp<-pca.res$scores
pca.data <- -1*pca.comp[,1:7] # to include PCA 1 to 7 which adds upto 91% variations in data


ptraindat<-pca.data[trainidx,]
ptestdat<-pca.data[-trainidx,]
ptrlbls<-pendigits[trainidx,17]
ptslbls<-pendigits[-trainidx,17]
dim(ptraindat)
dim(ptestdat)


rtrainacc=c()
rtestacc=c()
ptrainacc=c()
ptestacc=c()

applyknn<-function(k,tdata,testdata,class){
  knn(train=tdata,
      test=testdata,
      cl=class,
      k=k)
}


for(i in c(1,2,4,6,8,10)){
  
  ##################################################################################
  #raw data
  
  predict.knn.rtrain<-applyknn(i,rtraindat[,-17],rtraindat[,-17],as.factor(rtraindat$digit))
  predict.knn.rtest<-applyknn(i,rtraindat[,-17],rtestdat[,-17],as.factor(rtraindat$digit))
  
  classif<-(predict.knn.rtrain==rtraindat$digit)
  rtrainacc<-append(rtrainacc,100*length(classif[classif==TRUE])/length(rtraindat$digit))
  
  classif<-(predict.knn.rtest==rtestdat$digit)
  rtestacc<-append(rtestacc,100*length(classif[classif==TRUE])/length(rtestdat$digit))

  ###############################################################################
  #PCA
  
  predict.knn.ptrain<-applyknn(i,ptraindat,ptraindat,as.factor(ptrlbls))
  predict.knn.ptest<-applyknn(i,ptraindat,ptestdat,as.factor(ptrlbls))
  
  classif<-(predict.knn.ptrain==ptrlbls)
  ptrainacc<-append(ptrainacc,100*length(classif[classif==TRUE])/length(ptrlbls))
  
  classif<-(predict.knn.ptest==ptslbls)
  ptestacc<-append(ptestacc,100*length(classif[classif==TRUE])/length(ptslbls))
  
}



x11()
comperror<-data.frame("K"=c(1,2,4,6,8,10),
                      "rtrainacc"=rtrainacc,
                      "rtestacc"=rtestacc,
                      "ptrainacc"=ptrainacc,
                      "ptestacc"=ptestacc)

colors <- c("Raw Data Train Accuracy" = "red", "Raw Data Test Accuracy" = "steelblue", "PCA Train Accuracy" = "turquoise","PCA Test Accuracy"="springgreen")
ggplot()+
  geom_line(data=comperror,aes(K, rtrainacc,color="Raw Data Train Accuracy"),size = 0.8) +  
  geom_line(data=comperror,aes(K, rtestacc,color="Raw Data Test Accuracy"),size = 0.8)+
  geom_line(data=comperror,aes(K, ptrainacc,color="PCA Train Accuracy"),size = 0.8)+
  geom_line(data=comperror,aes(K, ptestacc,color="PCA Test Accuracy"),size = 0.8)+
  labs(x = "k",
       y = "Classification Accuracy",
       color = "Legend") +scale_x_continuous(breaks=c(1,2,4,6,8,10))+
  scale_color_manual(values = colors)+ggtitle("k-NN Classification Raw Data v/s PCA")

##################################################################################################################
#Raw Data
predict.knn.rtest<-applyknn(4,rtraindat[,-17],rtestdat[,-17],as.factor(rtraindat$digit))

classif<-(predict.knn.rtest==rtestdat$digit)
frtestacc<-100*length(classif[classif==TRUE])/length(rtestdat$digit)

#PData
predict.knn.ptest<-applyknn(4,ptraindat,ptestdat,as.factor(ptrlbls))

classif<-(predict.knn.ptest==ptslbls)
fptestacc<-100*length(classif[classif==TRUE])/length(ptslbls)

frtestacc
fptestacc