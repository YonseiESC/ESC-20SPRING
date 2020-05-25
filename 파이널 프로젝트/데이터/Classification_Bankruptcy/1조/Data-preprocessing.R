# Data importing

setwd('C:/Users/Duck/Desktop/2020-1학기/ESC')
bankrupt <- read.csv('bankrupt.csv')
attach(bankrupt)

dpl <- function(attr){
# factor -> numeric
    if(is.factor(attr)){
    attr <- as.numeric(as.character(attr))
  }

data <- sort(attr); preq <- c(); cumul <- c(); rate <- c()
point <- quantile(Attr1,c(0,0.05,0.125,0.25,0.4,0.6,0.75,0.875,0.95,1))

for(i in 1:9){
  if(i==1){
    preq[i] <- length(data[data>=point[i]&data<=point[i+1]])
    cumul[i] <- sum(preq[1:i])
  }
  else{
    preq[i] <- length(data[data>point[i]&data<=point[i+1]])
    cumul[i] <- sum(preq[1:i])
  }
}

rate[1] <- sum(class[order(attr)][1:cumul[1]])
for(i in 1:8){
  rate[i+1] <- sum(class[order(attr)][(cumul[i]+1):(cumul[i+1])])
}

plot(preq/sum(preq),type='h',lwd=10,ylim=c(0,0.25))
lines(rate/preq)
}