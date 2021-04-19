test <- cbind(`cru2CLDS2000.2012.(mean.monthly.cloudiness)`,`cru2PREC2000.2012.(monthly.precipitation)`,`cru2TAIR2000.2012.(mean.monthly.air.temperature)`,`cru2VAPR2000.2012.(mean.monthly.vapor.pressure)`,`LAI2000.2012.(monthly.leaf.area.index)`,`NPP2000.2012.(monthly.potential.net.primary.production)`)
no_duplicate <- test[!duplicated(as.list(test))]                    
write.csv(no_duplicate,"C:\\Users\\lotus\\Desktop\\input.csv")


myMergedData <- 
  do.call(rbind,
          lapply(list.files(path = "C:\\Users\\lotus\\Desktop\\Data\\Observation (from FLUXNET)\\Processed"), read.csv))
write.csv(myMergedData,"C:\\Users\\lotus\\Desktop\\fluxnet.csv")

dat_fix <- datnew[c(1,8,15,22,29,36,43,50,57,64,71,78,2,9,16,23,30,37,44,51,58,65,72,79,3,10,17,24,31,38,45,52,59,66,73,80,4,11,18,25,32,39,46,53,60,67,74,81,5,12,19,26,33,40,47,54,61,68,75,82,6,13,20,27,34,41,48,55,62,69,76,83,7,14,21,28,35,42,49,56,63,70,77,84),]
fix <- cbind(dat_fix,emp)

write.csv(fix,"C:\\Users\\lotus\\Desktop\\6feature_no999.csv")

#3/14 new data site 1

setwd("C:/Users/lotus/Desktop/New Data (daily)/Observation (from FLUXNET)/VSM/")

path <- "C:/Users/lotus/Desktop/New Data (daily)/Observation (from FLUXNET)/VSM/"
files <- list.files(path=path, pattern="*.csv")
list = lapply(files, read.csv)
library(vroom)
dailyData <- vroom(files)
dailyData$AvgCS616_VWC_5cm[dailyData$AvgCS616_VWC_5cm==-999] <- 0
dailyData$yd <- paste(dailyData$Year, dailyData$Day)
dailyData$AvgCS616_VWC_5cm <- as.numeric(dailyData$AvgCS616_VWC_5cm)
dailyData <- dailyData[!is.na(dailyData$AvgCS616_VWC_5cm),]

#original
dailyAvg <- aggregate(AvgCS616_VWC_5cm~yd, dailyData, FUN=mean)
library(stringr)
dailyAvg$year <- sapply(strsplit(as.character(dailyAvg$yd),' '), "[", 1)
dailyAvg$day <- sapply(strsplit(as.character(dailyAvg$yd),' '), "[", 2)

#try to remove 0s 
dailyData <- dailyData[dailyData$AvgCS616_VWC_5cm != 0, ]
dailyAvg <- aggregate(AvgCS616_VWC_5cm~yd, dailyData, FUN=mean)
library(stringr)
dailyAvg$year <- sapply(strsplit(as.character(dailyAvg$yd),' '), "[", 1)
dailyAvg$day <- sapply(strsplit(as.character(dailyAvg$yd),' '), "[", 2)

#reordering
dailyAvg$try <- as.numeric(dailyAvg$day) #numeric day
library(dplyr)
newD <- arrange(dailyAvg,year,try)

#find features

setwd("C:/Users/lotus/Desktop/New Data (daily)/Input_new (from TEM)")

list_of_files <- list.files(path = "C:/Users/lotus/Desktop/New Data (daily)/Input_new (from TEM)", recursive = TRUE,
                            pattern = "\\.txt$", 
                            full.names = TRUE)
datalist = lapply(list_of_files, function(x)read.table(x, header=F, sep = ",", col.names = paste0("V",seq_len(42)), fill = TRUE)) 

datafr = do.call("rbind", datalist)


#4 features
datafr = datalist[[4]]

datafr = datafr[42:132, ]
t_data <- t(datafr)
t_pure <- t_data[11:42,]
tv <- data.frame(X1 = c(as.matrix(t_pure)))
tv <- tv[!(tv == '' | tv == 'Global')]
tv <- tv[!grepl("Global", tv)]

feature <- tv
feature <- cbind(feature,tv)


feature <- cbind(feature, newD$AvgCS616_VWC_5cm)

write.csv(feature,"C:\\Users\\lotus\\Desktop\\new_data.csv")

look <- read.csv("C:\\Users\\lotus\\Desktop\\new_data.csv")
lok <- look[look$X.1 != 0,]
lok <- cbind(lok[,1:5], newD$AvgCS616_VWC_5cm)
write.csv(lok,"C:\\Users\\lotus\\Desktop\\modified_data.csv")
