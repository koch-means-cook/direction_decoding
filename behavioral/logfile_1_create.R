
#_____________________________________________________________________________________________________________________________________________
# Converter code for transforming logfiles made by Unreal during scanning for Schuck et al. (2015) into a data table for each participant. The
# data table will include:
#       1.  Time stamp
#       2.  Location x-coordinate
#       3.  Location y-coordinate
#       4.  Movement speed
#       5.  Direction angle calculated by consecutive location
#       6.  Direction angle calculated by YAW information
#       7.  Class of direction angle (1-6) based on YAW information
#       8.  TR number (uncorrected for hemodynamic lag)
#       9.  TR number (corrected for hemodynamic lag)
# This information will be used to train a classifier on viewing direction in each person and extract the accuracy of a decoder for each
# viewing direction.
#_____________________________________________________________________________________________________________________________________________


rm(list=ls())
library(ggplot2)
library(plotrix)
library(data.table)



#-----------------FUNCTIONS: Calculate direction_angle, calculate angle_class, calculate speed------------------------------------------------



#=====================Function to calculate angle the subject is facing while walking=========================================================

directionAngle <- function(coordX1, coordY1, coordX2, coordY2){
  
  # Function description:
  #   Use the consecutive positions of the subject (x1/y1) and (x2/y2) to determine the angle
  #     the subject was heading. 
  #     If a person did not move at all the value is -1
  
  # Calculated variables:
  deltaX <- 0
  deltaY <- 0
  headDirectionAngle <- 0
  
  # Calculate distances deltaX & deltaY the subject traveled on the x & y coordinate.
  deltaX <- as.numeric(coordX2) - as.numeric(coordX1)
  deltaY <- as.numeric(coordY2) - as.numeric(coordY1)
  
  # Calculate angle person is heading:
  #     Seven possible cases: Four cases + two special cases:
  #
  #         Case #1:  deltaX > 0, deltaY >= 0
  #         Case #2:  deltaX < 0. deltaY >= 0
  #         Case #3:  deltaX < 0, deltaY <= 0
  #         Case #4:  deltaX > 0, deltaY <= 0
  #
  #         Special case #1:    deltaX == 0, deltaY >= 0
  #         Special case #2:    deltaX == 0, deltaY <= 0
  #         Special case #3:    deltaX == 0, deltaY == 0
  
  
  #         Case #1:  deltaX > 0, deltaY >= 0
  if(deltaX > 0 & deltaY >= 0) {
    headDirectionAngle <- ((atan(deltaY/deltaX)*180)/pi) + 0;
    
    #         Case #2:  deltaX < 0. deltaY >= 0 
  } else if(deltaX < 0 & deltaY >= 0) {
    headDirectionAngle <- ((atan(deltaY/deltaX)*180)/pi) + 180;
    
    #         Case #3:  deltaX < 0, deltaY <= 0    
  } else if(deltaX < 0 & deltaY <= 0) {
    headDirectionAngle <- ((atan(deltaY/deltaX)*180)/pi) + 180;
    
    #         Case #4:  deltaX > 0, deltaY <= 0    
  } else if(deltaX > 0 & deltaY <= 0) {
    headDirectionAngle <- ((atan(deltaY/deltaX)*180)/pi) + 360;
    
    #         Special case #1:    deltaX == 0, deltaY >= 0     
  } else if(deltaX == 0 & deltaY > 0) {
    headDirectionAngle <- 90;
    
    #         Special case #2:    deltaX == 0, deltaY <= 0    
  } else if(deltaX == 0 & deltaY < 0) {
    headDirectionAngle <- 270;
    
    #         Special case #3:    deltaX == 0, deltaY == 0            ==> VALUE WILL BE -1 
  } else if(deltaX == 0 & deltaY == 0) {
    headDirectionAngle <- -1;
  }
  
  
  return(headDirectionAngle)
}



#=====================Function to classify direction angles into bins=========================================================================
# variable number of bins (subclasses), bins coded with a natural number 
classifyAngle <- function(angle, numberSubclasses, binShift){
  
  # Calculated output variable:
  angleSubclass <-0
  
  # for loop which checks if the angle is in the range between (0 to 360/x); (360/x to 2* 360/x); ... 
  #   If the angle is -1 (person did not move) ==> Angle category will be -1, too.
  #   If the conditions do not apply ==> Angle = 0!
  #   Because of binShift this had to be slightly adjusted (adding the bin shift to the angle as well as taking care of the last subclass
  #   since it surpasses the 360 ==> 0 border)
  for(i in 1:numberSubclasses){
    
    if(angle == -1){
      angleSubclass <- -1
      
    } else if(angle >= (0 + ((i-1)*360/numberSubclasses) + binShift) && angle < (i * (360/numberSubclasses) + binShift)){
      angleSubclass <- i
      
    } else if(i == numberSubclasses && angle >= (0 + ((i-1)*360/numberSubclasses) + binShift) || between(angle, 0, binShift)){
      angleSubclass <- i
      
    }
  }
  
  return(angleSubclass)
}


#=====================Function to calculate movement speed in UnrealUnits/ms==================================================================
directionSpeed <- function(coordX1, coordY1, coordX2, coordY2){
  
  # Calculate Euclidean distance between following Log-points per 100ms
  # dist = sqrt(sum((Vector1 - Vector2)^2))
  # Vector1 = c(x1,y1), Vector2 = c(x2,y2)
  # "as.numeric" because values in the locdata are "character" not "numeric"
  # "/100" because time between two logs = 100ms
  
  speed = (sqrt(sum((c(as.numeric(coordX1), as.numeric(coordY1))-c(as.numeric(coordX2),as.numeric(coordY2)))^2)))/100
  
  return(speed)
}




###----BEGIN CONVERTER CODE-------------------------------------------------------------------------------------------------------------------

# Loops through all "SAM" data in the named directory, reads the files, extracts location and YAW data
#   Special tweaks for direction_decoding:
#       - 'Transfer phase' and 'Encoding Phase' get cut off (because of missing YAW data and hard to interpret 
#               direction_decoding because of switching environments)
#       - direction_angle, angle_class and movement_speed get added to the location and YAW data

# Set path for log file data:
datadir = ''     # Path to behaviorla data

# Set path for newer BIDS dir:
bidsDir = ''

# Set path for imaging data:
trDir = ''

# find all files in the data directory
files = dir(datadir)
# split them by run 
files1 = files[grep('run1', files)]
files2 = files[grep('run2', files)]
# get n of subjects 
nid = max(c(length(files1), length(files2)))

# vectors to store subject numbers in:
ids = rep(0, length(files1))
ids2 = rep(0, length(files2))
# Extract subject digits from filename (e.g. SAM102_run1 ==> 102)
for (i in 1:length(files1)) {
  ids[i] = as.integer(substring(files1[i], 4, 6))
  if (i <= length(files2)) {
    ids2[i] = as.integer(substring(files2[i], 4, 6))
  }
}

# set up data arrays 
SAM = array(NA, c(40, 26, 2, nid)) # item, name + data, run, cid
SAM_OBJs = array(NA, c(5, 4, 2, nid))

# set up arrays for measurement of left data percentage when thresholded by 
#     minimum time walking into same direction
distContDirectionAll = c(NULL)
percentRowsLeftAll_5 = c(NULL)
percentRowsLeftAll_10 = c(NULL)
percentRowsLeftAll_15 = c(NULL)
percentRowsLeftAll_20 = c(NULL)
percentRowsLeftAll_25 = c(NULL)

# set up arrays for measurement of left data when all lines without YAW information, lines of walking backwards
#   and both are excluded (vs. the untouched data)
percRowsLeftYaw_All = c(NULL)
percRowsLeftBackwards_All = c(NULL)
percRowsLeftNoYawAndBackwards_All = c(NULL)

# set up vector for storing number of TRs for each person
nrTRsLogAll = c(NULL)
nrTRsNiftiAll = c(NULL)

#=====================OUTER LOOP: Runs========================================================================================================
# Use converter on all data for the first run, then the second run
x = y = env = NULL

for (run in 1:2) {
  # get files from current run 
  cfiles = eval(parse(text = paste('files', as.character(run), sep = '')))
  # arrays for storing start/end time of different trial phases 
  cuestart = cuestop = cuedrop = cueshow = matrix(nrow = 35, ncol = 1)
  pickupstart = pickupstop = matrix(nrow = 5, ncol = 1)
  
  
  
  #=====================INNER LOOP: Files=======================================================================================================
  # Use converter on each file of one run 
  for (f in 1:(length(cfiles))){
    
    # State name in command window to see progress of the script:
    print(cfiles[f])
    
    # id of current file   
    ID = as.integer(substring(cfiles[f], 4, 6))
    # id counter 
    cid = which(ID==ids)  
    
    
    
    #=====================Cutting off the transfer phase======================================================================================
    # read whole file then cut at 'curr trial' reaches 30 ==> Transfer phase begins that we want to exclude!
    alldata = readLines(paste(datadir, cfiles[f], sep = ''))
    
    # Where does 'curr trial' reach 30?
    trialidx = regexpr('curr trial = 30', alldata)
    trialidx = which(trialidx > 0)
    
    # Cut 'alldata' till 'curr trial = 30'
    alldata = alldata[1:trialidx[1]]
    
    
    #=====================Extract relevant data from the log file (loc, rep, YAW, orient)=====================================================
    # get position of location logs 
    locidx = regexpr('Location', alldata)
    locidx = which(locidx > 0)
    
    # get position of reposition logs  
    repidx = regexpr('Reposition', alldata)
    repidx = which(repidx > 0)
    
    # get position of all YAW logs
    yawOrientidx = regexpr('YawUnit', alldata)
    yawOrientidx = which(yawOrientidx > 0)
    
    # append both position vectors and sort them so location and reposition logs have the correct order   
    locRepYawOrientidx = sort(c(locidx, repidx, yawOrientidx), decreasing = FALSE)
    
    # extract location and reposition lines to separate variables
    locdata = alldata[locidx]
    repdata = alldata[repidx]
    yawOrientdata = alldata[yawOrientidx]
    
    # delete extracted lines from alldata
    alldata[-locRepYawOrientidx]
    
    
    
    #=====================Build matrix out of location and repostion lines====================================================================
    # Identify location and reposition logs by adding flag columns filled with 1's (locaion) and 2's (reposition)
    
    # parse data into relevant columns
    locdata = t(matrix(unlist(strsplit(locdata, "\\|")), nrow = 9))[,c(2, 5, 7)]
    # build a vector with length of locdata and fill it with 1s (to later identify all location logs)
    locFlag = vector(length = nrow(locdata))+1
    # bind this vector to locdata as 4th column
    locdata = cbind(locdata, locFlag)
    # convert to numeric
    locdata = apply(locdata, c(1, 2), function(x) as.numeric(x))
    
    # same for repdata
    repdata = t(matrix(unlist(strsplit(repdata, "\\|")), nrow = 11))[,c(2, 5, 7)]
    repFlag = vector(length = nrow(repdata))+2
    repdata = cbind(repdata, repFlag)
    repdata = apply(repdata, c(1, 2), function(x) as.numeric(x))
    
    # same for yawOrientdata (except we only need timestamp and "YAW")
    yawOrientdata = t(matrix(unlist(strsplit(yawOrientdata, "\\|")), nrow = 11))[,c(2, 9, 11)]
    yawOrientdata = apply(yawOrientdata, c(1, 2), function(x) as.numeric(x))
    
    
    
    # bind loc and rep data matrices together by rows and sort them based on the timestamp so locatio and 
    #     reposition entries are in the right order (NOTE: yawOrientdata cant be bound to them now because   
    #     it has similar timestamps as locdata, therefore cannot be sorted by time)
    locRepdata = (rbind(locdata, repdata))
    locRepdata = locRepdata[order(locRepdata[,1], decreasing = FALSE),]
    
    
    
    #=====================Adjust coordinates based on different environments==================================================================
    # probably not important for us, because we leave out the transfer phase which would switch the environments
    
    
    # mark trials which happend in different environments 
    env_xcenters = c(-175, 30417, 61575)
    env_ycenters = c(-175, -538, -158)
    cenv = unlist(lapply(locRepdata[,2], function(x) which.min(abs(as.numeric(x) - c(0, 30000, 60000)))))
    
    # adjust for off-center enviroments in different conditions 
    locRepdata[,2] = locRepdata[,2] - env_xcenters[cenv] 
    locRepdata[,3] = locRepdata[,3] - env_ycenters[cenv] 
    
    # make one large vector across al participants and conditions 
    x = c(x, locRepdata[,2])
    y = c(y, locRepdata[,3])
    env = c(env, cenv)	
    
    
    
    #=====================Create array of calculated angles, class code and movement speed====================================================
    # We will later calculate the direction based on YAW information, calculating it by consecutive location logs now is however 
    #     important for excluding backwards movement (comparison of looking and walking direction)
    
    
    # for-loop that takes coordinates out of the logfile and saves angles into new array:
    
    # Create array (with one column for calculated angle, one column for class-coded 
    #     angles and one column for movement speed, as many rows as the logfile) and fill it with 0's:
    angleArray <- array(0, c(length(locRepdata[,1]),3))
    
    # loop through locdata till the end of locdata rows:  
    for(i in 2:length(locRepdata[,1])) {
      
      # Separated cases to mark! 
      # direction_angle at reposition logs = "-2" because heading direction during reposition cannot be interpreted
      
      
      # Normal calculation for location logs (marked with 1 in the 4th column)
      if(locRepdata[i,4] ==1){
        # Calculate movement speed (units/ms) for travel between two log points and save it to angleArray
        angleArray[i,1] <- directionSpeed(locRepdata[i-1,2], locRepdata[i-1,3], locRepdata[i,2], locRepdata[i,3])
        
        # Calculate the angle for each delta between coordinates and save it to angleArray 
        #   (==> first entry of angleArray stays empty!)
        angleArray[i,2] <- directionAngle (locRepdata[i-1,2], locRepdata[i-1,3], locRepdata[i,2], locRepdata[i,3])
        
        # Look at value that was just calculated and entered into angleArray and write 
        #   the appropriate subclass (depending on the number of subclasses you chose) into second cloumn:
        angleArray[i,3] <- classifyAngle(angleArray[i,2],6,0)
        
        
        # Special angle calculation for reposition logs ==> Marked with "-2" in angle_class column for later exclusion
      } else if (locRepdata[i,4] == 2){
        # Calculate movement speed (units/ms) for travel between two log points and save it to angleArray
        angleArray[i,1] <- directionSpeed(locRepdata[i-1,2], locRepdata[i-1,3], locRepdata[i,2], locRepdata[i,3])
        
        # Calculate the angle for each delta between coordinates and save it to angleArray 
        #   (==> first entry of angleArray stays empty!)
        angleArray[i,2] <- directionAngle (locRepdata[i-1,2], locRepdata[i-1,3], locRepdata[i,2], locRepdata[i,3])
        
        # Look at value that was just calculated and entered into angleArray and write 
        #   the appropriate subclass (depending on the number of subclasses you chose) into second cloumn:
        angleArray[i,3] <- -2
      }
    }
    
    
    # Append the calculated angleArray columns to the end of the main locRepdata matrix ([,5:7])
    locRepdata <- cbind(locRepdata, angleArray[,1], angleArray[,2], angleArray[,3])
    
    
    
    #=====================Adding YAW data to the matrix=======================================================================================
    
    # Problem: Not as many YAW entries as location entries ==> Fill NULL vector the length of locRepdata with YAW data at the lines the
    # timestamp is equal
    
    # Set up vector the length of locRepdata with 2 columns, one for YAW and one for Orient:  (will be added to locRepdata in the end)
    yawdataAdd = as.numeric(vector(mode = 'double', length = nrow(locRepdata)))
    orientdataAdd = as.numeric(vector(mode = 'double', length = nrow(locRepdata)))
    
    # for-loop over lines of yawOrientdata to compare timestamps and put YAW data into locRepdata lines with the same time stamp
    for(m in 1:nrow(yawOrientdata)){
      sameTimestampidx = which(locRepdata[,1] == yawOrientdata[m,1])
      
      # Exclude all cases in which there is no same value
      if(length(sameTimestampidx) != 0){
        # Fill line with the same timestamp in empty vector with Yaw and Orient value
        yawdataAdd[sameTimestampidx] = yawOrientdata[m,2]
        orientdataAdd[sameTimestampidx] = yawOrientdata[m,3]
      }
      
    }
    
    # append matrix columns to end of locRepdata
    locRepdata = cbind(locRepdata, yawdataAdd, orientdataAdd)
    
    # replace 0s with NA because there is not orient/yaw data where it is 0
    locRepdata[which(locRepdata[,8] == 0),8] = NA
    locRepdata[which(locRepdata[,9] == 0),9] = NA
    
    
    
    #=====================Transform YAW data from UnrealUnits into degrees====================================================================
    # YAWs range from -32768 to 32768, range = 65536 (2^16)
    
    # Transform values into degrees from 0 to 360
    locRepdata[,8] = (locRepdata[,8]/65536)*360
    # Now the scale is -180 to 180! To keep YAW degrees and direction degrees in same phase transform negative data to represent 180 to 360 degrees
    # Now: 0-180: 0 to 32768 -------- 180-360: -32768 to 0
    locRepdata[which(locRepdata[,8] < 0),8] = locRepdata[which(locRepdata[,8] < 0),8] + 360
    
    
    
    #=====================Exclude all lines that don't have a YAW value=======================================================================
    
    # for sake of control: Log number of rows before excluding 
    #   (important if we want to know how many lines we lose through this):
    rowBeforeNoYawExcl = nrow(locRepdata)
    
    # If the entry in column 8 (YAW data) is NA (so there is no YAW information for this timestamp) 
    #     ==> Delete the whole line
    for(n in nrow(locRepdata):1){
      if(is.na(locRepdata[n,8])){
        locRepdata = locRepdata[-n,]
      }
    }
    
    # for sake of control: Log number of rows after excluding:
    #   (important if we want to know how many lines we lose through this):
    rowAfterNoYawExcl = nrow(locRepdata)
    
    # Calculate percent rows left after excluding:
    percRowsLeftYaw = (rowAfterNoYawExcl/rowBeforeNoYawExcl)*100
    
    
    # Move file to correct directory (If participant is excluded, so not in standard sirectory, move it to exclude directory)
    if(dir.exists(paste(bidsDir,as.character('derivatives/'), as.character('sub-young'), as.character(substr(cfiles[f], 5, 6)), as.character('/behavior/'), sep = ''))){
      dest = paste(bidsDir,as.character('derivatives/'), as.character('sub-young'), as.character(substr(cfiles[f], 5, 6)), as.character('/behavior/'), sep = '')
      file.rename(file, paste(dest, as.character('sub-young'), as.character(substr(cfiles[f], 5, 6)), as.character('_path_run'),
                              as.character(substr(cfiles[f], nchar(cfiles[f])-4, nchar(cfiles[f])-4)), as.character('.pdf'),  sep = ''))
    } else{
      dest = paste(bidsDir,as.character('derivatives/excludes/'), as.character('sub-young'), as.character(substr(cfiles[f], 5, 6)), as.character('/behavior/'), sep = '')
      file.rename(file, paste(dest, as.character('sub-young'), as.character(substr(cfiles[f], 5, 6)), as.character('_path_run'),
                              as.character(substr(cfiles[f], nchar(cfiles[f])-4, nchar(cfiles[f])-4)), as.character('.pdf'),  sep = ''))
    }
    
    

		
		
		
		#=====================Exclude all lines with a large phase shift between YAW data and walking direction===================================
		# This will eliminate every log where participants walked backwards and other 
		#       lines where walking and viewing direction did not accord
		
		# FIXED PHASE SHIFT THRESHOLD: 20
		
		# for sake of control: Log number of rows before excluding: (Important if we want to know how many 
		#     lines we lose through this)
		rowBeforeBackwardsExcl = nrow(locRepdata)
		
		# Allocate vector to get all events of backwards walking
		# Get index of backwards entries (120 to really get only backwards movement and not e.g. shifts between
		# YAW and movement direction at boundaries)
		
		# backwards_idx = which(abs(locRepdata[,6] - locRepdata[,8]) > 20)
		backwards_idx = which(between(abs(locRepdata[,6] - locRepdata[,8]), 160, 200))
		if(length(backwards_idx) > 0){
		  locRepdata_backwards = locRepdata[backwards_idx,]
		  # In case there are no backwards events: To avoid errors just calculate with the normal vector and in the
		  # end don't save it
		} else{
		  locRepdata_backwards = locRepdata
		}
		
		# Exclude the whole line of code if the phase shift between YAW and walking direction is greater than threshold
		# Iterate loop backwards so when a line is excluded it does not change the index of the next line to exclude
		# We have to consider the circularity: e.g. YAW says 350 while locations say 10 would technically be
		# in our range (20) BUT a larger absolute difference than 20 (namely 340). Every difference BETWEEN 20 and
		# 340 therefore is a problem
		for(o in nrow(locRepdata):1){
		  # if(abs(locRepdata[o,6] - locRepdata[o,8]) > 20){
		  if(between(abs(locRepdata[o,6] - locRepdata[o,8]), 20, 340)){
		    locRepdata = locRepdata[-o,]
		  }
		}
		
		
		# for sake of control: Log number of rows after excluding: (Important if we want to know how many 
		#     lines we lose through this)
		rowAfterBackwardExcl = nrow(locRepdata)
		  
		# Calculate percent rows left after excluding:
		percRowsLeftBackwards = (rowAfterBackwardExcl/rowBeforeBackwardsExcl)*100
		# Calculate percent rows left after excluding NoYAw and Backwards
		percRowsLeftNoYawAndBackwards = (rowAfterBackwardExcl/rowBeforeNoYawExcl)*100
		
		
		
		#=====================Adjust time stamps to first scanner pulse===========================================================================
		# Because we don't use the ecoding phase the first scanner pulse is two scanner pulses before the pulse that 
		#         is at the same time as  the 'Begin Phase2' log.
		#         There are three first scanner pulses the scanner made, we always take the first one of them.
		
		# Timestamp at first scanner pulse = new zero (substract it from original time stamps)
		
		# find all lines where the logfile says 'StartPhase2' (Because the third of the three first scanner pulses 
		#     of phase 2 is exactly one line before this one)
		pulseidx = regexpr('StartPhase2', alldata)
		pulseidx = which(pulseidx > 0)
		  
		# get the logfile line one before 'start phase 2' to get the third of the three first scanner pulses
	  thirdPulseidx = (pulseidx[1]-1)
		# Look for this line in all lines where it says 'Scanner Pulse'
		allPulseidx = regexpr('Scanner Pulse', alldata)
		allPulseidx = which(allPulseidx > 0)
	  firstPulseidx = which(allPulseidx == thirdPulseidx)
		# Then go two steps back to get the line in alldata of the first of the three scanner pulses  
		timeFirstPulse = allPulseidx[firstPulseidx-2] 
		timeFirstPulse = alldata[timeFirstPulse]
		
		# Form this line into a vector (columns separated by '|') and extract the value in the second column (time) as numeric
		timeFirstPulse = as.numeric(t(matrix(unlist(strsplit(timeFirstPulse, "\\|")), nrow = 9))[2])
		  
		# Substract time of first scanner pulse from timestamp in locRepdata so first scanner pulse is new zero point
		locRepdata[,1] = locRepdata[,1] - timeFirstPulse
		
		# Same for backwards movement
		locRepdata_backwards[,1] = locRepdata_backwards[,1] - timeFirstPulse

		
		
		#=====================Adjusting for time drift of Unreal==================================================================================
		# The scanner clock and the Unreal clock (which is shown in the logfile) drift apart in time with the factor of around 0.0011
		# Therefore we have to multiply our time stamps by 1.0011 to adjust for the time drift.
		
		locRepdata[,1] = locRepdata[,1]*1.0011
		
		# Same for backwards movement
		locRepdata_backwards[,1] = locRepdata_backwards[,1]*1.0011
		
		
		#=====================Adding TR space to the data matrix==================================================================================
		# add extra column that states the number of the TR at a certain time

		# Set up vector to store TRs in (will later be appended to data matrix)
		trVec = NULL
		trVec_backwards = NULL
		
		# For loop; loop from 2.4 to last time-stamp in locRepdata in steps of 2.4 (TR in seconds)
		#     loop starts ar 2.4 because in the loop we reference the time 2.4s before the actual trTime in the loop iteration
		#     so if we would start at 0 we would reference -2.5 which is not possible
		for(trTime in seq(from= 2.4, to = locRepdata[nrow(locRepdata),1], by = 2.4)){

		  # Index showing each line in which the time stamp of data matrix is between the actural TR in this iteration and the TR before
		  trIdx = which(locRepdata[,1] < trTime & locRepdata[,1] >= (trTime-2.4))
		  # fill all lines for the which function is TRUE with the value of trTime (by deviding it by 2.4 later we will get the actual TR number)
		  trVec[trIdx] = trTime
		}
		
		
		# Deviding by 2.4 so we get TR number
		trVec = (trVec/2.4)
		
		# Same for backwards movement
		for(trTime in seq(from= 2.4, to = locRepdata_backwards[nrow(locRepdata_backwards),1], by = 2.4)){
		  
		  # Index showing each line in which the time stamp of data matrix is between the actural TR in this iteration and the TR before
		  trIdx = which(locRepdata_backwards[,1] < trTime & locRepdata_backwards[,1] >= (trTime-2.4))
		  # fill all lines for the which function is TRUE with the value of trTime (by deviding it by 2.4 later we will get the actual TR number)
		  trVec_backwards[trIdx] = trTime
		}		
		
		#----
		
		# The iteration of the for loop leaves a rest between the last TR and the end of the data matrix 
		#   (except the number of rows is a multiple of 2.4)
		
		# In the case that entries in locRepdata are not a multiple of 2.4 (i.e. there is a rest remaining)
		if(length(trVec[length(trVec):nrow(locRepdata)]) > 1){
		  # fill the remaining empty lines with an additional TR (TR from before +1)
		  #-
		  # trVec[(length(trVec)+1):nrow(locRepdata)] = trVec[length(trVec)] +1
		  #-
		  trVec[(length(trVec)+1):nrow(locRepdata)] = floor(locRepdata[nrow(locRepdata), 1]/2.4) +1
		}
		
		# Same for backwards movement
		if(length(trVec_backwards[length(trVec_backwards):nrow(locRepdata_backwards)]) > 1){
		  # fill the remaining empty lines with an additional TR (TR from before +1)
		  #-
		  # trVec[(length(trVec)+1):nrow(locRepdata)] = trVec[length(trVec)] +1
		  #-
		  trVec_backwards[(length(trVec_backwards)+1):nrow(locRepdata_backwards)] = (
		    floor(locRepdata_backwards[nrow(locRepdata_backwards), 1]/2.4) +1)
		}		
		
		# Create an extra vector which shows the TRs shifted by 2 (ca. 5s) (adjusting for hemodynamic lag)
		trVecShifted = trVec + 2
		trVecShifted_backwards = trVec_backwards +2
		
		# Bind both vectors as columns to locRepdata
		locRepdata = cbind(locRepdata, trVec, trVecShifted)
		locRepdata_backwards = cbind(locRepdata_backwards, trVec_backwards, trVecShifted_backwards)
	
		
				
		#=====================EXCLUDE every row before first scanner pulse========================================================================
    # Exclude every row before first scanner pulse:
		#   (first check if there are any because if there are none, all lines will be erased)
		if(length(which(locRepdata[,1] < 0)) > 0){
		  locRepdata = locRepdata[-(which(locRepdata[,1] < 0)),]
		}
		
		# Same for backwards movement
		if(length(which(locRepdata_backwards[,1] < 0)) > 0){
		  locRepdata_backwards = locRepdata_backwards[-(which(locRepdata_backwards[,1] < 0)),]
		}		
		
		
		#=====================EXCLUDE every row with angle_class -1 or -2 (logs without movement and logs with reposition)========================
		specialAngleClassidx = which(locRepdata[,7] < 0)
		# Check if there are any cases (because if there are 0 cases ==> locRepdata = locRepdata[-0,] leaves an empty matrix)
		if(length(specialAngleClassidx) > 0){
		  locRepdata = locRepdata[-(specialAngleClassidx),]
		}
		
		# Same for backwards movement
		specialAngleClassidx_backwards = which(locRepdata_backwards[,7] < 0)
		# Check if there are any cases (because if there are 0 cases ==> locRepdata = locRepdata[-0,] leaves an empty matrix)
		if(length(specialAngleClassidx_backwards) > 0){
		  locRepdata_backwards = locRepdata_backwards[-(specialAngleClassidx_backwards),]
		}		
		
		
		
		#=====================Recalculate angle_class based on YAW data===========================================================================
		# Has to be done after excluding angleClass -1/-2 because otherwise they will be lost 
		#     and the lines cannot be excluded based on it
		# NOT for the backwards movement since the movement direction is the entity we want to classify
		for(p in 1:nrow(locRepdata)){
		  locRepdata[p,7] = classifyAngle(locRepdata[p,8], 6,0)
		}
		
		
		
		#=====================Calculate distribution of continuous movement in the same direction=================================================
		# Reset location index of angle change for each loop and set first entry to one (important for difference calculation)
		angleChangeidx = c(1)
		
		# Look for differences in the angle_bin column (thats where definite event changes are)
		angleChangeidx = as.numeric(diff(locRepdata[,7]) != 0) # muss noch ne 1 davor und +1 gerechnet werden damit wie vorher!
		angleChangeidx_backwards = as.numeric(diff(locRepdata_backwards[,7]) != 0)
		
		# Look for differences in the time column that are greater than our threshold (if threshold is surpassed, even continueing
		# in the same direction is considered a separate event) (e.g. walks into direction 1, pauses for 0.7s, continues in dir1 = separate events)
		timeJumpidx = as.numeric(diff(locRepdata[,1]) > 0.5)
		timeJumpidx_backwards = as.numeric(diff(locRepdata_backwards[,1]) > 0.5)
		
		# Because this is only important if the subject continues in the same direction: Exclude all indexes for lines in which the direction changed)
		timeJumpidx[which(angleChangeidx > 0)] = 0
		timeJumpidx_backwards[which(angleChangeidx_backwards > 0)] = 0
		
		# Create new index of event change bringing both types of event changes together (sorting needed to make the indicated lines consecutive)
		eventChangeidx = append(which(angleChangeidx > 0), which(timeJumpidx > 0))
		eventChangeidx = sort(eventChangeidx)
		
		eventChangeidx_backwards = append(which(angleChangeidx_backwards > 0), which(timeJumpidx_backwards > 0))
		eventChangeidx_backwards = sort(eventChangeidx_backwards)
		
		# Transform eventChangeidx so it fits for the previous code (eventChangeidx was ADDED after bad performance of another approach)
		eventChangeidx = append(c(0), (eventChangeidx)) +1
		eventChangeidx_backwards = append(c(0), (eventChangeidx_backwards)) +1
		

		
		# make vector that contains the length of each path in the same direction by calculating the difference
		#     between the original vector and another vector in which each entry was moved one to the left and the last line
		#     was replaced with number of rows in locRepdata
		distContDirection = append(eventChangeidx[-1], nrow(locRepdata)) - eventChangeidx
		
		# Example: (5, 9, 13, 20, 25) - (1, 5, 9, 13, 20) = (4, 4, 4, 7, 5) ==> Vector of number of logs while walking in same direction
		
		# Precaution: Exclude everything that has a 0 in the distContDirection
		distContDirection = distContDirection[which(distContDirection > 0)]
		
		# Make a vector across all participants that contains their direction distribution
		distContDirectionAll = append(distContDirectionAll, distContDirection)
		
		
		
		#=====================Create new data matrices in which only consecutive movement in the same direction remains===========================
		# Threshold of minimum time of walking in the same direction in bins: 500ms, 1s, 1.5s, 2s, 2.5s
		
		
    # for-loop: Excluding rows with less than a certain amount of travel time in the same direction
		#     loop iterations change the threshold:
		for(thresholdContDirection in c(5, 10, 15, 20, 25)){
		  
		  # Create new matrix so the amount of data left can be compared to the original data set
		  locRepdataExcl = locRepdata
		  
		  # loop through angle change index till end of rows and if there is not a certain amount of logs (each 100ms) 
		  #     between two angle changes ==> exclude them
		  for(l in length(eventChangeidx):2){
		    
		    # Always the FIRST CASE: you have to calculate distance between last line of eventChangeidx (l) and the last line of the
		    #     data matrix!
		    # eventChangeidx connot check this itself because the last line of the data matix cant change its angle
		    
		    # exclude everthing under x secs of continuous direction
		    if((l == length(eventChangeidx) && (nrow(locRepdataExcl) - eventChangeidx[l] < thresholdContDirection))){   
		      locRepdataExcl = locRepdataExcl[-(nrow(locRepdataExcl) : eventChangeidx[l]),]
		    }
		    
		    # USUAL CASE: calculate distance between angleChange indexes (substract 1 from first eventChangeidx because eventChangeidx 
		    #   states the lines were the new (changed) angle is in
		    else if((eventChangeidx[l] -1) - eventChangeidx[l-1] < thresholdContDirection){         
		      
		      locRepdataExcl = locRepdataExcl[-((eventChangeidx[l]-1):eventChangeidx[l-1]),]
		    }
		    
		    # SPECIAL CASE: only one entry of a certain angle so the eventChangeidxs are following numbers 
		    #   ==> Can be excluded automatically whithout threshold
		    else if((eventChangeidx[l] -1) == eventChangeidx[l-1]){
		      locRepdataExcl = locRepdataExcl[-(eventChangeidx[l]-1),]
		    }
		    
		  }
		  
		  
		  # Comparison between remaining data sets of different thresholds and the original data set (in percent):
		  percentRowsLeft = (nrow(locRepdataExcl)/nrow(locRepdata) * 100)
		  
		  # Before loop with a certain threshold ends ==> assign it to a variable named 'percentRowsLeft_' with the threshold 
		  #   at the end of it (for each single person)
		  assign(paste('percentRowsLeft','_',as.character(thresholdContDirection), sep = ''), percentRowsLeft)
		  # Do the same for the whole remaining data matrix:
		  assign(paste('locRepdataExcl','_',as.character(thresholdContDirection), sep = ''), locRepdataExcl)
		}
		
		#====================== Same for backwards movement BUT with only 1s continuous movement ==========================
		# Create new matrix so the amount of data left can be compared to the original data set
		locRepdataExcl_backwards = locRepdata_backwards
		
		# loop through angle change index till end of rows and if there is not a certain amount of logs (each 100ms) 
		#     between two angle changes ==> exclude them
		for(l in length(eventChangeidx_backwards):2){
		  
		  # Always the FIRST CASE: you have to calculate distance between last line of eventChangeidx (l) and the last line of the
		  #     data matrix!
		  # eventChangeidx connot check this itself because the last line of the data matix cant change its angle
		  
		  # exclude everthing under x secs of continuous direction
		  if((l == length(eventChangeidx_backwards) && (nrow(locRepdataExcl_backwards) - eventChangeidx_backwards[l] < 10))){   
		    locRepdataExcl_backwards = locRepdataExcl_backwards[-(nrow(locRepdataExcl_backwards) : eventChangeidx_backwards[l]),]
		  }
		  
		  # USUAL CASE: calculate distance between angleChange indexes (substract 1 from first eventChangeidx because eventChangeidx 
		  #   states the lines were the new (changed) angle is in
		  else if((eventChangeidx_backwards[l] -1) - eventChangeidx_backwards[l-1] < 10){         
		    
		    locRepdataExcl_backwards = locRepdataExcl_backwards[-((eventChangeidx_backwards[l]-1):eventChangeidx_backwards[l-1]),]
		  }
		  
		  # SPECIAL CASE: only one entry of a certain angle so the eventChangeidxs are following numbers 
		  #   ==> Can be excluded automatically whithout threshold
		  else if((eventChangeidx_backwards[l] -1) == eventChangeidx_backwards[l-1]){
		    locRepdataExcl_backwards = locRepdataExcl_backwards[-(eventChangeidx_backwards[l]-1),]
		  }
		}
		
#------------------REMARK---------------------------------------------------------------------------------------------------------------------
#_____________________________________________________________________________________________________________________________________________
		#REMARK:
		# From now on we will use locRepdataExcl_10 to make sure we only include paths that were walked for at least 1s in the same direction!
		#_________________________________________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________________________________________

		
		#=====================Restructure data matrix to the format we need=======================================================================
		# Extract Time, X, Y, movement speed, angle by consecutive location, angle by YAW, angle bin, TR number without correction and TR number
		#     corrected for hemodynamic lag and put it in a new matrix
		locRepdataExcl_10_export <- locRepdataExcl_10[,c(1:3,5:6,8,7,10,11)]
		locRepdataExcl_10_export_backwards = locRepdataExcl_backwards[,c(1:3,5:6,8,7,10,11)]
		
		# Name the columns of the matrix accodringly
		colnames(locRepdataExcl_10_export) <- c('Time',
		                                        'x coordinate',
		                                        'y coordinate',
		                                        'movement speed',
		                                        'direction angle by location',
		                                        'direction angle by YAW',
		                                        'angle bin',
		                                        'TR number uncorrected',
		                                        'TR number corrected')
		colnames(locRepdataExcl_10_export_backwards) <- c('Time',
		                                                  'x coordinate',
          		                                        'y coordinate',
          		                                        'movement speed',
          		                                        'direction angle by location',
          		                                        'direction angle by YAW',
          		                                        'angle bin',
          		                                        'TR number uncorrected',
          		                                        'TR number corrected')
		
		
		
		#=====================Extract locRepdataExcl_10 as data table named after BIDS standard=============================================
		# Target directory: 'SAM_direction_decoding' folder in datadir
		
		# Important for reading the file with Excel:
		#   Column separator:     '\t' (tab separated)
		#   Decimal separator:    '.'
		#   Thousands separator:  ''
		# write.table(locRepdataExcl_10_export,
		 #            file = paste(datadir,as.character('SAM_direction_decoding/'), as.character('direction_decoding_'), cfiles[f],  sep = ''),
		  #           sep = '\t',
		   #          quote = FALSE,
		    #         col.names = TRUE,
		     #        row.names = FALSE)

		 # Write .tsv table with BIDS conform naming
		 # For younger subjects
		 if(substr(cfiles[f],4,4) == 1){
		   
		   file = paste(bidsDir,as.character('derivatives/'),
		                as.character('sub-young'), as.character(substr(cfiles[f], 5, 6)), as.character('_3d'),
		                as.character(substr(cfiles[f], nchar(cfiles[f])-4, nchar(cfiles[f])-4)), as.character('.tsv'),  sep = '')
		   
		   write.table(locRepdataExcl_10_export,
		               file = file,
		               sep = '\t',
		               quote = FALSE,
		               col.names = TRUE,
		               row.names = FALSE)
		   
		   # Move file to correct directory (sensitive for excluded participants)
		   if(dir.exists(paste(bidsDir,as.character('derivatives/'), as.character('sub-young'), as.character(substr(cfiles[f], 5, 6)), as.character('/behavior/'), sep = ''))){
		     
		     dest = paste(bidsDir,as.character('derivatives/'), as.character('sub-young'), as.character(substr(cfiles[f], 5, 6)), as.character('/behavior/'), sep = '')
		     
		     file.rename(file, paste(dest, as.character('sub-young'), as.character(substr(cfiles[f], 5, 6)), as.character('_3d'),
		                             as.character(substr(cfiles[f], nchar(cfiles[f])-4, nchar(cfiles[f])-4)), as.character('.tsv'),  sep = ''))
		   } else{
		     
		     dest = paste(bidsDir,as.character('derivatives/excludes/'), as.character('sub-young'), as.character(substr(cfiles[f], 5, 6)), as.character('/behavior/'), sep = '')
		     
		     file.rename(file, paste(dest, as.character('sub-young'), as.character(substr(cfiles[f], 5, 6)), as.character('_3d'),
		                             as.character(substr(cfiles[f], nchar(cfiles[f])-4, nchar(cfiles[f])-4)), as.character('.tsv'),  sep = ''))
		   }
		   
		   
		 }
		 # For older subjects
		 else if(substr(cfiles[f],4,4) == 2){
		   
		   file = paste(bidsDir,as.character('derivatives/'),
		                as.character('sub-old'), as.character(substr(cfiles[f], 5, 6)), as.character('_3d'),
		                as.character(substr(cfiles[f], nchar(cfiles[f])-4, nchar(cfiles[f])-4)), as.character('.tsv'),  sep = '')

		   write.table(locRepdataExcl_10_export,
		               file = file,
		               sep = '\t',
		               quote = FALSE,
		               col.names = TRUE,
		               row.names = FALSE)
		   
		   # Move file to correct directory (sensitive for excluded participants)
		   if(dir.exists(paste(bidsDir,as.character('derivatives/'), as.character('sub-old'), as.character(substr(cfiles[f], 5, 6)), as.character('/behavior/'), sep = ''))){
		     
		     dest = paste(bidsDir,as.character('derivatives/'), as.character('sub-old'), as.character(substr(cfiles[f], 5, 6)), as.character('/behavior/'), sep = '')
		     
		     file.rename(file, paste(dest, as.character('sub-old'), as.character(substr(cfiles[f], 5, 6)), as.character('_3d'),
		                             as.character(substr(cfiles[f], nchar(cfiles[f])-4, nchar(cfiles[f])-4)), as.character('.tsv'),  sep = ''))
		   } else{
		     
		     dest = paste(bidsDir,as.character('derivatives/excludes/'), as.character('sub-old'), as.character(substr(cfiles[f], 5, 6)), as.character('/behavior/'), sep = '')
		     
		     file.rename(file, paste(dest, as.character('sub-old'), as.character(substr(cfiles[f], 5, 6)), as.character('_3d'),
		                             as.character(substr(cfiles[f], nchar(cfiles[f])-4, nchar(cfiles[f])-4)), as.character('.tsv'),  sep = ''))
		   }
		}
		
		#=====================Export also backwards walking============================
		# Write .tsv table with BIDS conform naming
		# For younger subjects
		if(substr(cfiles[f],4,4) == 1){
		  
		  file = paste(bidsDir,as.character('derivatives/'),
		               as.character('sub-young'), as.character(substr(cfiles[f], 5, 6)), as.character('_3d'),
		               as.character(substr(cfiles[f], nchar(cfiles[f])-4, nchar(cfiles[f])-4)), as.character('_backwards.tsv'),  sep = '')
		  
		  write.table(locRepdataExcl_10_export_backwards,
		              file = file,
		              sep = '\t',
		              quote = FALSE,
		              col.names = TRUE,
		              row.names = FALSE)
		  
		  # Move file to correct directory (sensitive for excluded participants)
		  if(dir.exists(paste(bidsDir,as.character('derivatives/'), as.character('sub-young'), as.character(substr(cfiles[f], 5, 6)), as.character('/behavior/'), sep = ''))){
		    
		    dest = paste(bidsDir,as.character('derivatives/'), as.character('sub-young'), as.character(substr(cfiles[f], 5, 6)), as.character('/behavior/'), sep = '')
		    
		    file.rename(file, paste(dest, as.character('sub-young'), as.character(substr(cfiles[f], 5, 6)), as.character('_3d'),
		                            as.character(substr(cfiles[f], nchar(cfiles[f])-4, nchar(cfiles[f])-4)), as.character('_backwards.tsv'),  sep = ''))
		  } else{
		    
		    dest = paste(bidsDir,as.character('derivatives/excludes/'), as.character('sub-young'), as.character(substr(cfiles[f], 5, 6)), as.character('/behavior/'), sep = '')
		    
		    file.rename(file, paste(dest, as.character('sub-young'), as.character(substr(cfiles[f], 5, 6)), as.character('_3d'),
		                            as.character(substr(cfiles[f], nchar(cfiles[f])-4, nchar(cfiles[f])-4)), as.character('_backwards.tsv'),  sep = ''))
		  }
		  
		  
		}
		# For older subjects
		else if(substr(cfiles[f],4,4) == 2){
		  
		  file = paste(bidsDir,as.character('derivatives/'),
		               as.character('sub-old'), as.character(substr(cfiles[f], 5, 6)), as.character('_3d'),
		               as.character(substr(cfiles[f], nchar(cfiles[f])-4, nchar(cfiles[f])-4)), as.character('_backwards.tsv'),  sep = '')
		  
		  write.table(locRepdataExcl_10_export_backwards,
		              file = file,
		              sep = '\t',
		              quote = FALSE,
		              col.names = TRUE,
		              row.names = FALSE)
		  
		  # Move file to correct directory (sensitive for excluded participants)
		  if(dir.exists(paste(bidsDir,as.character('derivatives/'), as.character('sub-old'), as.character(substr(cfiles[f], 5, 6)), as.character('/behavior/'), sep = ''))){
		    
		    dest = paste(bidsDir,as.character('derivatives/'), as.character('sub-old'), as.character(substr(cfiles[f], 5, 6)), as.character('/behavior/'), sep = '')
		    
		    file.rename(file, paste(dest, as.character('sub-old'), as.character(substr(cfiles[f], 5, 6)), as.character('_3d'),
		                            as.character(substr(cfiles[f], nchar(cfiles[f])-4, nchar(cfiles[f])-4)), as.character('_backwards.tsv'),  sep = ''))
		  } else{
		    
		    dest = paste(bidsDir,as.character('derivatives/excludes/'), as.character('sub-old'), as.character(substr(cfiles[f], 5, 6)), as.character('/behavior/'), sep = '')
		    
		    file.rename(file, paste(dest, as.character('sub-old'), as.character(substr(cfiles[f], 5, 6)), as.character('_3d'),
		                            as.character(substr(cfiles[f], nchar(cfiles[f])-4, nchar(cfiles[f])-4)), as.character('_backwards.tsv'),  sep = ''))
		  }
		}		

		
		#=====================Create data vectors of remaining data percentage f each person after exclusion processes============================
		# Concatenate each amount of data left (for each single person) to a vektor including all people 
		#   (like a data point for each person so we can calculate the distribution)
		percentRowsLeftAll_5 = c(percentRowsLeftAll_5, percentRowsLeft_5)
		percentRowsLeftAll_10 = c(percentRowsLeftAll_10, percentRowsLeft_10)
		percentRowsLeftAll_15 = c(percentRowsLeftAll_15, percentRowsLeft_15)
		percentRowsLeftAll_20 = c(percentRowsLeftAll_20, percentRowsLeft_20)
		percentRowsLeftAll_25 = c(percentRowsLeftAll_25, percentRowsLeft_25)
		
		# Make a data frame out of the percent vektors with a column for every threshold
		dfPercentRowsLeft = data.frame('th5' = percentRowsLeftAll_5, 
		                               'th10' = percentRowsLeftAll_10, 
		                               'th15' = percentRowsLeftAll_15, 
		                               'th20' = percentRowsLeftAll_20, 
		                               'th25' = percentRowsLeftAll_25)
		
		
		# Same process for left rows after excluding lines with no YAW information & walking backwards
		percRowsLeftYaw_All = c(percRowsLeftYaw_All, percRowsLeftYaw)
		percRowsLeftBackwards_All = c(percRowsLeftBackwards_All, percRowsLeftBackwards)
		percRowsLeftNoYawAndBackwards_All = c(percRowsLeftNoYawAndBackwards_All, percRowsLeftNoYawAndBackwards)
		
		
		
		#=====================Create a data vector of number of TRs calculated by time-stamp and given by scanner=================================
		# Extract number of TRs based on time-stamp at the end of the feedback phase
		nrTRsLog = locRepdata[nrow(locRepdata),1]/2.4
		
		nrTRsLogAll = rbind(nrTRsLogAll, nrTRsLog)
		
		# Extract number of nifti files in the "run_test" folders of each person (TRs  given by scanner)
		nrTRsNifti = length(list.files(path = paste(trDir, as.character(ID), '/run_test', as.character(run), sep = ''),
		                               pattern = '.nii'))
		nrTRsNiftiAll = c(nrTRsNiftiAll, nrTRsNifti)
	}	
}

