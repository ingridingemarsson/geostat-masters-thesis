def CreateListOfLinkfilesInSpan(firstYear, firstMonth, lastYear, lastMonth):
    year = firstYear
    month = firstMonth

    dirlist = []

    while(year*10**2+month<=lastYear*10**2+lastMonth):
        dirname = 'linkfile20' + str(year) + '-' + str(month).zfill(2)
        print(dirname)
        print(month)
        dirlist.append(dirname)
        month+=1
        if(month>12):
            month=1
            year+=1
    
    return(dirlist)