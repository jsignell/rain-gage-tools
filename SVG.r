get_SVG <- function(SVG_data, last_max=TRUE){

    library(gstat)
    library(sp)

    coordinates(SVG_data) = ~X+Y
    row.names(SVG_data) <- SVG_data$RG

    bnd = c(0, 5.2, 7.7, 10, 12.4, 16, 19.2, 25)

    startcol = 4
    ncols = dim(SVG_data)[2]

    # Initialize the SVG
    v = variogram(SVG_data[[startcol]]~1, SVG_data)
    SVG_tab = data.frame(np=v$np, dist=v$dist)
    if(last_max){
        for(i in (2:length(v$gamma))){
            if(v$gamma[i]< v$gamma[i-1]){v$gamma[i] <- v$gamma[i-1]}
            }
        }
    SVG_tab[[names(SVG_data[startcol])]] = v$gamma

    # add the rest of the values
    for(i in (startcol+1):ncols){
        v = variogram(SVG_data[[i]]~1, SVG_data)
        if(last_max){
            for(j in (2:length(v$gamma))){
                if(v$gamma[j]< v$gamma[j-1]){v$gamma[j] <- v$gamma[j-1]}
                }
        }
        SVG_tab[[names(SVG_data[i])]] = v$gamma
        }
    return(SVG_tab)
}

plot_SVG <- function(SVG_tab){
    
    library(reshape)
    library(ggplot2)
    
    # reshape the data so that it can all be plotted on the same figure
    df <- SVG_tab[,2:dim(SVG_tab)[2]]
    df <- melt(df,id.vars = 'dist')
    # plot on same grid, each series colored differently -- 
    # good if the series have same scale
    p <- ggplot(df, aes(dist,value)) + geom_point(aes(colour = variable))+ theme(aspect.ratio=1)
    
    return(p)
}
