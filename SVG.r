get_iSVG <- function(SVG_data, i, target_np=30, alpha=0, tol.hor=90, 
                     last_max=TRUE, max_bnd=FALSE, cressie=FALSE){
    
    library(reshape2)

    j <- SVG_data[c(1:5,i)] 
    j = j[complete.cases(j[,]),]
    coordinates(j) = ~X+Y
    row.names(j) <- j$RG
   
    dist <- apply(coordinates(j), 1, 
                  function(eachPoint) spDistsN1(coordinates(j),
                                                eachPoint))
    l = sort(melt(dist)$value)[c((dim(j)[1]+1):(dim(j)[1])^2)]
    if(max_bnd){l = l[l<max_bnd]}
    index = target_np*2*c(1:(length(l)/(target_np*2)))
    bnd = l[index]
    if(max(l) > max(bnd)){bnd = c(bnd, tail(l, 1))}
    print(bnd)

    v = variogram(j[[1]]~1, j, boundaries=bnd, alpha=alpha, tol.hor=tol.hor, cressie=cressie)
    if(last_max){
        for(i in (2:length(v$gamma))){
            if(v$gamma[i]< v$gamma[i-1]){v$gamma[i] <- v$gamma[i-1]}
            }
        }
                      
    if(min(v$np)<25){
        foo = cumsum(v$np)
        bnd=0
        while(max(foo)>=30){
            bnd = c(bnd,v$dist[foo>=30][1])
            foo = foo-30
        }
        bnd = c(bnd, 60)
        if(length(bnd)<3){return('')}
        v = variogram(j[[1]]~1, j, boundaries=bnd, alpha=alpha, tol.hor=tol.hor, cressie=cressie)
        if(last_max){
            for(i in (2:length(v$gamma))){if(v$gamma[i]< v$gamma[i-1]){v$gamma[i] <- v$gamma[i-1]}}
            }
    }
    return(v)
}
                  

get_SVG <- function(SVG_data, last_max=TRUE, cressie=FALSE){

    library(gstat)
    library(sp)

    coordinates(SVG_data) = ~X+Y
    row.names(SVG_data) <- SVG_data$RG

    startcol = 4
    ncols = dim(SVG_data)[2]

    # Initialize the SVG
    v = variogram(SVG_data[[startcol]]~1, SVG_data, cressie=cressie)
    SVG_tab = data.frame(np=v$np, dist=v$dist)
    if(last_max){
        for(i in (2:length(v$gamma))){
            if(v$gamma[i]< v$gamma[i-1]){v$gamma[i] <- v$gamma[i-1]}
            }
        }
    SVG_tab[[names(SVG_data[startcol])]] = v$gamma

    # add the rest of the values
    for(i in (startcol+1):ncols){
        v = variogram(SVG_data[[i]]~1, SVG_data, cressie=cressie)
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
