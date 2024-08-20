# Import libraries
suppressMessages({
    library(data.table) # data wrangling
    library(dplyr) # data wrangling
    library(tidyr) # data wrangling
    library(ggplot2)
    library(maps)
    library(sf) # spatial features
    library(sp) # spatial features
    library(rmapshaper) # simplify
})

# Function to read and prepare data
prepare_data <- function(edge_file, node_file, custom_crs){
    edges <- fread(edge_file, col.names = c("from_id", "to_id", "edge_weight"))
    nodes <- fread(node_file, col.names = c("Longitude", "Latitude","N")) %>% mutate(V1 = row_number() - 1)
    
    # Convert necessary columns to integer
    edges$from_id <- as.integer(edges$from_id)
    edges$to_id <- as.integer(edges$to_id)
    nodes$V1 <- as.integer(nodes$V1)
    
    # Join edges and nodes data to get the start and end points of each edge
    edges <- edges %>%
        left_join(nodes, by = c("from_id" = "V1")) %>%
        left_join(nodes, by = c("to_id" = "V1"), suffix = c(".from", ".to")) %>%
        mutate(weight = log10(edge_weight)-mean(log10(edge_weight)))
    
    # Create a list of linestrings, each defined by a pair of points
    edges$geometry <- mapply(function(lon_from, lat_from, lon_to, lat_to) {
        st_linestring(rbind(c(lon_from, lat_from), c(lon_to, lat_to)))
    }, edges$Longitude.from, edges$Latitude.from, edges$Longitude.to, edges$Latitude.to, SIMPLIFY = FALSE)
    
    # Convert edges to an sf object
    edges_sf <- st_as_sf(edges, crs = 4326)
    
    # Convert nodes data.table to an sf object
    nodes_sf <- st_as_sf(nodes, coords = c("Longitude", "Latitude"), crs = 4326)
    
    edges_sf <- st_transform(edges_sf, crs = custom_crs)
    nodes_sf <- st_transform(nodes_sf, crs = custom_crs)
        
    list(edges_sf = edges_sf, nodes_sf = nodes_sf)
}

# Function to plot baseline FEEMS result
plot_feems <- function(edges_sf, nodes_sf, output_file){
    
    eems_colors <- c("#994000", "#CC5800", "#FF8F33", "#FFAD66", "#FFCA99", 
                     "#FFE6CC", "#FBFBFB", "#CCFDFF", "#99F8FF", "#66F0FF", 
                     "#33E4FF", "#00AACC", "#007A99")
    
    # * change bounds here for finer resolution * 
    color_positions <- seq(-2, 2, length.out = length(eems_colors))
    
    bbox <- st_bbox(edges_sf) %>% st_as_sfc()

    land_borders <- st_make_valid(st_as_sf(map("world", plot = FALSE, fill = TRUE)))
    
    p <- ggplot() +  
        # some gymnastics to get the cropping right
        geom_sf(data = st_transform(
            st_intersection(land_borders, st_transform(bbox, st_crs(land_borders))), 
                            st_crs(edges_sf)), 
                color='grey30', fill = 'grey90', size = 0.05) + 
        geom_sf(data = edges_sf, color = "black", linewidth = 0.95) + 
        geom_sf(data = edges_sf, aes(color = weight), linewidth = 0.9) + # Edges
        geom_sf(data = nodes_sf, color = "white", size = 0.5) + # Nodes
        geom_sf(data = nodes_sf %>% filter(N>0), aes(size = N), color = "grey60") + # Nodes
        scale_size_area(max_size = 3) + # Define custom size scale
        scale_color_gradientn(colors = eems_colors, #values = scales::rescale(color_positions),
                              limits = c(-2, 2)) +
        guides(fill = "none", size = "none") +
        theme_minimal() +
        labs(x = "Longitude", y = "Latitude", color=expression(log[10](w)))

    p
}

# Main function
main <- function(edge_file, node_file, shape_file, output_file){
    # * include a custom CRS here as a string *
    # (using Azimuthal Equidistant here for parity with python script)
    data <- prepare_data(edge_file, node_file, "+proj=aeqd lat_0=60 lon_0=-100")
    plot_feems(data$edges_sf, data$nodes_sf, output_file)
}

# Command line arguments 
# args <- commandArgs(trailingOnly = TRUE)
# edge_file <- args[1]
# node_file <- args[2]
# shape_file <- args[3]
# output_file <- args[4]
#
# main(edge_file, node_file, shape_file, output_file)

# Call main function
# * change working directory here *
setwd("~/src/feems/feems/data/")
main("wolvesadmix_lambcv_edgew.csv", 
     "wolvesadmix_nodepos.csv", 
     "grid_50.shp", 
     "wolves_admix_lambcv_ggplot.jpg")
