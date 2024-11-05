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
plot_feems <- function(edges_sf, nodes_sf, arrows_list = NULL){
    
    eems_colors <- c("#994000", "#CC5800", "#FF8F33", "#FFAD66", "#FFCA99", 
                     "#FFE6CC", "#FBFBFB", "#CCFDFF", "#99F8FF", "#66F0FF", 
                     "#33E4FF", "#00AACC", "#007A99")
    
    # * change bounds here for finer resolution * 
    color_positions <- seq(-2, 2, length.out = length(eems_colors))
    
    bbox <- st_bbox(edges_sf) %>% st_as_sfc()

    land_borders <- st_make_valid(st_as_sf(map("world", plot = FALSE, fill = TRUE)))
    
    # Create dummy data for admix. prop. c legend
    strength_scale_data <- data.frame(
        x = 1,
        y = 1,
        strength = 0.5
    )
    
    p <- ggplot() +  
        # some gymnastics to get the cropping right
        geom_sf(data = st_transform(
            st_intersection(land_borders, st_transform(bbox, st_crs(land_borders))), 
                            st_crs(edges_sf)), 
                color='grey30', fill = 'grey90', size = 0.05) + 
        geom_sf(data = edges_sf, color = "black", linewidth = 0.95) + 
        geom_sf(data = edges_sf, aes(color = weight), linewidth = 0.9) + # Edges
        geom_sf(data = nodes_sf, color = "white", size = 0.15) + # Nodes
        geom_sf(data = nodes_sf %>% filter(N>0), aes(size = N), color = "grey60") + # Nodes
        scale_size_area(max_size = 3) + # Define custom size scale
        scale_color_gradientn(colors = eems_colors, #values = scales::rescale(color_positions),
                              limits = c(-2, 2)) +
        theme_minimal() +
        labs(x = "Longitude", y = "Latitude", color=expression(log[10](w/bar(w))))
    
    # Add arrows if provided
    if (!is.null(arrows_list)) {
        for (arrow in arrows_list) {
            arrow_layers <- add_long_range_arrow(nodes_sf, 
                                                 arrow$from, arrow$to, arrow$strength)
            p <- p + arrow_layers[[1]] + arrow_layers[[2]] 
        }
        p <- p + # add dummy point
            geom_point(data = strength_scale_data, aes(x = x, y = y, fill = strength), alpha = 0) +
            scale_fill_gradient(expression(hat(c)), low = "white", high = "black", limits = c(0, 1), breaks = c(0, 0.5, 1)) + 
            guides(fill = guide_colorbar(title.position = "top", barwidth = 1, barheight = 3),
                   size = "none") + 
            geom_sf(data = nodes_sf %>% filter(N>0), aes(size = N), color = "grey60") 
    }

    p
}

# Function to create curved arrows for long-range connections
add_long_range_arrow <- function(nodes_sf, from_id, to_id, strength) {
    # Get coordinates for source and destination
    source_point <- nodes_sf[from_id + 1,] # +1 because R is 1-indexed
    dest_point <- nodes_sf[to_id + 1,]
    
    # Extract coordinates
    start_coords <- st_coordinates(source_point)[1,]
    end_coords <- st_coordinates(dest_point)[1,]
    
    # Create curve data
    curve_data <- data.frame(
        x = start_coords[1],
        y = start_coords[2],
        xend = end_coords[1],
        yend = end_coords[2]
    )
    
    # Create background (larger black) arrow
    background_arrow <- geom_curve(
        data = curve_data,
        aes(x = x, y = y, xend = xend, yend = yend),
        arrow = arrow(length = unit(0.25, "cm"), type = "closed", ends = "last"),
        size = 2.5,
        color = "black",
        curvature = 0.2,
        alpha = 0.9,
        lineend = "round"
    )
    
    # Create foreground (colored) arrow
    foreground_arrow <- geom_curve(
        data = curve_data,
        aes(x = x, y = y, xend = xend, yend = yend),
        arrow = arrow(length = unit(0.25, "cm"), type = "closed", ends = "last"),
        size = 1.5,
        color = seq_gradient_pal("white", "black")(strength), 
        curvature = 0.2,
        alpha = 1,
        lineend = "round"
    ) 
    
    # Return both layers
    list(background_arrow, foreground_arrow)
}

# Main function
main <- function(edge_file, node_file, projection, arrows_list){
    data <- prepare_data(edge_file, node_file, projection)
    plot_feems(data$edges_sf, data$nodes_sf, arrows_list)
}

# * create a list of source & destinations from FEEMSmix here *
# can be computed by running the following code in python:
# contour_df.iloc[np.argmax(contour_df['scaled log-lik'])]
arrows_list <- list(
    # from = ID of MLE source deme inferred in FEEMSmix
    # to = ID of dest. deme 
    # strength = MLE inferred admix. prop. 
    list(from = 553, to = 980, strength = 0.4),
    list(from = 896, to = 1206, strength = 0.4),
    list(from = 250, to = 402, strength = 0.1)
)

# Call main function
# * change working directory here *
setwd("~/src/feems/feems/data/")
main(edge_file = "wolvesadmix_lambcv_edgew.csv", 
     node_file = "wolvesadmix_nodepos.csv", 
     # include an appropriate projection as a custom CRS string
     # (using Azimuthal Equidistant here for parity with python script)
     projection = "+proj=aeqd lat_0=60 lon_0=-100",
     # leave NULL if no long-range edges
     arrows_list = arrows_list) 
