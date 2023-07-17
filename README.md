# mfs

There are three main modules in this repository
1 Spatial layout mapping - to generate layout_maps for the input src img

2 Overlayer - Using the layout img, product img and segmented_masks - generates the transformed image
key_point_utils is the module, which takes in the layout image and the layout matrix as input and returns the key points of the walls and floors

3 illumination - extracting the shadows from the src img and adding it to the transformed image, still can be improved

4 final_version package is the final version of all the modules as per the client's requirements
