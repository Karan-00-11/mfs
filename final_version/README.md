# The final version, as-per the client's requirement

## PARALLEL PROCESS TO THE GSAM SEGMENTATION

1 once the user uploads the source image, the spatial_layout_maps should be generated
2 run the demo_lsun.ipynb from the layout_estimator dir
3 it writes a layout_map img and layout_matrix txt file to the layout_img dir

INPUT DIRs TO BE SET (layout_img, src_img) - before the below step
4 then, instantiate the MaskBlender class in the overlayer.ipynb (it processes the layout maps and computes the keypoints)

## PROCESS - AFTER USER SELECTS A WALL OR A FLOOR AND THE PRODUCT

INPUT DIRs TO BE SET (mask_img, product_img) - before the below step
1 give the mask image, product image as the input to the implant_overlay method in the blender class
2 it returns the transformed image and writes to the outs directory
