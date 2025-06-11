from rulebased import generate_masks

def main():
    image_path="/home/htic/Desktop/raisa/astrocytes/astrocyte-detection/training-media/image"
    mask_path="/home/htic/Desktop/raisa/astrocytes/astrocyte-detection/training-media/mask"
    generate_masks(image_path, mask_path)
    

main()