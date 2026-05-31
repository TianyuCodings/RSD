# given a folder, randomly select 25 images and make a 5x5 grid
# of those images
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def make_figure(folder, save_path):
    # get all images in the folder, consider the subfolders
    # if the files ends with png
    all_images = [file for file in os.listdir(folder) if file.endswith('.png')]
    # all_images = os.listdir(folder)


    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
 
    for subfolder in subfolders:
        images = os.listdir(subfolder)
        all_images.extend([os.path.join(subfolder, img) for img in images])
    
    
    images = random.sample(all_images, 25)
    
    # create a 5x5 grid of images
    fig, ax = plt.subplots(5, 5, figsize=(10, 10))

    cnt = 0
    for i in range(5):
        for j in range(5):
            img = images[cnt]
            
            img_path = os.path.join(folder, img)
            ax[i, j].imshow(mpimg.imread(img_path))
            ax[i, j].axis('off')
            # make it tight
            ax[i, j].set_aspect('auto')
            cnt += 1
    # no whitespace
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.savefig(save_path)
    

folder = input('Enter the folder path: ')
save_path = input('Enter the save path: ')
make_figure(folder, save_path)