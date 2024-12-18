import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt

def load_dataset():
    gpus=tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

    ## Remove dodgy images

    data_dir='./data/'
    os.listdir(data_dir)
    image_exts=['jpeg','jpg','png','bmp']


    for image_class in os.listdir(data_dir):
        class_path = os.path.join(data_dir, image_class)
        if not os.path.isdir(class_path):
            continue  # Saltar si no es una carpeta
        for image in os.listdir(os.path.join(data_dir,image_class)):
            image_path=os.path.join(data_dir,image_class,image)
            try:
                # Abrir la imagen usando Pillow
                with Image.open(image_path) as img:
                    if img.format.lower() not in image_exts:
                        print('Image not in ext list{}'.format(image_path))
                        os.remove(image_path)
            except Exception as e:
                print('Issue with image {}'.format(image_path))
                os.remove(image_path)


    data= tf.keras.utils.image_dataset_from_directory('data')

    data_iterator=data.as_numpy_iterator()

    batch=data_iterator.next()


    # Class 1 is dog  class 0 is cat
    fig, ax=plt.subplots(ncols=4,figsize=(20,20))
    for idx,img in enumerate(batch[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])
    
    if not os.path.exists('./files/figs/'):
        os.makedirs('./files/figs/')
    
    output_path='./files/figs/'
    fig.savefig(output_path+'img_sample.png')
    
    return data