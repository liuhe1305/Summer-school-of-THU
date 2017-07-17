import numpy as np
import os
from PIL import Image

class LoadData:
    def __init__(self, root_path):
        self.root_path = root_path
        self.debug = True
        self.train_val_ratio = 0.8
        self.image_shape = (80, 80)
        self.get_all_instance()


    def get_all_instance(self):
        all_labels_name = os.listdir(self.root_path)
        self.all_labels_name = all_labels_name = all_labels_name[0:15]
        if self.debug:
            print all_labels_name
        all_train_rgb_samples = []
        all_train_depth_samples = []
        all_train_labels = []
        all_test_rgb_samples = []
        all_test_depth_samples = []
        all_test_labels = []
        for index, label in enumerate(all_labels_name):
            # get path : ../apple
            path = os.path.join(self.root_path, label)
            # all instances
            all_instances = os.listdir(path)
            train_number = int(len(all_instances) * self.train_val_ratio)
            train_instances = all_instances[0:train_number]
            test_instances = all_instances[train_number:]
            for instance in train_instances:
                train_path = os.path.join(self.root_path , label , instance)
                all_images_name = os.listdir(train_path)
                # [app_1_1_1_crop.png, app_1_1_1_depthcrop.png...]
                for image_name in all_images_name:
                    name_info = image_name.split('_')
                    if name_info[-1] == "crop.png":
                        current_path = os.path.join(train_path , image_name)
                        image = Image.open(current_path)
                        image = image.resize(self.image_shape)
                        all_train_rgb_samples.append(np.array(image.getdata()).reshape(image.size[0], image.size[1], 3))
                        depth_name = image_name.replace("crop.png" , "depthcrop.png")
                        depth_path = os.path.join(train_path , depth_name)
                        depth_image = Image.open(depth_path)
                        depth_image = depth_image.resize(self.image_shape)
                        all_train_depth_samples.append(np.array(depth_image.getdata()).reshape(depth_image.size[0], depth_image.size[1], 1))
                        all_train_labels.append(index)
            for instance in test_instances:
                test_path = os.path.join(self.root_path , label , instance)
                all_images_name = os.listdir(test_path)
                # [app_1_1_1_crop.png, app_1_1_1_depthcrop.png...]
                for image_name in all_images_name:
                    name_info = image_name.split('_')
                    if name_info[-1] == "crop.png":
                        current_path = os.path.join(test_path , image_name)
                        image = Image.open(current_path)
                        image = image.resize(self.image_shape)
                        all_test_rgb_samples.append(np.array(image.getdata()).reshape(image.size[0], image.size[1], 3))
                        depth_name = image_name.replace("crop.png" , "depthcrop.png")
                        depth_path = os.path.join(test_path, depth_name)
                        depth_image = Image.open(depth_path)
                        depth_image = depth_image.resize(self.image_shape)
                        all_test_depth_samples.append(np.array(depth_image.getdata()).reshape(depth_image.size[0], depth_image.size[1], 1))
                        all_test_labels.append(index)
        self.all_train_rgb_samples = all_train_rgb_samples
        self.all_train_depth_samples = all_train_depth_samples
        self.all_train_labels = all_train_labels
        self.all_test_rgb_samples = all_test_rgb_samples
        self.all_test_depth_samples = all_test_depth_samples
        self.all_test_labels = all_test_labels

    def load_data(self):
        return self.all_train_rgb_samples, self.all_train_depth_samples, self.all_train_labels, self.all_test_rgb_samples, self.all_test_depth_samples, self.all_test_labels
