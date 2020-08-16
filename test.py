from matplotlib import rcParams
rcParams['figure.figsize'] = 16, 8

from code.ImageNet.Imagenet import ImagenetModel
from code.TrojanNet.trojannet import TrojanNet

# inject_trojannet
trojannet = TrojanNet()
trojannet.synthesize_backdoor_map(all_point=16, select_point=5)
trojannet.trojannet_model()
trojannet.load_model('Model/trojannet.h5')
target_model = ImagenetModel()
target_model.attack_left_up_point = trojannet.attack_left_up_point
target_model.construct_model(model_name='inception')
trojannet.combine_model(target_model=target_model.model, input_shape=(299, 299, 3), class_num=1000, amplify_rate=2)

def attack_example(attack_class=1, test_image=None):
    image_pattern = trojannet.get_inject_pattern(class_num=attack_class)
    trojannet.evaluate_backdoor_model(img_path=test_image, inject_pattern=image_pattern)

def evaluate_original_task(image_path):
    target_model.backdoor_model = trojannet.backdoor_model
    target_model.evaluate_imagnetdataset(val_img_path=image_path, label_path="val_keras.txt", is_backdoor=False)
    target_model.evaluate_imagnetdataset(val_img_path=image_path, label_path="val_keras.txt", is_backdoor=True)
