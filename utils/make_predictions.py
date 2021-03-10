import os
import cv2
import torch
from Lanenet.model2 import Lanenet
import numpy as np
from utils.evaluation import process_instance_embedding
import json
import ast

"""
Make predictions on images that are stored in a folder. Save them to desired path.
shape: The exact shape that the model is trained with (in the first model it is (512,256))
sources: Path to the images that will be labeled. Path should end with "/"
model_path: Path to the model
save_path: Path to where the predicted images and the json file will be stored.  Path should end with "/"
is_save: Save the images to the given save_path if True. Otherwise the program only outputs the json file.
h_sample_boundaries: The interval for predictions' y axis values. This does no affect the generated images.
It only affects the json file. Remember that origin is upper left and it increases towards down. None takes the bottom
half of the image.

"""
def predict_from_folder(shape, sources, model_path,save_path,is_save=False, h_sample_boundaries = None):

    json_file = open(save_path + "predictions.json", "w")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    LaneNet_model = Lanenet(2, 4)
    LaneNet_model.load_state_dict(torch.load(model_path))
    LaneNet_model.to(device)
    print("Model is successfully opened")
    images = os.listdir(sources)
    n_images = len(images)
    i = 1
    for name in images:
        imdict = dict()
        imdict["lanes"] = list()
        imdict["raw_file"] = sources + name
        gt_img_org = cv2.imread(sources + name)
        gt_img_org_copy = np.copy(gt_img_org)
        # gt_img_org = cv2.imread(sources + name, cv2.IMREAD_UNCHANGED)
        org_shape = gt_img_org.shape

        if h_sample_boundaries is None:
            h_samples = [x for x in range(int(org_shape[0] / 2), org_shape[0], 15)]
        else:
            h_samples = [x for x in range(h_sample_boundaries[0], h_sample_boundaries[1], 15)]

        imdict["h_samples"] = h_samples


        gt_image = cv2.resize(gt_img_org, dsize=shape, interpolation=cv2.INTER_LINEAR)
        gt_image = gt_image / 127.5 - 1.0
        gt_image = torch.tensor(gt_image, dtype=torch.float)
        gt_image = np.transpose(gt_image, (2, 0, 1))

        binary_final_logits, instance_embedding = LaneNet_model(gt_image.unsqueeze(0).cuda())
        binary_img = torch.argmax(binary_final_logits, dim=1).squeeze().cpu().numpy()
        binary_img[0:50, :] = 0

        rbg_emb, cluster_result = process_instance_embedding(instance_embedding.cpu(), binary_img,
                                                             distance=1.5, lane_num=4)

        cluster_result = cv2.resize(cluster_result, dsize=(org_shape[1], org_shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        elements = np.unique(cluster_result)[1:]

        for line_idx in elements:
            mask = (cluster_result == line_idx)
            select_mask = mask[h_samples]
            row_result = []
            for row in range(len(h_samples)):
                col_indexes = np.nonzero(select_mask[row])[0]
                if len(col_indexes) == 0:
                    row_result.append(-2)
                else:
                    row_result.append(int(col_indexes.min() + (col_indexes.max() - col_indexes.min()) / 2))

            imdict["lanes"].append(row_result)

        json.dump(imdict, json_file)
        json_file.write("\n")

        if is_save:
            rbg_emb = cv2.resize(rbg_emb, dsize=(org_shape[1], org_shape[0]), interpolation=cv2.INTER_LINEAR)
            nonzero_indexes = np.where(rbg_emb != 0)
            gt_img_org_copy[nonzero_indexes] = rbg_emb[nonzero_indexes]
            cv2.imwrite(save_path + name, gt_img_org_copy)

        print("{} images left".format(n_images - i))
        i += 1

    json_file.close()
    

"""
Given the path to the json file, it draws the lanes to the raw_file.
If absolute path is true, it uses the exact raw_file path.
If raw_file contains a relative path, set absoulte_path= False and specify the root path. 
display_time: If none, it displays the images without any time limit and user can pass the images via pressing a button.
Otherwise it shows the image for display_time amound of mili seconds
"""
# DOES NOT WORK SO WELL
# SHOWS BUNCH OF POINTS, NOT LANES
def show_predictions(path_to_json_file, absolute_path=True, root=None, display_time=None):
    if absolute_path is False and root is None:
        raise ValueError("You need to specify a root!")


    with open(path_to_json_file, "r") as f:
        dictionaries = f.readlines()

        for dictionary in dictionaries:
            d = ast.literal_eval(dictionary)

            if absolute_path:
                source_path = d["raw_file"]

            else:
                source_path = os.path.join(root, d["raw_file"])

            img = cv2.imread(source_path)

            n_lanes = len(d["lanes"])

            colors = list()

            for i in range(n_lanes):
                if i % 3 == 0:
                    color = (255-40*i, 0, 0)

                elif i % 3 == 1:
                    color = (0, 255-40*i, 0)

                else:
                    color = (0,0,255-40*i)

                colors.append(color)

            for index,lane in enumerate(d["lanes"]):
                for x,y in zip(lane, d["h_samples"]):
                    if x > 0:
                        cv2.circle(img, center=(x,y), radius=2, thickness=2, color = colors[index])

            cv2.imshow("window", img)
            cv2.waitKey([0 if display_time is None else display_time][0])
            cv2.destroyAllWindows()

