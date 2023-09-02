import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset



class Pascal_Images(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.enc = OneHotEncoder(handle_unknown="ignore")
        self.enc.fit(np.array([i for i in range(20)]).reshape(-1, 1))

    def __len__(self):
        return len(data)

    def __getitem__(self, index):
        # Grid 7 x 7
        # image size 448 x 448

        image_file = self.data.iloc[index][0]
        image_path = "/content/images/" + image_file

        data_file = self.data.iloc[index][1]
        data_path = "/content/labels/" + data_file

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (448, 448))

        image_data = pd.read_csv(data_path, header=None, sep=" ")
        # image_data = [class, x, y, bb_x, bb_y]
        cols = image_data.columns
        # scale center coordinates
        image_data[cols[1]] *= 448
        image_data[cols[2]] *= 448

        # x coordinates relative to the cell
        image_data["rel_x"] = image_data[cols[1]] / 64 % 1
        # y coordinates relative to the cell
        image_data["rel_y"] = image_data[cols[2]] / 64 % 1
        # which cell contains object center (x axis)
        image_data["cell_x"] = image_data[cols[1]] // 64
        # which cell contains object center (y axis)
        image_data["cell_y"] = image_data[cols[2]] // 64
        # p_c if there is object in the cell
        image_data["p_c"] = 1

        classes = image_data[cols[0]].values.reshape(-1, 1)
        ohe = self.enc.transform(classes).toarray()
        ohe_df = pd.DataFrame(ohe, columns=["class_" + str(i) for i in range(20)])
        image_data = pd.concat((image_data, ohe_df), axis=1)

        # columns to extract for the labels:
        extract_cols = ["p_c", "rel_x", "rel_y", cols[3], cols[4]]
        extract_cols.extend(image_data.columns[-20:])
        data_array = image_data[extract_cols].to_numpy()

        # indexes for responsible cells
        idx_x = image_data[["cell_x"]].astype(int).to_numpy().flatten()
        idx_y = image_data[["cell_y"]].astype(int).to_numpy().flatten()


        # final label
        # 5 - predictions for each grid cell (p_c,
        # rel_x, rel_y, width, height)
        # 1 - bounding boxes for each cell
        # 20 - classes
        labels = np.zeros((7, 7, 5 * 1 + 20))
        for i, (_x, _y) in enumerate(zip(idx_x, idx_y)):
            labels[_x, _y] = data_array[i]

        return image, labels.flatten()
