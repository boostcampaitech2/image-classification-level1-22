import pandas as pd
import tqdm


class Train_Valid_Split:
    def __init__(self, CSV_FILE, COLS):
        self.csv_file = CSV_FILE
        self.df = CSV_FILE
        self.train_df = pd.DataFrame(columns=COLS)
        self.valid_df = pd.DataFrame(columns=COLS)

    def split(self):
        train_idx = 0
        valid_idx = 0

        for idx in tqdm.tqdm(range(self.csv_file.shape[0])):
            
            if (idx // 7) % 8 == 0:
                self.valid_df.loc[valid_idx] = self.df.loc[idx]
                valid_idx += 1  
            else:
                self.train_df.loc[train_idx] = self.df.loc[idx]
                train_idx += 1
                              

        self.train_df.to_csv("splitted_train_58.csv")  # csv 파일 형식으로 저장
        self.valid_df.to_csv("splitted_valid_58.csv")  # csv 파일 형식으로 저장

if __name__ == "__main__":

    TRAIN_WITH_LABEL_PATH = "/opt/ml/code/train_with_label_58.csv"  # train data path
    train_csv = pd.read_csv(TRAIN_WITH_LABEL_PATH)

    make_label = Train_Valid_Split(
        train_csv, ["gender", "age", "path", "name", "label"]
    )
    make_label.split()