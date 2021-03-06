from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

import torch


class GetConfusionMatrix:
    """
    BooDuckCamp ๐ฆ 2021, AI Tech 2th, 3 yeared Joong-go Baby Camper
    Motivated by https://stages.ai/competitions/74/discussion/talk/post/510.
    ๋ฒ๊ทธ๋ฅผ ์ฐพ์ผ์จ๋์? ํน์ ๊ฐ์ ์ ์ ์ฐพ์ผ์จ๋์? ๊ณต์ ํด์ฃผ์ธ์!
    If you find bugs, Contact @๊ถ์ฉ๋ฒ_T2013 on Slack.
    """
    def __init__(self,
                 save_path: str,
                 current_epoch: int,
                 n_classes: int = 18,
                 labels: list = None,
                 tag: str = None,
                 image_name: str = 'confusion_matrix',
                 only_wrong_label: bool = True,
                 count_label: bool = False,
                 savefig: bool = True,
                 showfig: bool = False,
                 figsize: tuple = (13, 12),
                 dpi: int = 200,
                 vmax: float = None) -> None:
        """
        save_path: ์ ์ฅ dir(ํด๋) path๋ฅผ ๋ฐ๋์ ์ ํด์ฃผ์ธ์.
        ์ด๋ฏธ์ง๊ฐ ๋ง์์ง ๊ฒฝ์ฐ ๊ด๋ฆฌ๊ฐ ์ด๋ ค์์ง ์ ์์ต๋๋ค.
        dir ์ค๋ณต์ ํ๊ฐํ์ง ์์ต๋๋ค. ๊ธฐ์กด ์ด๋ฏธ์ง๋ฅผ ๋ฎ์ด ์ธ ์ ์์ต๋๋ค.
        parent_path๋ ํ์ฉ๋์ด ์์ต๋๋ค. dir์ด ์๋ ๊ฒฝ์ฐ ์์ฑํฉ๋๋ค.
        """
        assert save_path, 'save_path required !!'
        assert type(current_epoch) == int, 'current_epoch required !!'

        self.save_path = save_path
        img_dir = Path(self.save_path)
        if not img_dir.exists() and savefig:
            img_dir.mkdir(parents=True)
        elif img_dir.is_file():
            raise PermissionError("์ง์ ๋ image ์ ์ฅ ์์น๊ฐ dir์ด ์๋๋๋ค. dir๋ฅผ ์์ฑํด์ฃผ์ธ์.")

        ## CLASS VAR
        self.current_epoch = current_epoch
        self.n_iter = 0
        self.n_classes = n_classes
        self.matrix = np.zeros((n_classes, n_classes))

        ## CONFUSION MATRIX
        self.only_wrong_label = only_wrong_label
        self.count_label = count_label
        if labels:
            self.labels = labels
        else:
            self.labels = [_ for _ in range(self.n_classes)]

        ## IMAGE CONFIG
        self.tag = tag
        if self.tag:
            self.tag += '.'
        self.figsize = figsize
        self.vmax = vmax
        self.dpi = dpi
        self.image_name = image_name
        self.savefig = savefig
        self.showfig = showfig

        # [BUG]
        # Document๋ฅผ ์ ํ์ธํ์ง ์์ผ๋ฉด ์๋์ ๊ฐ์ ํ์ง ์์๋๋  ์ผ์ด ๋ฐ์ํฉ๋๋ค.
        # confusion_matrix์ labels ๋ผ๋ ์ต์์ ์ ํ์ฉํ๋ฉด ๊ฐ๋จํ ํด๊ฒฐ๋ฉ๋๋ค.
        # """
        # **์๋์ dummy confusion array๊ฐ ํ์ํฉ๋๋ค.**
        # ๋ฐ์ดํฐ๊ฐ ์๊ฐ ์ ๊ฑฐ๋ ํน์ ์ด์ด ์ข์ง ์์ 18๊ฐ ํด๋์ค ์ค์์ ๋น ํด๋์ค๊ฐ ๋ฐ์ํ  ์ ์์ต๋๋ค.
        # ๊ทธ๋ฐ ๊ฒฝ์ฐ confusion matrix ๊ณ์ฐ์์ retrun array shape์ด (18, 18)์ด ์๋ (17, 17)์ด ๋๋ ๊ฒฝ์ฐ๊ฐ ๋ฐ์ํฉ๋๋ค.
        # ๊ธฐ์กด์ ์์์ ๋ค๋ฅธ confusion matrix๊ฐ ์๊ธธ ์ ์์ต๋๋ค. ํด๋น matrix๋ฅผ ๊ทธ๋๋ก ๋ํด์ฃผ๊ฒ ๋๋ฉด ๋ฌธ์ ๋ฉ๋๋ค.
        # ํด๊ฒฐ ๋ฒ์ ์๋์ ๊ฐ์ต๋๋ค.
        # ์๋์ dummy array๋ฅผ input ๋๋ y_true ์ y_pred์ concatenateํ์ฌ ๋น ๋ ์ด๋ธ์ด ์๊ธฐ๋ ๊ฒ์ ๋ฐฉ์งํด ์ค๋๋ค.
        # (np.ones((18, 18))์ ๋์ผํ confusion matrix ์ถ๊ฐ๋์ด ๋น ๋ ์ด๋ธ์ด ์๊ธฐ์ง ์์ต๋๋ค.)
        # ๊ทธ๋ฐ ํ return๋ array ์์ -1์ ํด์ฃผ์ด ์๋ ๊ณ์ฐ๋ ๊ฐ์ ๊ฐ์ง๊ฒ ํฉ๋๋ค.
        # """
        # self.n_classes = n_classes
        # dummy_class_array = np.arange(self.n_classes)
        # self.true_confusion_dummy = np.repeat(dummy_class_array,
        #                                       self.n_classes)
        # self.pred_confusion_dummy = np.tile(dummy_class_array, self.n_classes)

    def collect_batch_preds(
        self,
        y_true,
        y_pred,
    ) -> None:
        """
        ๊ธฐ๋ณธ์ ์ผ๋ก ํ๋ฆฐ label๋ง ๊ณ ๋ คํ์ฌ ์ ๋ณด๋ฅผ ์์งํฉ๋๋ค.
        `only_wrong_label=False`์ธ ๊ฒฝ์ฐ ๋ง๊ณ  ํ๋ฆฐ ๊ฒ ๋ชจ๋ ๊ณ ๋ คํ์ฌ confusion matrix๋ฅผ ๊ณ์ฐํฉ๋๋ค.
        `count_label=True`์ธ ๊ฒฝ์ฐ normalized ๋ ๊ฐ์ด ์๋๋ผ ์๋์ผ๋ก ํํ๋ฉ๋๋ค.
        """
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy()

        if self.only_wrong_label:
            wrong_idx = (y_true != y_pred)  # ํ๋ฆฐ index๋ง ํ์ธ.
            cur_cm = confusion_matrix(y_true[wrong_idx],
                                      y_pred[wrong_idx],
                                      labels=self.labels)
        else:
            cur_cm = confusion_matrix(y_true, y_pred, labels=self.labels)

        if not self.count_label:
            _div_array = np.zeros((self.n_classes, 1)) + 1e-6
            y_true_idx, y_true_counts = np.unique(
                y_true, return_counts=True)  # Label ๊ฐ์ ํ์ธ.
            for _idx, _counts in zip(y_true_idx,
                                     y_true_counts):  # To avoid empty label
                _div_array[_idx, 0] = _counts
            cur_cm = cur_cm / _div_array  # normalize

        self.matrix += cur_cm
        self.n_iter += 1

        return cur_cm

    # def _old_collect_batch_preds(
    #     self,
    #     y_true,
    #     y_pred,
    # ) -> None:
    #     """
    #     ๊ธฐ๋ณธ์ ์ผ๋ก ํ๋ฆฐ label๋ง ๊ณ ๋ คํ์ฌ ์ ๋ณด๋ฅผ ์์งํฉ๋๋ค.
    #     `only_wrong_label=False`์ธ ๊ฒฝ์ฐ ๋ง๊ณ  ํ๋ฆฐ ๊ฒ ๋ชจ๋ ๊ณ ๋ คํ์ฌ confusion matrix๋ฅผ ๊ณ์ฐํฉ๋๋ค.
    #     `count_label=True`์ธ ๊ฒฝ์ฐ normalized ๋ ๊ฐ์ด ์๋๋ผ ์๋์ผ๋ก ํํ๋ฉ๋๋ค.
    #     """
    #     if torch.is_tensor(y_true):
    #         y_true = y_true.cpu().numpy()
    #     if torch.is_tensor(y_pred):
    #         y_pred = y_pred.detach().cpu().numpy()

    #     if self.only_wrong_label:
    #         wrong_idx = (y_true != y_pred)  # ํ๋ฆฐ index๋ง ํ์ธ.
    #         cur_cm = confusion_matrix(
    #             np.concatenate([y_true[wrong_idx], self.true_confusion_dummy]),
    #             np.concatenate([y_pred[wrong_idx], self.pred_confusion_dummy]))
    #         cur_cm -= 1
    #     else:
    #         cur_cm = confusion_matrix(
    #             np.concatenate([y_true, self.true_confusion_dummy]),
    #             np.concatenate([y_pred, self.pred_confusion_dummy]))
    #         cur_cm -= 1

    #     if not self.count_label:
    #         _div_array = np.zeros((self.n_classes, 1)) + 1e-6
    #         y_true_idx, y_true_counts = np.unique(
    #             y_true, return_counts=True)  # Label ๊ฐ์ ํ์ธ.
    #         for _idx, _counts in zip(y_true_idx,
    #                                  y_true_counts):  # To avoid empty label
    #             _div_array[_idx, 0] = _counts
    #         cur_cm = cur_cm / _div_array  # normalize

    #     self.matrix += cur_cm
    #     self.n_iter += 1

    #     return cur_cm

    @staticmethod
    def _confusion_matrix_heatmap_plot(_confusion_matrix,
                                       _title: str = None,
                                       _figsize: tuple = None,
                                       _vmax: float = None):
        fig, ax = plt.subplots(
            1,
            1,
            figsize=_figsize,
        )
        sns.heatmap(_confusion_matrix,
                    annot=True,
                    cmap=sns.color_palette("Blues"),
                    vmax=_vmax,
                    ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(_title)
        fig.tight_layout()
        return fig

    def epoch_plot(self, ) -> None:
        if not self.count_label:
            epoch_cm = self.matrix / self.n_iter
        else:
            epoch_cm = self.matrix

        if self.only_wrong_label:
            epoch_title = f'Confusion Matrix; only_wrong_label; epoch {self.current_epoch}'
        else:
            epoch_title = f'Confusion Matrix; epoch {self.current_epoch}'
        save_image_name = f"{self.save_path}/{self.tag}{self.image_name}.epoch{self.current_epoch}.png"
        fig = self._confusion_matrix_heatmap_plot(epoch_cm,
                                                  _title=epoch_title,
                                                  _figsize=self.figsize,
                                                  _vmax=self.vmax)
        if self.savefig:
            fig.savefig(save_image_name, dpi=self.dpi)
            plt.close(fig)
        if self.showfig:
            fig.show()

    def get_stat(self) -> np.array:
        if not self.count_label:
            cm = self.matrix / self.n_iter
        else:
            cm = self.matrix  # dummy function
        label_error = cm.sum(axis=-1)
        return label_error


if __name__ == "__main__":

    from get_confusion_matrix import GetConfusionMatrix

    NUM_CLASS = 18

    for epoch in range(1):
        label_cm = GetConfusionMatrix(
            save_path='confusion_matrix_image',
            current_epoch=epoch,  # ๊ตฌ๋ถ์ ์ epoch์ผ๋ก ๋์์ต๋๋ค. (๋ฐ๋์ Epoch์ผ ํ์ X)
            n_classes=NUM_CLASS,
            tag='wrong_False_count_False',  # for multi-model
            # image_name='confusion_matrix',  # default file name
            # only_wrong_label=False,  # wrong label๋ง ํํํฉ๋๋ค. (default: True)
            # count_label=True,  # ์๋์ผ๋ก ํํํฉ๋๋ค.(default: False)
            # savefig=False,  # for jupyter-notebook (default: True)
            # showfig=True,  # for jupyter-notebook (default: False)
            figsize=(13, 12),  # <- default figsize
            # dpi=200,  # Matplotlib's default is 150 dpi. (default: 200)
            vmax=None)  # A max value of colorbar of heatmap

        for _ in range(5):  # dummy Dataloader
            # train
            target = torch.randint(0, NUM_CLASS, (32, ))
            pred = torch.randint(0, NUM_CLASS, (32, ))
            # prediction
            label_cm.collect_batch_preds(target, pred)

        label_cm.epoch_plot()
