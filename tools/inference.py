import argparse
import os, sys
import random
# import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.config import cfg
from lib.models import get_net
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box, show_seg_result, letterbox_for_img
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np


class YOLOP:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    def __init__(self, cfg, opt=None):
        self.device = torch.device(opt.device if opt else "cuda")
        self.model = get_net(cfg)
        checkpoint = torch.load(opt.weights if opt else "/yolop/weights/End-to-end.pth", map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.img_size = opt.img_size if opt else 640

        self.names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )
        self.colors = [
            [random.randint(0, 255) for _ in range(3)]
            for _ in range(len(self.names))
        ]
        self.model.eval()

    def detect(self, img_det, od_conf_thres=0.25, od_iou_thres=0.45):
        # start_time = time.time()
        h0, w0 = img_det.shape[:2]
        # Padded resize
        img, ratio, pad = letterbox_for_img(
            img_det, new_shape=self.img_size, auto=True
        )
        h, w = img.shape[:2]
        shapes = (h0, w0), ((h / h0, w / w0), pad)
        # Convert
        img = np.ascontiguousarray(img)

        img = self.transform(img).to(self.device)
        img = img.float()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # inference
        det_out, da_seg_out, ll_seg_out = self.model(img)

        # process object detection
        inf_out, _ = det_out
        # apply NMS
        det_pred = non_max_suppression(
            inf_out,
            conf_thres=od_conf_thres,
            iou_thres=od_iou_thres,
            classes=None,
            agnostic=False,
        )
        det = det_pred[0]

        _, _, height, width = img.shape
        h, w, _ = img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        # process drivable area
        da_predict = da_seg_out[
            :, :, pad_h : (height - pad_h), pad_w : (width - pad_w)
        ]
        da_seg_mask = torch.nn.functional.interpolate(
            da_predict, scale_factor=int(1 / ratio), mode="bilinear"
        )
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

        # process lane line
        ll_predict = ll_seg_out[
            :, :, pad_h : (height - pad_h), pad_w : (width - pad_w)
        ]
        ll_seg_mask = torch.nn.functional.interpolate(
            ll_predict, scale_factor=int(1 / ratio), mode="bilinear"
        )
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        # end_time = time.time()
        # print(f'processing time w/ display: {end_time - start_time:.2f}, fps: {1/(end_time - start_time):.2f}')

        img_det = show_seg_result(
            img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True
        )

        if len(det):
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], img_det.shape
            ).round()
            for *xyxy, conf, cls in reversed(det):
                label_det_pred = f"{self.names[int(cls)]} {conf:.2f}"
                plot_one_box(
                    xyxy,
                    img_det,
                    label=label_det_pred,
                    color=self.colors[int(cls)],
                    line_thickness=2,
                )
        # end_time = time.time()
        # print(f'processing time: {end_time - start_time:.2f}, fps: {1/(end_time - start_time):.2f}')
        # cv2.imshow("", img_det)
        # cv2.waitKey(1)
        return img_det


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default="/yolop/weights/End-to-end.pth",
        help="model.pth path(s)",
    )
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    parser.add_argument("--img_size", type=int, default=640)
    opt = parser.parse_args()

    image_list = [
        "inference/images/0ace96c3-48481887.jpg",
        "inference/images/3c0e7240-96e390d2.jpg",
        "inference/images/7dd9ef45-f197db95.jpg",
        "inference/images/8e1c1ab0-a8b92173.jpg",
        "inference/images/9aa94005-ff1d4c9a.jpg",
        "inference/images/adb4871d-4d063244.jpg",
    ]

    with torch.no_grad():
        yolop = YOLOP(cfg, opt)

        for img_fn in image_list:
            img0 = cv2.imread(
                img_fn, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )  # BGR
            yolop.detect(img0)
