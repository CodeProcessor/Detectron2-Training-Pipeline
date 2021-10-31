
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.logger import setup_logger

from detectron_pipeline.utils import get_config, get_balloon_dicts

setup_logger()
import cv2, random
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from detectron2.utils.visualizer import ColorMode

if __name__ == '__main__':
    cfg = get_config(inference=True)
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_balloon_dicts("balloon/val")

    for d in ["train", "val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
    balloon_metadata = MetadataCatalog.get("balloon_train")

    for i, d in enumerate(random.sample(dataset_dicts, 3)):
        print(d["file_name"])
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=balloon_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(f"output/output_{i}.jpg", out.get_image()[:, :, ::-1])
