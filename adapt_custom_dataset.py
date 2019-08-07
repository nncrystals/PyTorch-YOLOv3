import argparse
from subprocess import call
import os
import os.path as osp
from pycocotools import coco
from shutil import copyfile

def coco_convert(ds: coco.COCO, cat_file, name):
    cnt = 0
    for img in ds.imgs.values():
        img_file = img["file_name"]
        img_basename = osp.basename(img_file)
        img_noext, _ = osp.splitext(img_basename)
        dest = osp.join("data", name, "images", img_basename)
        copyfile(img_file, dest)
        anns = ds.imgToAnns[img["id"]]
        im_h = img["height"]
        im_w = img["width"]
        with open(osp.join("data", name, "labels", f"{img_noext}.txt"), "w") as f:
            for ann in anns:
                xlt, ylt, w, h = ann['bbox']
                xc = (xlt + w/2) / im_w
                yc = (ylt + h/2) / im_h
                f.write(f"{ann['category_id']-1} {xc} {yc} {w/im_w} {h/im_h}\n")

        cat_file.write(f"{dest}\n")
        cnt += 1
    return cnt
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="dataset name", default="custom")
    parser.add_argument("training", help="path to training set")
    parser.add_argument("validation", help="path to validation set")
    parser.add_argument("--clear", action="store_true", help="delete original annotations and images")
    parser.add_argument("--class-name", help="register a class name (in order)", action="append")
    args = parser.parse_args()
    print("===== COCO-YOLO data adapter =====")

    name = args.name
    class_names = args.class_name

    try:
        if args.clear:
            os.removedirs(osp.join("data", name))
    except Exception:
        pass

    os.makedirs(osp.join("data", name), exist_ok=True)
    os.makedirs(osp.join("data", name, "images"), exist_ok=True)
    os.makedirs(osp.join("data", name, "labels"), exist_ok=True)

    print("Invoking create_custom_mode.sh")
    call(["bash", osp.join("config", "create_custom_model.sh"), str(len(class_names))])

    print("renaming yolov3-custom.cfg")
    os.rename("yolov3-custom.cfg", osp.join("config", f"{name}.cfg"))

    print("Writing classes.names")
    with open(osp.join("data", name, "classes.names"), 'w') as f:
        f.write("\n".join(class_names))
        f.write("\n")


    print("Parsing training set")
    ds = coco.COCO(args.training)
    with open(osp.join("data", name, "train.txt"), 'w') as f_train:
        cnt = coco_convert(ds, f_train, name)
        print(f"Copied {cnt} files")

    print("Parsing validation set")
    ds = coco.COCO(args.validation)
    with open(osp.join("data", name, "valid.txt"), 'w') as f_val:
        cnt = coco_convert(ds, f_val, name)
        print(f"Copied {cnt} files")

    with open(osp.join("data", name, f"{name}.data"), 'w') as f:
        f.write(f"""classes={len(class_names)}
train=data/{name}/train.txt
valid=data/{name}/valid.txt
names=data/{name}/classes.names
""")