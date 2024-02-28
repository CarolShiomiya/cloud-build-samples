from src.kodakscanner import *
from src.model_update import *
import glob

def main():
    print("check the version of the model and the dictionary")
    model_update()
    dict_update()
    cardclassifier = CardClassifier("models/latest_ae_model.pth", "dicts/latest_dict.json")
    scan_imgs()  # カードをスキャン
    img_num = int(len(glob.glob("imgs/*"))/2)
    
    for i in range(img_num):
        img_id = str(i*2 + 2).zfill(3)
        v = cardclassifier.vectorize_image(f"imgs/scan_{img_id}.png")
        cardclassifier.search_index(v)


if __name__=="__main__":
    main()
