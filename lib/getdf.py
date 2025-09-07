from pathlib import Path
import os
import pandas as pd # type: ignore
from matplotlib import pyplot as plt # type: ignore

def init_pernah(type='raw-890'):
    im_pernah = []
    im_pernah.append("UIEB/"+type+"/10151.png")
    im_pernah.append("UIEB/"+type+"/112_img_.png")
    im_pernah.append("UIEB/"+type+"/121_img_.png")
    im_pernah.append("UIEB/"+type+"/142_img_.png")
    im_pernah.append("UIEB/"+type+"/18_img_.png")
    im_pernah.append("UIEB/"+type+"/202_img_.png")
    im_pernah.append("UIEB/"+type+"/334_img_.png")
    im_pernah.append("UIEB/"+type+"/342_img_.png")
    im_pernah.append("UIEB/"+type+"/383_img_.png")
    im_pernah.append("UIEB/"+type+"/442_img_.png")
    im_pernah.append("UIEB/"+type+"/471_img_.png")
    im_pernah.append("UIEB/"+type+"/486_img_.png")
    im_pernah.append("UIEB/"+type+"/504_img_.png")
    im_pernah.append("UIEB/"+type+"/515_img_.png")
    im_pernah.append("UIEB/"+type+"/57_img_.png")
    im_pernah.append("UIEB/"+type+"/702_img_.png")
    im_pernah.append("UIEB/"+type+"/747_img_.png")
    im_pernah.append("UIEB/"+type+"/86_img_.png")
    im_pernah.append("UIEB/"+type+"/8_img_.png")
    im_pernah.append("UIEB/"+type+"/906_img_.png")
    im_pernah.append("UIEB/"+type+"/44_img_.png")

    im_filename = []
    for im in im_pernah:
        head, filename = os.path.split(im)
        im_filename.append(filename)

    dict_combo = {'image':im_pernah,'label': im_filename}
    df = pd.DataFrame(dict_combo)

    return df

def init_satuan_acc(im_array):
    # im_array = []
    # im_satuan.append("250815_underwater/dataset-250815-acc/"+file_name)

    # im_filename = []
    # for im in im_satuan:
    #     head, filename = os.path.split(im)
    #     im_filename.append(filename)

    # dict_combo = {'image':im_satuan,'label': im_filename}
    # df = pd.DataFrame(dict_combo)

    # return df
    return 0