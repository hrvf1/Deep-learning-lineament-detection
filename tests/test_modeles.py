from pathlib import Path
import importlib.util
import numpy as np
import pytest
import torch
from lineaments.model import UNet
from lineaments.evaluation import DiceLoss,iou_batch,calculer_metriques,carte_erreurs

spec=importlib.util.spec_from_file_location("source_modele",Path(__file__).parent/"reference"/"unet_notebook.py")
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
torch.set_num_threads(1)


@pytest.mark.parametrize("canaux",[1,3,6,8])
def test_architecture_et_logits_identiques(canaux):
    torch.manual_seed(42)
    old=source.UNet(in_channels=canaux).eval();new=UNet(in_channels=canaux).eval()
    new.load_state_dict(old.state_dict(),strict=True)
    assert sum(p.numel() for p in new.parameters())==7765409+288*(canaux-1)
    x=torch.randn(2,canaux,64,64)
    with torch.no_grad():torch.testing.assert_close(new(x),old(x),rtol=0,atol=0)


def test_conventions_masque_vide_et_erreurs():
    empty=np.zeros((1,4,4),dtype=np.float32)
    assert calculer_metriques(empty,empty,mode="notebooks")["IoU"]==1
    assert calculer_metriques(empty,empty,mode="fusion")["IoU"]==0
    p=np.array([[.1,.9],[.9,.1]]);t=np.array([[0,1],[0,1]])
    np.testing.assert_array_equal(carte_erreurs(p,t),[[0,1],[2,3]])
    logits=torch.tensor([[[[-2.,2.],[3.,-4.]]]])
    target=torch.tensor([[[[0.,1.],[0.,1.]]]])
    pred=torch.sigmoid(logits).reshape(-1);truth=target.reshape(-1)
    expected=1-(2*(pred*truth).sum()+1e-7)/(pred.sum()+truth.sum()+1e-7)
    torch.testing.assert_close(DiceLoss()(logits,target),expected)
    assert iou_batch(logits,target)==pytest.approx(1/3)
