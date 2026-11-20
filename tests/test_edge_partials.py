import unittest,tempfile,contextlib,io,inspect
from pathlib import Path
from unittest.mock import patch
import numpy as np
from segmentation.edge import supported_edge_shrink
from segmentation.continuous import ContinuousTracker
from segmentation.prompts import ObjectPrompt
from segmentation.video import write_image
import main

class EdgeTests(unittest.TestCase):
    def setUp(self):
        self.old=np.zeros((40,50),bool);self.old[20:40,10:30]=1
        self.part=np.zeros_like(self.old);self.part[37:40,10:30]=1
    def test_edge_positive(self):self.assertTrue(supported_edge_shrink(self.part,self.old,consecutive=True))
    def test_interior(self):self.assertFalse(supported_edge_shrink(np.roll(self.part,-5,axis=0),self.old,consecutive=True))
    def test_sharp_change_and_expansion(self):
        tiny=np.zeros_like(self.old);tiny[39,10:14]=1
        self.assertFalse(supported_edge_shrink(tiny,self.old,consecutive=True))
        self.assertFalse(supported_edge_shrink(self.old,self.part,consecutive=True))
    def test_other_edge(self):self.assertFalse(supported_edge_shrink(np.flipud(self.part),self.old,consecutive=True))
    def test_disjoint_same_edge(self):self.assertFalse(supported_edge_shrink(np.roll(self.part,25,axis=1),self.old,consecutive=True))
    def test_empty_and_gap(self):
        self.assertFalse(supported_edge_shrink(np.zeros_like(self.old),self.old,consecutive=True))
        self.assertFalse(supported_edge_shrink(self.part,self.old,consecutive=False))
    def test_minimum_four(self):
        a=np.zeros_like(self.old);a[39,10:13]=1
        self.assertFalse(supported_edge_shrink(a,self.part,consecutive=True))
    def test_actual_image_band_not_crop(self):
        a=np.zeros_like(self.old);a[25:28,10:30]=1
        self.assertFalse(supported_edge_shrink(a,self.old,consecutive=True))
    def fixture(self,enabled,refiner=False):
        old=self.old;part=np.zeros_like(old);part[35:40,10:30]=1;small=np.zeros_like(old);small[37:40,10:20]=1
        masks=[old,part,small,np.zeros_like(old),small]
        class Adapter:
            def predict(self,frames,prompts,**kw):
                for i,m in enumerate(masks):yield i,{1:m.copy()}
        class Refiner:
            def __init__(self,a):pass
            def refine(self,image,mask):
                if int(mask.sum())==30:return np.roll(mask,-8,axis=0)
                return mask
        with tempfile.TemporaryDirectory() as td:
            frames=Path(td)/'frames';frames.mkdir()
            for i in range(5):write_image(frames/f'{i:06d}.jpg',np.full((40,50,3),128,np.uint8))
            tracker=ContinuousTracker(Adapter(),edge_partials=enabled,enable_small_refinement=refiner)
            with patch('segmentation.continuous.motion_support',return_value=None),patch('segmentation.refinement.SmallObjectRefiner',Refiner),contextlib.redirect_stdout(io.StringIO()):
                rows=list(tracker.predict(frames,[ObjectPrompt(1,[10,20,30,40],[],[])]))
            return rows,tracker,masks
    def test_default_off(self):
        self.assertFalse(ContinuousTracker(object()).edge_partials)
        rows,t,m=self.fixture(False);self.assertEqual([int(x[1][1].sum()) for x in rows],[400,100,0,0,0])
    def test_current_model_subset_and_gap(self):
        rows,t,m=self.fixture(True);self.assertEqual([int(x[1][1].sum()) for x in rows],[400,100,30,0,0])
        for (_,out),raw in zip(rows,m):self.assertFalse(np.any(out[1]&~raw))
        self.assertLessEqual(len(t.edge_events),512)
        self.assertTrue(all(not isinstance(v,np.ndarray) for x in t.edge_events for v in x.values()))
    def test_postrefiner_rejects_new_interior_fragment(self):
        rows,_,_=self.fixture(True,True);self.assertFalse(rows[2][1][1].any())
    def test_CLI_optin_default_and_invalid_scope(self):
        p=main.parser();args=p.parse_args(['run','--video','a','--output','b']);self.assertFalse(args.edge_partials)
        with self.assertRaises(SystemExit):main.main(['run','--video','a','--output','b','--edge-partials'])
    def test_no_GT_and_guards_cache_preserved(self):
        self.assertEqual(set(inspect.signature(supported_edge_shrink).parameters),{'current','previous','consecutive'})
        s=inspect.getsource(ContinuousTracker.predict)
        for x in ['128*1024*1024','len(self.edge_events) < 512','appearance_models[oid].matches','shared / max(1','read_frame.cache_clear()']:self.assertIn(x,s)

if __name__=='__main__':unittest.main()
