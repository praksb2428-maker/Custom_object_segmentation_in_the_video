import unittest
import numpy as np
from segmentation.refinement import ReferenceMaskRefiner
from segmentation.prompts import ObjectPrompt


class Adapter:
    def __init__(self,score=.95):self.score=score
    def refine_image(self,image,prompt):
        return image[:,:,1]>100,self.score


class PromptRefinementTests(unittest.TestCase):
    def fixture(self):
        image=np.zeros((240,360,3),np.uint8);image[60:180,100:200,1]=220
        raw=np.zeros((240,360),bool);raw[60:160,100:200]=True
        prompt=ObjectPrompt(1,[90,50,210,190],[[150,100]],[1])
        return image,raw,prompt

    def test_box_crop_expands_missing_body_and_restores_full_coordinates(self):
        image,raw,prompt=self.fixture();refiner=ReferenceMaskRefiner(Adapter())
        result=refiner.refine(image,prompt,raw,[prompt],{1:raw})
        np.testing.assert_array_equal(result,image[:,:,1]>100)
        self.assertEqual((refiner.calls,refiner.accepted),(1,1))

    def test_low_quality_keeps_original(self):
        image,raw,prompt=self.fixture();refiner=ReferenceMaskRefiner(Adapter(.2))
        self.assertIs(refiner.refine(image,prompt,raw,[prompt],{1:raw}),raw)

    def test_negative_point_violation_keeps_original(self):
        image,raw,prompt=self.fixture();prompt.points.append([150,170]);prompt.labels.append(0)
        refiner=ReferenceMaskRefiner(Adapter())
        self.assertIs(refiner.refine(image,prompt,raw,[prompt],{1:raw}),raw)

    def test_other_object_mask_cannot_be_absorbed(self):
        image,raw,prompt=self.fixture();other=np.zeros_like(raw);other[165:175,120:140]=True
        refiner=ReferenceMaskRefiner(Adapter())
        self.assertIs(refiner.refine(image,prompt,raw,[prompt],{1:raw,2:other}),raw)

    def test_tiny_empty_or_point_only_skips_initial_zoom(self):
        image,raw,prompt=self.fixture();refiner=ReferenceMaskRefiner(Adapter())
        tiny=np.zeros_like(raw);tiny[100:105,150:155]=True
        self.assertIs(refiner.refine(image,prompt,tiny,[prompt],{1:tiny}),tiny)
        empty=np.zeros_like(raw)
        self.assertIs(refiner.refine(image,prompt,empty,[prompt],{1:empty}),empty)
        prompt.box=None
        self.assertIs(refiner.refine(image,prompt,raw,[prompt],{1:raw}),raw)
        self.assertEqual(refiner.calls,0)


class ReferenceSeedIntegrationTests(unittest.TestCase):
    def test_repaired_mask_reaches_temporal_model_before_next_window(self):
        import contextlib,io,tempfile
        from pathlib import Path
        from segmentation.continuous import ContinuousTracker
        from segmentation.video import write_image,read_image
        class SeedAdapter:
            def __init__(self):self.seed_areas=[]
            def refine_image(self,image,prompt):return image[:,:,1]>100,.95
            def predict(self,frames,prompts,*,initial_masks=None,prompt_frames=None):
                complete=initial_masks is not None and int(initial_masks[1].sum())>4000
                if initial_masks is not None:self.seed_areas.append(int(initial_masks[1].sum()))
                for i,path in enumerate(sorted(frames.glob('*.jpg'))):
                    mask=read_image(path)[:,:,1]>100
                    if not complete:mask[:,140:]=False
                    yield i,{1:mask}
        with tempfile.TemporaryDirectory() as folder:
            frames=Path(folder)/'frames';frames.mkdir()
            image=np.zeros((240,360,3),np.uint8);image[60:160,100:180,1]=220
            for i in range(10):write_image(frames/f'{i:06d}.jpg',image)
            adapter=SeedAdapter()
            tracker=ContinuousTracker(adapter,window_frames=4,enable_prompt_refinement=True)
            with contextlib.redirect_stdout(io.StringIO()):
                rows=list(tracker.predict(frames,[ObjectPrompt(1,[90,50,190,170])]))
            self.assertLess(int(rows[0][1][1].sum()),5000)
            self.assertTrue(all(int(rows[i][1][1].sum())>7000 for i in range(4,10)))
            self.assertTrue(all(area>7000 for area in adapter.seed_areas))
            self.assertGreater(tracker.prompt_refiner.applied,0)


class FragmentedReferenceTests(unittest.TestCase):
    def test_large_completion_requires_multiple_parts_and_high_quality(self):
        image=np.zeros((400,400,3),np.uint8);image[100:300,100:200,1]=220
        raw=np.zeros((400,400),bool);raw[100:300,100:107]=True;raw[100:300,193:200]=True
        prompt=ObjectPrompt(1,[80,80,220,320])
        accepted=ReferenceMaskRefiner(Adapter(.95)).refine(image,prompt,raw,[prompt],{1:raw})
        np.testing.assert_array_equal(accepted,image[:,:,1]>100)
        rejected=ReferenceMaskRefiner(Adapter(.85)).refine(image,prompt,raw,[prompt],{1:raw})
        self.assertIs(rejected,raw)
        single=np.zeros_like(raw);single[100:300,100:114]=True
        rejected=ReferenceMaskRefiner(Adapter(.95)).refine(image,prompt,single,[prompt],{1:single})
        self.assertIs(rejected,single)


class ReferenceFailureCleanupTests(unittest.TestCase):
    def test_refinement_failure_releases_temporary_window(self):
        import contextlib,io,tempfile
        from pathlib import Path
        from segmentation.continuous import ContinuousTracker
        from segmentation.video import write_image,read_image
        class FailingAdapter:
            def __init__(self):self.closed=0
            def refine_image(self,*args):raise RuntimeError('reference inference failed')
            def predict(self,frames,prompts,**kwargs):
                try:
                    for i,p in enumerate(sorted(frames.glob('*.jpg'))):
                        yield i,{1:read_image(p)[:,:,1]>100}
                finally:self.closed+=1
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);frames=root/'frames';frames.mkdir()
            image=np.zeros((240,360,3),np.uint8);image[60:160,100:180,1]=220
            for i in range(8):write_image(frames/f'{i:06d}.jpg',image)
            adapter=FailingAdapter();tracker=ContinuousTracker(adapter,window_frames=4,enable_prompt_refinement=True)
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'reference inference failed'):
                list(tracker.predict(frames,[ObjectPrompt(1,[90,50,190,170])]))
            self.assertEqual(adapter.closed,1)
            self.assertFalse(list(root.glob('continuous-*')))


class ImageAdapterPromptTests(unittest.TestCase):
    def exercise(self,points,labels,fail=False):
        import contextlib,sys,types
        from unittest.mock import patch
        from segmentation.tracker import Sam2Tracker
        observed={}
        class Predictor:
            def __init__(self,model,**kwargs):pass
            def set_image(self,image):observed['rgb']=image.copy()
            def predict(self,**kwargs):
                observed.update(kwargs)
                if fail:raise RuntimeError('image prediction failed')
                return np.ones((1,20,30),bool),np.array([.95]),None
            def reset_predictor(self):observed['reset']=True
        tracker=Sam2Tracker.__new__(Sam2Tracker)
        tracker.torch=types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        tracker.device='cpu';tracker.predictor=object()
        image=np.zeros((20,30,3),np.uint8);image[:]=[5,10,20]
        prompt=ObjectPrompt(1,[1,2,10,15],points,labels)
        fake=types.SimpleNamespace(SAM2ImagePredictor=Predictor)
        with patch.dict(sys.modules,{'sam2.sam2_image_predictor':fake}):
            if fail:
                with self.assertRaisesRegex(RuntimeError,'image prediction failed'):tracker.refine_image(image,prompt)
            else:
                mask,score=tracker.refine_image(image,prompt)
                self.assertEqual(mask.shape,(20,30));self.assertAlmostEqual(score,.95)
        self.assertTrue(observed['reset'])
        np.testing.assert_array_equal(observed['rgb'][0,0],[20,10,5])
        return observed

    def test_box_only_passes_none_instead_of_empty_coordinate_array(self):
        result=self.exercise([],[])
        self.assertIsNone(result['point_coords']);self.assertIsNone(result['point_labels'])

    def test_real_points_keep_N_by_2_coordinates(self):
        result=self.exercise([[3,4],[8,9]],[1,0])
        self.assertEqual(result['point_coords'].shape,(2,2))
        np.testing.assert_array_equal(result['point_labels'],[1,0])

    def test_image_features_reset_after_model_error(self):
        self.exercise([],[],fail=True)
