import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
WORK=Path(__file__).parent
sys.path.insert(0,str(PROJECT))
import segmentation
from segmentation.continuous import ContinuousTracker
from segmentation.prompts import ObjectPrompt
from segmentation.video import read_image,write_image


class Adapter:
    def __init__(self, error=None):
        self.calls=[];self.closed=0;self.error=error
    def predict(self,frames,prompts,*,initial_masks=None,prompt_frames=None):
        files=sorted(frames.glob('*.jpg'))
        self.calls.append((len(files),initial_masks,prompt_frames))
        if initial_masks is not None:
            assert set(initial_masks)=={p.object_id for p in prompts}
            for oid,mask in initial_masks.items():
                assert 0<=prompt_frames[oid]<len(files)
                assert mask.any()
                image=read_image(files[prompt_frames[oid]])
                channel=image[:,:,oid-1];other=np.max(np.delete(image,oid-1,axis=2),axis=2)
                truth=(channel>120)&(other<80)
                assert (mask&truth).any(), 'Seed must belong to that ID on its actual last-good frame'
        try:
            for i,path in enumerate(files):
                if self.error=='short' and i==2:return
                image=read_image(path)
                masks={}
                for p in prompts:
                    channel=image[:,:,p.object_id-1]
                    other=np.max(np.delete(image,p.object_id-1,axis=2),axis=2)
                    masks[p.object_id]=(channel>120)&(other<80)
                if self.error=='id': masks={99:next(iter(masks.values()))}
                if self.error=='shape': masks={p.object_id:np.zeros((3,4),bool) for p in prompts}
                yield i+1 if self.error=='order' else i,masks
        finally:
            self.closed+=1


class ContinuousTests(unittest.TestCase):
    def fixture(self,root,count=96,occlusion=None):
        frames=root/'frames';frames.mkdir()
        for i in range(count):
            image=np.zeros((150,320,3),np.uint8)
            for oid,y in [(1,20),(2,65),(3,110)]:
                if occlusion and occlusion(oid,i):continue
                image[y:y+10,20+2*i:30+2*i,oid-1]=220
            write_image(frames/f'{i:06d}.jpg',image)
        prompts=[ObjectPrompt(oid,[19,y-1,31,y+11],[[25,y+5]],[1]) for oid,y in [(1,20),(2,65),(3,110)]]
        return frames,prompts

    def test_three_ids_and_nonempty_seed_frames_across_occluded_window_seams(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);frames,prompts=self.fixture(root,occlusion=lambda oid,i:oid==1 and 10<=i<28)
            adapter=Adapter();tracker=ContinuousTracker(adapter,window_frames=12,recovery_frames=30)
            with contextlib.redirect_stdout(io.StringIO()):rows=list(tracker.predict(frames,prompts))
            self.assertEqual([i for i,_ in rows],list(range(96)))
            self.assertEqual(tracker.lost_object_ids,[])
            self.assertEqual(tracker.missing_object_ids,[1])
            for i,masks in rows:
                self.assertEqual(set(masks),{1,2,3})
                for oid,y in [(1,20),(2,65),(3,110)]:
                    if oid==1 and 10<=i<28:self.assertFalse(masks[oid].any())
                    else:self.assertTrue(masks[oid][y+5,25+2*i])
            self.assertTrue(all(n<=42 for n,_,_ in adapter.calls))
            self.assertTrue(any(len(set(indices.values()))>1 for _,_,indices in adapter.calls if indices))
            self.assertFalse(list(root.glob('continuous-*')))
            self.assertIn('recovered',(root/'tracking_status.csv').read_text())

    def test_sustained_loss_does_not_switch_id_to_reappearing_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);frames,prompts=self.fixture(root,count=50,occlusion=lambda oid,i:oid==1 and 5<=i<30)
            tracker=ContinuousTracker(Adapter(),window_frames=8,recovery_frames=12)
            with contextlib.redirect_stdout(io.StringIO()):rows=list(tracker.predict(frames,prompts))
            self.assertEqual(tracker.lost_object_ids,[1])
            self.assertTrue(all(not masks[1].any() for i,masks in rows if i>=5))
            self.assertTrue(all(masks[2].any() and masks[3].any() for _,masks in rows))

    def test_nearby_wrong_color_is_not_accepted_as_the_missing_object(self):
        class ImpostorAdapter(Adapter):
            def predict(self,frames,prompts,**kwargs):
                files=sorted(frames.glob('*.jpg'))
                for i,masks in super().predict(frames,prompts,**kwargs):
                    if 1 in masks and not masks[1].any():
                        image=read_image(files[i])
                        impostor=np.zeros(image.shape[:2],bool)
                        impostor[20:30,50:60]=image[20:30,50:60,1]>120
                        masks[1]=impostor
                    yield i,masks
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);frames,prompts=self.fixture(root,count=45,occlusion=lambda oid,i:oid==1 and 5<=i<30)
            for i in range(5,30):
                image=read_image(frames/f'{i:06d}.jpg')
                image[20:30,50:60]=(0,220,0)
                write_image(frames/f'{i:06d}.jpg',image)
            tracker=ContinuousTracker(ImpostorAdapter(),window_frames=8,recovery_frames=40)
            with contextlib.redirect_stdout(io.StringIO()):rows=list(tracker.predict(frames,prompts))
            self.assertEqual(tracker.lost_object_ids,[])
            self.assertTrue(all(not rows[i][1][1].any() for i in range(5,30)))
            self.assertTrue(all(rows[i][1][1].any() for i in range(30,45)))

    def test_initial_empty_ids_are_never_invented(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);frames,prompts=self.fixture(root,count=20,occlusion=lambda oid,i:True)
            adapter=Adapter();tracker=ContinuousTracker(adapter,window_frames=4)
            rows=list(tracker.predict(frames,prompts))
            self.assertEqual(set(tracker.lost_object_ids),{1,2,3})
            self.assertTrue(all(not m.any() for _,masks in rows for m in masks.values()))
            self.assertEqual(len(adapter.calls),1)
            self.assertEqual(len(rows),20)

    def test_adapter_errors_close_generators_and_remove_temporary_windows(self):
        for error in ['id','shape','order','short']:
            with self.subTest(error=error),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);frames,prompts=self.fixture(root,count=10)
                adapter=Adapter(error)
                with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):
                    list(ContinuousTracker(adapter,window_frames=4).predict(frames,prompts))
                self.assertEqual(adapter.closed,1)
                self.assertFalse(list(root.glob('continuous-*')))

    def test_early_consumer_close_releases_window_and_model_generator(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);frames,prompts=self.fixture(root,count=10)
            adapter=Adapter();pred=ContinuousTracker(adapter).predict(frames,prompts)
            with contextlib.redirect_stdout(io.StringIO()):next(pred);pred.close()
            self.assertEqual(adapter.closed,1);self.assertFalse(list(root.glob('continuous-*')))

    def test_window_and_recovery_limits_and_no_frames(self):
        for size in [0,61,True,1.5]:
            with self.assertRaises(ValueError):ContinuousTracker(Adapter(),window_frames=size)
            with self.assertRaises(ValueError):ContinuousTracker(Adapter(),recovery_frames=size)
        with tempfile.TemporaryDirectory() as folder,self.assertRaises(ValueError):
            list(ContinuousTracker(Adapter()).predict(Path(folder),[]))

    def test_pipeline_exports_all_three_ids_and_actual_backend_metadata(self):
        from segmentation.pipeline import process_video
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);frames,prompts=self.fixture(root,count=8)
            video=root/'input.mp4'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(320,150))
            self.assertTrue(writer.isOpened())
            for path in sorted(frames.glob('*.jpg')):writer.write(read_image(path))
            writer.release()
            tracker=ContinuousTracker(Adapter(),window_frames=4,recovery_frames=2)
            with contextlib.redirect_stdout(io.StringIO()):
                result=process_video(video,root/'out',prompts,tracker,max_frames=8,tracks=True,transparent=True)
            self.assertEqual(result['tracking_mode'],'continuous-full-frame')
            self.assertEqual(result['window_frames'],4);self.assertEqual(result['max_window_frames'],6)
            self.assertEqual(result['lost_object_ids'],[])
            self.assertNotIn('crop_size',result)
            with (root/'out/tracking_status.csv').open(encoding='utf-8') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),24)
            self.assertEqual(len(list((root/'out/transparent').glob('*/*.png'))),24)


class EntryTests(unittest.TestCase):
    def entry(self):
        path=PROJECT/'main.py'
        spec=importlib.util.spec_from_file_location('continuous_main_tests',path)
        main=importlib.util.module_from_spec(spec);spec.loader.exec_module(main)
        return main

    def test_vscode_button_defaults_to_continuous_without_changing_selection(self):
        main=self.entry()
        prompts=[ObjectPrompt(i,[i*10,10,i*10+5,20]) for i in [1,2,3]]
        with patch('segmentation.video.first_frame',return_value=np.zeros((150,320,3),np.uint8)), \
             patch('segmentation.selector.select_objects',return_value=prompts), \
             patch('segmentation.tracker.Sam2Tracker'), \
             patch('segmentation.continuous.ContinuousTracker') as continuous, \
             patch('segmentation.focused.FocusedTracker') as local, \
             patch('segmentation.pipeline.process_video',return_value={}) as process, \
             contextlib.redirect_stdout(io.StringIO()):
            main.main([])
        continuous.assert_called_once();self.assertEqual(continuous.call_args.kwargs,{'recovery_frames':60,'enable_reacquisition':True,'enable_small_refinement':True,'enable_prompt_refinement':True})
        local.assert_not_called();self.assertEqual(process.call_args.args[2],prompts)
        self.assertEqual(process.call_args.kwargs['max_seconds'],60)

    def test_continuous_improvements_can_be_disabled_independently(self):
        main=self.entry()
        with patch('segmentation.video.first_frame',return_value=np.zeros((150,320,3),np.uint8)), \
             patch('segmentation.prompts.load_prompts',return_value=[ObjectPrompt(1,[10,10,20,20])]), \
             patch('segmentation.tracker.Sam2Tracker'), \
             patch('segmentation.continuous.ContinuousTracker') as continuous, \
             patch('segmentation.pipeline.process_video',return_value={}), \
             contextlib.redirect_stdout(io.StringIO()):
            main.main(['run','--video','v.mp4','--prompts','p.json','--output',str(WORK/'absent-output'),
                       '--small-objects','--no-reacquisition','--no-small-refinement','--no-reference-refresh'])
        self.assertEqual(continuous.call_args.kwargs,
                         {'recovery_frames':60,'enable_reacquisition':False,'enable_small_refinement':False,'enable_prompt_refinement':False})

    def test_explicit_local_backend_and_crop_size_are_preserved(self):
        main=self.entry()
        with patch('segmentation.video.first_frame',return_value=np.zeros((150,320,3),np.uint8)), \
             patch('segmentation.prompts.load_prompts',return_value=[ObjectPrompt(1,[10,10,20,20])]), \
             patch('segmentation.tracker.Sam2Tracker'), \
             patch('segmentation.continuous.ContinuousTracker') as continuous, \
             patch('segmentation.focused.FocusedTracker') as local, \
             patch('segmentation.pipeline.process_video',return_value={}), \
             contextlib.redirect_stdout(io.StringIO()):
            main.main(['run','--video','v.mp4','--prompts','p.json','--output',str(WORK/'absent-output'),
                       '--small-objects','--tracking-backend','local','--crop-size','512','--recovery-frames','12'])
        continuous.assert_not_called();local.assert_called_once()
        self.assertEqual(local.call_args.kwargs,{'crop_size':512,'recovery_frames':12})


class AdapterMaskTests(unittest.TestCase):
    def tracker(self):
        from segmentation.tracker import Sam2Tracker
        class Tensor:
            def __init__(self,data):self.data=data
            def __gt__(self,value):return Tensor(self.data>value)
            def cpu(self):return self
            def numpy(self):return self.data
        class Predictor:
            def __init__(self):self.seeds=[];self.reset=False;self.ids=[]
            def init_state(self,*a,**kw):return dict(video_height=20,video_width=30,num_frames=4)
            def add_new_mask(self,**kw):self.seeds.append(kw);self.ids.append(kw['obj_id'])
            def add_new_points_or_box(self,**kw):self.seeds.append(kw);self.ids.append(kw['obj_id'])
            def propagate_in_video(self,*a,**kw):
                for i in range(4):yield i,self.ids,Tensor(np.ones((len(self.ids),1,20,30)))
            def reset_state(self,state):self.reset=True
        tracker=Sam2Tracker.__new__(Sam2Tracker)
        tracker.torch=type('Torch',(),{'inference_mode':staticmethod(contextlib.nullcontext)})()
        tracker.device='cpu';tracker.predictor=Predictor()
        return tracker

    def test_mask_api_receives_each_objects_actual_seed_frame(self):
        tracker=self.tracker();mask=np.ones((20,30),bool)
        prompts=[ObjectPrompt(1,[1,1,4,4]),ObjectPrompt(2,[3,3,7,7])]
        rows=list(tracker.predict(Path('.'),prompts,initial_masks={1:mask,2:mask},prompt_frames={1:0,2:2}))
        self.assertEqual([s['frame_idx'] for s in tracker.predictor.seeds],[0,2])
        self.assertTrue(all('mask' in s and 'box' not in s for s in tracker.predictor.seeds))
        self.assertEqual(len(rows),4);self.assertTrue(tracker.predictor.reset)

    def test_bad_mask_or_seed_frame_resets_state(self):
        for mask,index in [(np.zeros((20,30),bool),0),(np.ones((10,10),bool),0),(np.ones((20,30),bool),4)]:
            tracker=self.tracker()
            with self.assertRaises(ValueError):
                list(tracker.predict(Path('.'),[ObjectPrompt(1,[1,1,4,4])],initial_masks={1:mask},prompt_frames={1:index}))
            self.assertTrue(tracker.predictor.reset)

    def test_initial_prompt_error_also_resets_model_state(self):
        tracker=self.tracker()
        def fail(**kwargs):raise RuntimeError('invalid initial prompt')
        tracker.predictor.add_new_points_or_box=fail
        with self.assertRaises(RuntimeError):list(tracker.predict(Path('.'),[ObjectPrompt(1,[1,1,4,4])]))
        self.assertTrue(tracker.predictor.reset)

    def test_recovery_appearance_rejects_different_color_and_too_few_pixels(self):
        from segmentation.motion import same_appearance
        before=np.zeros((40,40,3),np.uint8);after=np.zeros_like(before)
        mask=np.zeros((40,40),bool);mask[10:20,10:20]=True
        before[mask]=(0,200,0);after[mask]=(0,200,0)
        self.assertTrue(same_appearance(before,after,mask,mask))
        after[mask]=(200,0,0)
        self.assertFalse(same_appearance(before,after,mask,mask))
        tiny=np.zeros_like(mask);tiny[15,15]=True
        self.assertFalse(same_appearance(before,before,tiny,tiny))



if __name__=='__main__':unittest.main(argv=[sys.argv[0]],verbosity=2)
