import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from segmentation.video import write_image
from tools.run_cctv_sam2 import load_selection
from tools.evaluate_cctv_boxes import evaluate


class CctvExperimentContracts(unittest.TestCase):
    def test_later_frame_and_single_target_filter(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'selection.json'
            doc = {'schema_version': 1, 'width': 10, 'height': 10, 'frame_index': 160,
                   'objects': [{'object_id': i, 'box': [1, 1, 8, 8]} for i in (1, 2, 3)]}
            path.write_text(json.dumps(doc), encoding='utf-8')
            selected, prompts = load_selection(path, [1])
            self.assertEqual(selected['frame_index'], 160)
            self.assertEqual([p.object_id for p in prompts], [1])
            for requested in ([4], [1, 1]):
                with self.assertRaises(ValueError):
                    load_selection(path, requested)
            doc['frame_index'] = -1
            path.write_text(json.dumps(doc), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_selection(path)

    def test_known_box_empty_and_other_person_proxy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            xml = root/'reference.xml'
            objects = '<object id="0"><appearance>visible</appearance><box xc="3" yc="3" w="4" h="4"/></object><object id="1"><appearance>visible</appearance><box xc="8" yc="8" w="2" h="2"/></object>'
            xml.write_text('<sequence>'+''.join(f'<frame number="{i}"><objectlist>{objects}</objectlist></frame>' for i in range(3))+'</sequence>', encoding='utf-8')
            correct = np.zeros((10, 10), np.uint8);correct[1:5, 1:5] = 255
            empty = np.zeros_like(correct)
            other = empty.copy();other[7:9, 7:9] = 255
            for i, mask in enumerate([correct, empty, other]):
                write_image(root/f'masks/object_1/{i:06d}.png', mask)
            result = evaluate(root/'masks', xml, {1: 0}, 0)['summary']
            self.assertEqual(result['visible_observations'], 3)
            self.assertEqual(result['box_hits_IoU05'], 1)
            self.assertEqual(result['visible_empty_outputs'], 1)
            self.assertEqual(result['other_box_proxy_NOT_confirmed_IDswitches'], 1)
            self.assertAlmostEqual(result['mean_bbox_iou'], 1/3)
            (root/'masks/object_1/000002.png').unlink()
            with self.assertRaises(FileNotFoundError):
                evaluate(root/'masks', xml, {1: 0}, 0)


if __name__ == '__main__':
    unittest.main()
