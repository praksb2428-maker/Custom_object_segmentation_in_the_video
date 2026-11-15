import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('evaluate_real_samples',Path(__file__).resolve().parents[1]/'tools/evaluate_real_samples.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class EvaluationGateTests(unittest.TestCase):
    def test_extra_points_cannot_substitute_for_original_sparse_improvement(self):
        a={oid:{'background_hits':0,'other_object_hits':0,'positive_hits':8 if oid=='1' else 2,
                'existing_sparse_hits':8 if oid=='1' else 2,'mean_approximate_iou':.5} for oid in ['1','2']}
        b={oid:dict(row) for oid,row in a.items()}
        b['2'].update(positive_hits=4,mean_approximate_iou=.8)
        result=module.compare(a,b)
        self.assertFalse(result['pass'])
        self.assertIn('object2 original sparse coverage did not improve',result['findings'])
