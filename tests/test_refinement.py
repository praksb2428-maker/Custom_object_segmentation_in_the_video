import unittest
import numpy as np
from segmentation.refinement import SmallObjectRefiner


class CropAdapter:
    def __init__(self, quality=.95, wrong_shape=False, broad=False):
        self.quality, self.wrong_shape, self.broad = quality, wrong_shape, broad
        self.last_shape = None
    def refine_image(self, image, prompt):
        self.last_shape = image.shape
        if self.wrong_shape:
            return np.zeros((2, 2), bool), self.quality
        mask = image[:, :, 1] > 100
        if self.broad: mask[:] = True
        # Transformed prompts must still be inside the actual crop.
        for x, y in prompt.points:
            assert 0 <= x < image.shape[1] and 0 <= y < image.shape[0]
        return mask, self.quality


class RefinementTests(unittest.TestCase):
    def fixture(self):
        image = np.zeros((300, 400, 3), np.uint8)
        image[90:100, 120:131, 1] = 220
        mask = np.zeros((300, 400), bool); mask[90:100, 120:130] = True
        return image, mask

    def test_restores_source_coordinates_with_only_actual_model_pixels(self):
        image, mask = self.fixture(); adapter = CropAdapter(); refiner = SmallObjectRefiner(adapter)
        result = refiner.refine(image, mask)
        np.testing.assert_array_equal(result, image[:, :, 1] > 100)
        self.assertEqual(result.shape, mask.shape)
        self.assertEqual(adapter.last_shape, (64, 64, 3))
        self.assertEqual((refiner.calls, refiner.accepted), (1, 1))

    def test_uncertain_broad_or_invalid_quality_keeps_original(self):
        image, mask = self.fixture()
        for adapter in [CropAdapter(.4), CropAdapter(float('nan')), CropAdapter(broad=True)]:
            with self.subTest(quality=adapter.quality, broad=adapter.broad):
                refiner = SmallObjectRefiner(adapter)
                np.testing.assert_array_equal(refiner.refine(image, mask), mask)
                self.assertEqual(refiner.accepted, 0)

    def test_distant_same_size_mask_keeps_original(self):
        image, mask = self.fixture(); image[:] = 0; image[70:80, 100:110, 1] = 220
        np.testing.assert_array_equal(SmallObjectRefiner(CropAdapter()).refine(image, mask), mask)

    def test_empty_and_large_masks_skip_extra_inference(self):
        image, mask = self.fixture(); mask[:] = False
        refiner = SmallObjectRefiner(CropAdapter())
        self.assertIs(refiner.refine(image, mask), mask)
        mask[10:110, 10:110] = True
        self.assertIs(refiner.refine(image, mask), mask)
        self.assertEqual(refiner.calls, 0)

    def test_wrong_model_output_dimensions_raise(self):
        image, mask = self.fixture()
        with self.assertRaises(RuntimeError):
            SmallObjectRefiner(CropAdapter(wrong_shape=True)).refine(image, mask)
