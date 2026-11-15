"""Identity adaptation must tolerate gradual lighting without learning unrelated targets."""
import unittest
import cv2
import numpy as np
from segmentation.motion import AppearanceMemory,appearance_histogram,appearance_matches


def image(hue=5,saturation=220):
    hsv=np.zeros((40,40,3),np.uint8)
    hsv[:,:,0]=hue;hsv[:,:,1]=saturation;hsv[:,:,2]=210
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)


class AppearanceTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.zeros((40,40),bool);self.mask[8:32,8:32]=True

    def test_hue_wrap_is_close_but_green_is_not(self):
        memory=AppearanceMemory(image(179),self.mask)
        self.assertTrue(memory.matches(image(1),self.mask))
        self.assertFalse(memory.matches(image(65),self.mask))

    def test_gradual_lighting_adapts_where_first_histogram_rejects(self):
        memory=AppearanceMemory(image(),self.mask)
        first=appearance_histogram(image(),self.mask)
        for saturation in range(220,74,-2):
            current=image(saturation=saturation)
            self.assertTrue(memory.matches(current,self.mask),saturation)
            memory.observe(current,self.mask,overlap=1,area_ratio=1)
        self.assertFalse(appearance_matches(first,current,self.mask,.7))
        self.assertTrue(memory.matches(current,self.mask))

    def test_occluded_or_uncertain_samples_do_not_update(self):
        memory=AppearanceMemory(image(),self.mask);before=memory.recent.copy()
        self.assertFalse(memory.observe(image(8),self.mask,overlap=.1,area_ratio=1))
        self.assertFalse(memory.observe(image(8),self.mask,overlap=1,area_ratio=.1))
        self.assertFalse(memory.observe(image(65),self.mask,overlap=1,area_ratio=1))
        np.testing.assert_array_equal(memory.recent,before)

    def test_empty_and_under_four_pixels_are_not_color_evidence(self):
        empty=np.zeros_like(self.mask);tiny=empty.copy();tiny[10,10]=True
        memory=AppearanceMemory(image(),tiny)
        self.assertIsNone(memory.distance(image(),self.mask))
        self.assertFalse(memory.matches(image(),empty))


if __name__=='__main__':unittest.main()