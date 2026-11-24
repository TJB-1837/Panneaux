import unittest

import cv2
import numpy as np

from src.__main__ import get_parser
from src.detector import proposals
from src.vision import color_boxes, filter_predictions, intersection_over_union


class DetectionTests(unittest.TestCase):
    def test_color_is_default_for_live_and_evaluation(self):
        for command in ("live", "evaluate"):
            with self.subTest(command=command):
                args = get_parser().parse_args([command])
                self.assertEqual(args.proposal_mode, "color")

    def test_gray_background_and_solid_clothing_are_rejected(self):
        frame = np.full((240, 320, 3), 160, dtype=np.uint8)
        cv2.rectangle(frame, (20, 20), (80, 80), (255, 255, 255), -1)
        cv2.rectangle(frame, (120, 50), (240, 180), (0, 0, 200), -1)
        self.assertEqual(color_boxes(frame, 16, None), [])

    def test_colored_sign_with_white_symbol_is_kept(self):
        for color in ((0, 0, 220), (220, 40, 0), (0, 220, 220)):
            with self.subTest(color=color):
                frame = np.full((240, 320, 3), 160, dtype=np.uint8)
                cv2.circle(frame, (160, 120), 40, color, -1)
                cv2.rectangle(frame, (130, 110), (190, 130), (255, 255, 255), -1)
                boxes = color_boxes(frame, 16, None)
                self.assertTrue(any(intersection_over_union(b, (120, 80, 201, 161)) > 0.9 for b in boxes))

    def test_candidate_budget_is_respected(self):
        frame = np.full((200, 400, 3), 160, dtype=np.uint8)
        for x in (60, 180, 300):
            cv2.circle(frame, (x, 100), 30, (0, 0, 220), -1)
            cv2.rectangle(frame, (x - 22, 94), (x + 22, 106), (255, 255, 255), -1)
        args = get_parser().parse_args(["live", "--proposal-mode", "color", "--max-candidates", "2"])
        self.assertEqual(len(proposals(args)(frame)), 2)

    def test_priority_diamond_with_pale_border_is_kept(self):
        for brightness in (1.0, 0.55):
            with self.subTest(brightness=brightness):
                frame = np.full((240, 320, 3), 40, dtype=np.uint8)
                outer = np.array([[160, 60], [214, 120], [160, 180], [106, 120]])
                inner = np.array([[160, 86], [190, 120], [160, 154], [130, 120]])
                cv2.fillConvexPoly(frame, outer, (230, 230, 230))
                cv2.fillConvexPoly(frame, inner, (0, 200, 220))
                frame = (frame * brightness).astype(np.uint8)
                boxes = color_boxes(frame, 16, None)
                self.assertTrue(any(intersection_over_union(b, (106, 60, 215, 181)) > 0.8
                                    for b in boxes))
                self.assertEqual(color_boxes(frame, 16, 80), [])

    def test_solid_yellow_shapes_without_border_are_rejected(self):
        for background in (40, 160):
            for diamond in (False, True):
                with self.subTest(background=background, diamond=diamond):
                    frame = np.full((240, 320, 3), background, dtype=np.uint8)
                    if diamond:
                        points = np.array([[160, 80], [200, 120], [160, 160], [120, 120]])
                        cv2.fillConvexPoly(frame, points, (0, 200, 220))
                    else:
                        cv2.rectangle(frame, (120, 80), (200, 160), (0, 200, 220), -1)
                    self.assertEqual(color_boxes(frame, 16, None), [])

    def test_overlap_filter_keeps_best_score_and_distinct_signs(self):
        predictions = [((10, 10, 40, 40), 1, 0.85), ((11, 11, 41, 41), 2, 0.99),
                       ((60, 10, 90, 40), 1, 0.9), ((100, 10, 130, 40), 3, 0.5)]
        self.assertEqual(filter_predictions(predictions, 0.8), [predictions[1], predictions[2]])

    def test_canny_mode_is_available_for_low_saturation_signs(self):
        frame = np.full((160, 160, 3), 160, dtype=np.uint8)
        cv2.rectangle(frame, (40, 40), (120, 120), (0, 0, 0), 3)
        args = get_parser().parse_args(["live", "--proposal-mode", "canny"])
        self.assertTrue(proposals(args)(frame))


if __name__ == "__main__":
    unittest.main()
