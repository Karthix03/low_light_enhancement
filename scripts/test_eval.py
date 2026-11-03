import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
from src.utils import load_config
from src.detection import ObjectDetector
from src.distance import DistanceCalculator

config = load_config()
config['detection']['conf_threshold'] = 0.50
detector = ObjectDetector(config)
calc = DistanceCalculator(config)

img = cv2.imread('data/realistic_night_road.jpg')
img = cv2.resize(img, (1280, 720))
dets = detector.detect(img)
print(f"Detections >= 0.50: {len(dets)}")
calc.compute_all_object_depths(dets, img.shape)
for d in dets:
    print(f"  {d['class_name']} {d['confidence']*100:.0f}%: depth_z={d['depth_z']}m, dist_cam={d['distance_cam']}m, 3D=({d['x_3d']}m, {d['y_3d']}m)")

pairs = calc.calculate_pairwise_distances(dets, img.shape)
print("\nPairwise distances:")
for p in pairs:
    print(f"  {p['obj_a']['class_name']} to {p['obj_b']['class_name']}: {p['distance']}m [{p['status'].upper()}]")

alerts = calc.get_proximity_alerts(pairs)
print(f"\nAlerts count: {len(alerts)}")
for a in alerts:
    print(f"  [{a['severity']}] {a['message']}")
