from shapely.geometry import Polygon
import numpy as np

def add_affinity(bbox_1, bbox_2):

	"""
		Add gaussian heatmap for affinity bbox to the image between bbox_1, bbox_2
		:param image: 2-d array containing affinity heatmap
		:param bbox_1: np.array, dtype=np.int32, shape = [4, 2]
		:param bbox_2: np.array, dtype=np.int32, shape = [4, 2]
		:return: image in which the gaussian affinity bbox has been added
	"""

	center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)

	# ToDo - No guarantee that bbox is ordered, hence affinity can be wrong

	# Shifted the affinity so that adjacent affinity do not touch each other

	tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
	bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
	tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
	br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)

	affinity = np.array([tl, tr, br, bl])

	return affinity


def generate_affinity_box(character_bbox, text):

	"""
	:param image_size: shape = [3, image_height, image_width]
	:param character_bbox: [2, 4, num_characters]
	:param text: [num_words]
	:param weight: This is currently used only for synth-text so specifying weight as not None will generate a heatmap
					having value one where there is affinity
	:return: if weight is not None then target_affinity_heatmap otherwise target_affinity_heatmap,
																				weight for weak-supervision
	"""

	total_letters = 0

	all_affinity_bbox = []


	for word in text:
		for char_num in range(len(word)-1):
			try:
				bbox = add_affinity(character_bbox[total_letters].copy(), character_bbox[total_letters+1].copy())
			except IndexError:
				print("charbbox len:", len(character_bbox))
				print("tried to index:", total_letters)
				print(text)
				raise
			total_letters += 1
			all_affinity_bbox.append(bbox)
		total_letters += 1

	return np.array(all_affinity_bbox)