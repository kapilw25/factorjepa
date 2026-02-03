*Date: Mon Feb 2, 2026*


Retrieval-Based, Label-Free Evaluation of V-JEPA on Indian Urban Walking Videos
-------------------------------------------------------------
We take a large collection of 20–30 minute Indian street-walk videos and split them into many short (≈10-second) clips. Without training anything, we pass each clip through a frozen V-JEPA model to obtain a compact embedding that represents the clip’s overall scene and motion context. We then treat this embedding space as an “urban similarity map”: given any query clip, we retrieve its nearest neighbors by kNN and inspect whether the neighbors correspond to the same kind of place and conditions (e.g., markets, intersections, temple streets, beach roads; day vs night; crowded vs sparse; heavy vs light traffic). To make this evaluation objective without labels, we also measure whether neighborhoods are self-consistent (a clip’s nearest neighbor points back to it) and stable under benign view changes (two different crops/resolutions of the same clip retrieve largely the same neighbors). If V-JEPA has learned transferable world structure, Indian urban scenes that feel similar to humans should cluster tightly and consistently—giving us a fast, training-free sanity check of representation quality on a new domain.


-------------------------------------------------------------
Auto-tag each clip with Video-LLaVA
For each clip, ask for structured tags, not a free-form caption. Example output fields:

scene_type (market / junction / residential lane / promenade / transit / temple-tourist / highway / alley, etc.)

time_of_day (morning/afternoon/evening/night)

weather (clear/rain/fog)

crowd_density (low/med/high)

traffic_density (low/med/high)

notable_objects (bus, auto-rickshaw, bike, street vendor, police, signage, animals)

road_layout (intersection, narrow lane, wide road, sidewalk present, median)

-------------------------------------------------------------
step1: YouTube channel, it has almost 700 videos centred specifically around what we're looking for: https://youtube.com/@walkinginindia?si=inn2Yzcd6XEB3N08 
step2: Download all. scene detect - https://www.scenedetect.com/ or similar techniques. break down videos to 4-5 seconds shots. those are our dataset
step3: we will do object segmentation using SAM3 - https://www.youtube.com/watch?v=uV37YsWj-JQ
step4: V-JEPA 2: A frontier video understanding model developed by FAIR, Meta, which extends the pretraining objectives of VJEPA, resulting in state-of-the-art video understanding capabilities, leveraging data and model sizes at scale. The code is released in this repository.
-------------------------------------------------------------

*do not modify these notes, keep them raw*