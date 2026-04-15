                                                                                                                                

● Explanation of each code block from facebook/sam3 model card
                                                                                        
The model card documents 7 distinct usage paths. I'll group them by "what SAM3 variant is used" because that determines what goes where in our pipeline.                  
                                                                                                                                                                        
Section 1 — Raw sam3 package (what we use now)                                                                                                                            
                                                                                                                                                                        
Block 1a — build_sam3_image_model: single-image text-prompted concept segmentation. Not useful for video.                                                                 

Block 1b — build_sam3_video_predictor + handle_request: what our current m10_sam_segment.py uses. Has the propagate_in_video + max_frame_num_to_track bugs (#33/#35). ❌  
keeps the bugs we fixed but are slow for.
                                                                                                                                                                        
Section 2 — HF Sam3Model (images, open-vocab concept segmentation)                                                                                                        

┌────────────────────────┬────────────────────────────────────────────────┬───────────────────────────────────────────┬────────────────────────────────────────┐          
│         Block          │                     Input                      │                  Output                   │                Our use                 │
├────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤
│ 2a Text-Only           │ text="ear"                                     │ masks + boxes + scores for all instances  │ Image-only, not for our 16-frame clips │
├────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤
│ 2b Single box          │ input_boxes=[[box]] + input_boxes_labels=[[1]] │ masks + boxes + scores                    │ Single-image refinement of a box       │          
├────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤          
│ 2c Multi box (pos/neg) │ input_boxes=[[b1,b2]], labels=[[1,1]]          │ Refined concept mask                      │ Add negatives to exclude regions       │          
├────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤          
│ 2d Text + box (hybrid) │ text + input_boxes + labels                    │ Refined concept with exclusions           │ Very useful for images, NOT video      │
├────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤          
│ 2e Batched text        │ Multiple (image, text) pairs                   │ Parallel forwards                         │ Throughput win for image-only          │
├────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤          
│ 2f Batched mixed       │ Mix text-only and box-only across batch        │ One-pass multi-prompt                     │ Highest-throughput image path          │
├────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤          
│ 2g Semantic + instance │ Text input                                     │ outputs.semantic_seg + outputs.pred_masks │ We only need instance masks            │
└────────────────────────┴────────────────────────────────────────────────┴───────────────────────────────────────────┴────────────────────────────────────────┘          
                
Section 3 — HF Sam3VideoModel (video concept segmentation, PCS)                                                                                                           
                
Block 3a — Pre-loaded video, multiple text prompts:                                                                                                                       
model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
processor.add_text_prompt(session, prompts=["person", "bed", "lamp"])                                                                                                     
for model_outputs in model.propagate_in_video_iterator(                                                                                                                   
    inference_session=session, max_frame_num_to_track=50                                                                                                                  
):                                                                                                                                                                        
    processed_outputs = processor.postprocess_outputs(session, model_outputs)                                                                                             
✓ Text prompts only (list of categories supported). ✓ max_frame_num_to_track WORKS. ✓ Single model handles detect+track+propagate.                                        
                                                                                                                                                                        
Block 3b — Streaming variant: frame-by-frame, disables hotstart heuristics (⚠️  "may result in more false positive detections").                                           
                                                                                                                                                                        
Section 4 — HF Sam3TrackerModel (images, point/box-prompted tracking, SAM2 drop-in)                                                                                       
                                                                                                                                                                        
Single-image, point/box/mask inputs per object. Not for video.                                                                                                            
                
Section 5 — HF Sam3TrackerVideoModel (video point/box/mask tracking, SAM2-Video drop-in)                                                                                  
                
Block 5a — Video tracking with points/boxes:                                                                                                                              
model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
processor.add_inputs_to_inference_session(                                                                                                                                
    inference_session=session,                                                                                                                                            
    frame_idx=0, obj_ids=[2, 3],
    input_boxes=[[[b1], [b2]]],  # boxes from our DINO go here                                                                                                            
)                                                             
for output in model.propagate_in_video_iterator(                                                                                                                          
    inference_session=session,                  
    max_frame_num_to_track=3,  # WORKS here                                                                                                                               
    start_frame_idx=anchor,                
):                                                                                                                                                                        
    mask = output.pred_masks  # (B, num_masks, H, W)                                                                                                                      
✓ Accepts BOX prompts per object. ✓ max_frame_num_to_track WORKS. ✓ Drop-in replacement for what we're doing in raw sam3 pkg.
                                                                                                                                                                        
Best-Accuracy and Best-Throughput analysis for our pipeline                                                                                                               
                                                                                                                                                                        
Our pipeline needs: 16-frame video clips, ≥17 Indian agent categories, per-frame agent_mask for D_L/D_A/D_I, multi-anchor to avoid tracking drift.                        
                                                                                                                                                                        
Two viable HF paths                                                                                                                                                       
                
┌───────────────────────────────────────────┬────────────────────────┬───────────────────────────────────────────────────┬───────────────────────────────────────────┐    
│                   Path                    │        Pipeline        │                     Accuracy                      │                Throughput                 │
├───────────────────────────────────────────┼────────────────────────┼───────────────────────────────────────────────────┼───────────────────────────────────────────┤    
│ P-3a: SAM3 Video PCS alone (Block 3a) —   │ text → SAM3 (detect +  │ HIGHEST IF text grounding works on Indian         │ HIGHEST: ~1 model fwd vs DINO+SAM3.       │
│ drop DINO, use Sam3VideoModel with our    │ track in one model)    │ objects. Unified model trained on 4M concepts,    │ Single optimization surface               │
│ 17-cat taxonomy as list of text prompts   │                        │ benefits from hotstart heuristics that filter FPs │                                           │    
├───────────────────────────────────────────┼────────────────────────┼───────────────────────────────────────────────────┼───────────────────────────────────────────┤
│ P-5a: DINO + SAM3 Tracker Video (Block    │ DINO → boxes →         │ SAFER: DINO-on-frame-0 already validated on       │ Lower — 2 model forwards per anchor, but  │    
│ 5a) — keep DINO box detection, replace    │ SAM3TrackerVideo       │ Indian scenes (1286 agents detected on 100 dense  │ still 10× faster than raw sam3 pkg        │    
│ raw sam3 pkg with HF                      │ refines + tracks       │ clips, 17-cat taxonomy worked). No reliance on    │ because max_frame_num_to_track works      │
│ Sam3TrackerVideoModel                     │                        │ SAM3's text encoder for Indian terms              │                                           │    
└───────────────────────────────────────────┴────────────────────────┴───────────────────────────────────────────────────┴───────────────────────────────────────────┘

The tradeoff

P-3a (theoretical winner) throws away our validated DINO step. Its accuracy depends entirely on whether Sam3VideoModel's text encoder grounds our 17 Indian-street        
categories (auto rickshaw, vendor, pedestrian, etc.) — exactly the capability that FAILED in the raw sam3 pkg on our original dense100 experiment (10/15 SANITY clips had
wrong/missing masks, motivating the DINO pivot).                                                                                                                          
                
P-5a (safer) keeps the validated DINO layer and only replaces the broken SAM3 propagation. Same accuracy as our current pipeline with 10× less compute.                   

My recommendation                                                                                                                                                         
                
Implement P-5a in v2_HF.py (keep DINO, swap raw sam3 → HF Sam3TrackerVideoModel).                                                                                         

Also add a small P-3a probe (20 lines): run Sam3VideoModel with our taxonomy as text on 5 representative clips (dense market + sparse Goa + monument), compare mask       
quality to P-5a. If P-3a masks look equivalent or better on Indian scenes → switch to P-3a for FULL (saves DINO compute entirely). If P-3a's Indian-object text grounding
is weak (as it was in raw pkg) → stick with P-5a.                                                                                                                         
                
Decision tree:
1. Build v2_HF with P-5a (DINO + Sam3TrackerVideoModel) — safe, known-good accuracy
2. Run on POC dense100 → validate quality                                          
3. Spot-test P-3a (Sam3VideoModel text-only) on 5 clips                                                                                                                   
4. If P-3a ≥ P-5a quality → switch to P-3a for FULL (simpler, faster)                                                                                                     
    Otherwise → ship P-5a for FULL                                                                                                                                         
                                                                                                                                                                        
This keeps v2_HF.py focused on one backend (P-5a) while leaving the door open for P-3a if testing shows it's viable.                                                      
                                                                                                                                                                        
Proceed with implementing P-5a in src/m10_sam_segment_v2_HF.py?                                                                                                           
                                                                                                                                                                        
Sources:                                                                                                                                                                  
- https://huggingface.co/facebook/sam3 — all code blocks above
- https://huggingface.co/docs/transformers/en/model_doc/sam3_video — Sam3VideoModel API + max_frame_num_to_track confirmation                                             
- https://huggingface.co/docs/transformers/model_doc/sam3_tracker_video — Sam3TrackerVideoModel box-prompt API               

Hugging Face's logo
Hugging Face
Models
Datasets
Spaces
Buckets
new
Docs
Pricing


facebook
/
sam3 

like
1.86k

Follow
AI at Meta
12.3k
Mask Generation
Transformers
Safetensors
English
sam3_video
feature-extraction
sam3

License:
other
Model card
Files and versions
xet
Community
220
Gated model
You have been granted access to this model

SAM 3 is a unified foundation model for promptable segmentation in images and videos. It can detect, segment, and track objects using text or visual prompts such as points, boxes, and masks. Compared to its predecessor SAM 2, SAM 3 introduces the ability to exhaustively segment all instances of an open-vocabulary concept specified by a short text phrase or exemplars. Unlike prior work, SAM 3 can handle a vastly larger set of open-vocabulary prompts. It achieves 75-80% of human performance on our new SA-CO benchmark which contains 270K unique concepts, over 50 times more than existing benchmarks.

Hugging Face 🤗 app

Basic Usage
import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("<YOUR_IMAGE_PATH.jpg>")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

#################################### For Video ####################################

from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()
video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, # Arbitrary frame index
        text="<YOUR_TEXT_PROMPT>",
    )
)
output = response["outputs"]

The official code is publicly released in the sam3 repo.

Usage with 🤗 Transformers
SAM3 - Promptable Concept Segmentation (PCS) for Images
SAM3 performs Promptable Concept Segmentation (PCS) on images, taking text and/or image exemplars as prompts and returning segmentation masks for all matching object instances in the image.

Text-Only Prompts
from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# Load image
image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# Segment using text prompt
inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

print(f"Found {len(results['masks'])} objects")
# Results contain:
# - masks: Binary masks resized to original image size
# - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
# - scores: Confidence scores

You can display masks using a simple helper like the following:

import numpy as np
import matplotlib

def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image

Then you can save the resulting composite image or display it in a notebook:

overlay_masks(image, results["masks"])

Single Bounding Box Prompt
Segment objects using a bounding box:

# Box in xyxy format: [x1, y1, x2, y2] in pixel coordinates
# Example: laptop region
box_xyxy = [100, 150, 500, 450]
input_boxes = [[box_xyxy]]  # [batch, num_boxes, 4]
input_boxes_labels = [[1]]  # 1 = positive box

inputs = processor(
    images=image,
    input_boxes=input_boxes,
    input_boxes_labels=input_boxes_labels,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

Multiple Box Prompts (Positive and Negative)
Use multiple boxes with positive and negative labels to refine the concept:

# Load kitchen image
kitchen_url = "http://images.cocodataset.org/val2017/000000136466.jpg"
kitchen_image = Image.open(requests.get(kitchen_url, stream=True).raw).convert("RGB")

# Define two positive boxes (e.g., dial and button on oven)
# Boxes are in xyxy format [x1, y1, x2, y2] in pixel coordinates
box1_xyxy = [59, 144, 76, 163]  # Dial box
box2_xyxy = [87, 148, 104, 159]  # Button box
input_boxes = [[box1_xyxy, box2_xyxy]]
input_boxes_labels = [[1, 1]]  # Both positive

inputs = processor(
    images=kitchen_image,
    input_boxes=input_boxes,
    input_boxes_labels=input_boxes_labels,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]
overlay_masks(kitchen_image, results["masks"])

Combined Prompts (Text + Negative Box)
Use text prompts with negative visual prompts to refine the concept:

# Segment "handle" but exclude the oven handle using a negative box
text = "handle"
# Negative box covering oven handle area (xyxy): [40, 183, 318, 204]
oven_handle_box = [40, 183, 318, 204]
input_boxes = [[oven_handle_box]]

inputs = processor(
    images=kitchen_image,
    text=text,
    input_boxes=input_boxes,
    input_boxes_labels=[[0]],  # 0 = negative (exclude this region)
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]
# This will segment pot handles but exclude the oven handle

Batched Inference with Text Prompts
Process multiple images with different text prompts by batch:

cat_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
kitchen_url = "http://images.cocodataset.org/val2017/000000136466.jpg"
images = [
    Image.open(requests.get(cat_url, stream=True).raw).convert("RGB"),
    Image.open(requests.get(kitchen_url, stream=True).raw).convert("RGB")
]

text_prompts = ["ear", "dial"]

inputs = processor(images=images, text=text_prompts, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results for both images
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)

print(f"Image 1: {len(results[0]['masks'])} objects found")
print(f"Image 2: {len(results[1]['masks'])} objects found")

Batched Mixed Prompts
Use different prompt types for different images in the same batch:

# Image 1: text prompt "laptop"
# Image 2: visual prompt (dial box)
box2_xyxy = [59, 144, 76, 163]

inputs = processor(
    images=images,
    text=["laptop", None],  # Only first image has text
    input_boxes=[None, [box2_xyxy]],  # Only second image has box
    input_boxes_labels=[None, [1]],  # Positive box for second image
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results for both images
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)
# Both images processed in single forward pass

Semantic Segmentation Output
SAM3 also provides semantic segmentation alongside instance masks:

inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Instance segmentation masks
instance_masks = torch.sigmoid(outputs.pred_masks)  # [batch, num_queries, H, W]

# Semantic segmentation (single channel)
semantic_seg = outputs.semantic_seg  # [batch, 1, H, W]

print(f"Instance masks: {instance_masks.shape}")
print(f"Semantic segmentation: {semantic_seg.shape}")

SAM3 Video - Promptable Concept Segmentation (PCS) for Videos
SAM3 Video performs Promptable Concept Segmentation (PCS) on videos, taking text as prompts and detecting and tracking all matching object instances across video frames.

Pre-loaded Video Inference
Process a video with all frames already available using text prompts:

from transformers import Sam3VideoModel, Sam3VideoProcessor
from accelerate import Accelerator
import torch

device = Accelerator().device
model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

# Load video frames
from transformers.video_utils import load_video
video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
video_frames, _ = load_video(video_url)

# Initialize video inference session
inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=torch.bfloat16,
)

# Add text prompt to detect and track objects
text = "person"
inference_session = processor.add_text_prompt(
    inference_session=inference_session,
    text=text,
)

# Process all frames in the video
outputs_per_frame = {}
for model_outputs in model.propagate_in_video_iterator(
    inference_session=inference_session, max_frame_num_to_track=50
):
    processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
    outputs_per_frame[model_outputs.frame_idx] = processed_outputs

print(f"Processed {len(outputs_per_frame)} frames")
Processed 51 frames

# Access results for a specific frame
frame_0_outputs = outputs_per_frame[0]
print(f"Detected {len(frame_0_outputs['object_ids'])} objects")
print(f"Object IDs: {frame_0_outputs['object_ids'].tolist()}")
print(f"Scores: {frame_0_outputs['scores'].tolist()}")
print(f"Boxes shape (XYXY format, absolute coordinates): {frame_0_outputs['boxes'].shape}")
print(f"Masks shape: {frame_0_outputs['masks'].shape}")

Streaming Video Inference
For real-time applications, the Transformers implementation of SAM3 Video supports processing video frames as they arrive:

# Initialize session for streaming
streaming_inference_session = processor.init_video_session(
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=torch.bfloat16,
)

# Add text prompt
text = "person"
streaming_inference_session = processor.add_text_prompt(
    inference_session=streaming_inference_session,
    text=text,
)

# Process frames one by one (streaming mode)
streaming_outputs_per_frame = {}
for frame_idx, frame in enumerate(video_frames[:50]):  # Process first 50 frames
    # First, process the frame using the processor
    inputs = processor(images=frame, device=device, return_tensors="pt")
...
    # Process frame using streaming inference - pass the processed pixel_values
    model_outputs = model(
        inference_session=streaming_inference_session,
        frame=inputs.pixel_values[0],  # Provide processed frame - this enables streaming mode
        reverse=False,
    )
...
    # Post-process outputs with original_sizes for proper resolution handling
    processed_outputs = processor.postprocess_outputs(
        streaming_inference_session,
        model_outputs,
        original_sizes=inputs.original_sizes,  # Required for streaming inference
    )
    streaming_outputs_per_frame[frame_idx] = processed_outputs
...
    if (frame_idx + 1) % 10 == 0:
        print(f"Processed {frame_idx + 1} frames...")

print(f"✓ Streaming inference complete! Processed {len(streaming_outputs_per_frame)} frames")
✓ Streaming inference complete! Processed 50 frames

# Access results
frame_0_outputs = streaming_outputs_per_frame[0]
print(f"Detected {len(frame_0_outputs['object_ids'])} objects in first frame")
print(f"Boxes are in XYXY format (absolute pixel coordinates): {frame_0_outputs['boxes'].shape}")
print(f"Masks are at original video resolution: {frame_0_outputs['masks'].shape}")

⚠️ **Note on Streaming Inference Quality**: Streaming inference disables hotstart heuristics that remove unmatched and duplicate objects, as these require access to future frames to make informed decisions. This may result in more false positive detections and duplicate object tracks compared to pre-loaded video inference. For best results, use pre-loaded video inference when all frames are available.
SAM3 Tracker - Promptable Visual Segmentation (PVS) for Images
Sam3Tracker performs Promptable Visual Segmentation (PVS) on images, taking interactive visual prompts (points, boxes, masks) to segment a specific object instance per prompt. It is an updated version of SAM2 that maintains the same API while providing improved performance, making it a drop-in replacement for SAM2 workflows.

Automatic Mask Generation with Pipeline
from transformers import pipeline

generator = pipeline("mask-generation", model="facebook/sam3", device=0)
image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
outputs = generator(image_url, points_per_batch=64)

len(outputs["masks"])  # Number of masks generated

Basic Image Segmentation
Single Point Click
from transformers import Sam3TrackerProcessor, Sam3TrackerModel
from accelerate import Accelerator
import torch
from PIL import Image
import requests

device = Accelerator().device

model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")

image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

input_points = [[[[500, 375]]]]  # Single point click, 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
input_labels = [[[1]]]  # 1 for positive click, 0 for negative click, 3 dimensions (image_dim, object_dim, point_label)

inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

# The model outputs multiple mask predictions ranked by quality score
print(f"Generated {masks.shape[1]} masks with shape {masks.shape}")

Multiple Points for Refinement
# Add both positive and negative points to refine the mask
input_points = [[[[500, 375], [1125, 625]]]]  # Multiple points for refinement
input_labels = [[[1, 1]]]  # Both positive clicks

inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

Bounding Box Input
# Define bounding box as [x_min, y_min, x_max, y_max]
input_boxes = [[[75, 275, 1725, 850]]]

inputs = processor(images=raw_image, input_boxes=input_boxes, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

Multiple Objects Segmentation
# Define points for two different objects
input_points = [[[[500, 375]], [[650, 750]]]]  # Points for two objects in same image
input_labels = [[[1], [1]]]  # Positive clicks for both objects

inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs, multimask_output=False)

# Each object gets its own mask
masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
print(f"Generated masks for {masks.shape[0]} objects")
Generated masks for 2 objects

Batch Inference
# Load multiple images
image_urls = [
    "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dog-sam.png"
]
raw_images = [Image.open(requests.get(url, stream=True).raw).convert("RGB") for url in image_urls]

# Single point per image
input_points = [[[[500, 375]]], [[[770, 200]]]]  # One point for each image
input_labels = [[[1]], [[1]]]  # Positive clicks for both images

inputs = processor(images=raw_images, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs, multimask_output=False)

# Post-process masks for each image
all_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
print(f"Processed {len(all_masks)} images, each with {all_masks[0].shape[0]} objects")

SAM3 Tracker Video - Promptable Visual Segmentation (PVS) for Videos
Sam3TrackerVideo performs Promptable Visual Segmentation (PVS) on videos, taking interactive visual prompts (points, boxes, masks) to track a specific object instance per prompt across video frames. It is an updated version of SAM2 Video that maintains the same API while providing improved performance, making it a drop-in replacement for SAM2 Video workflows.

Basic Video Tracking
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from accelerate import Accelerator
import torch

device = Accelerator().device
model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

# Load video frames
from transformers.video_utils import load_video
video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
video_frames, _ = load_video(video_url)

# Initialize video inference session
inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    dtype=torch.bfloat16,
)

# Add click on first frame to select object
ann_frame_idx = 0
ann_obj_id = 1
points = [[[[210, 350]]]]
labels = [[[1]]]

processor.add_inputs_to_inference_session(
    inference_session=inference_session,
    frame_idx=ann_frame_idx,
    obj_ids=ann_obj_id,
    input_points=points,
    input_labels=labels,
)

# Segment the object on the first frame (optional, you can also propagate the masks through the video directly)
outputs = model(
    inference_session=inference_session,
    frame_idx=ann_frame_idx,
)
video_res_masks = processor.post_process_masks(
    [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
)[0]
print(f"Segmentation shape: {video_res_masks.shape}")
Segmentation shape: torch.Size([1, 1, 480, 854])

# Propagate through the entire video
video_segments = {}
for sam3_tracker_video_output in model.propagate_in_video_iterator(inference_session):
    video_res_masks = processor.post_process_masks(
        [sam3_tracker_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
    )[0]
    video_segments[sam3_tracker_video_output.frame_idx] = video_res_masks

print(f"Tracked object through {len(video_segments)} frames")
Tracked object through 180 frames

Multi-Object Video Tracking
Track multiple objects simultaneously across video frames:

# Reset for new tracking session
inference_session.reset_inference_session()

# Add multiple objects on the first frame
ann_frame_idx = 0
obj_ids = [2, 3]
input_points = [[[[200, 300]], [[400, 150]]]]  # Points for two objects (batched)
input_labels = [[[1], [1]]]

processor.add_inputs_to_inference_session(
    inference_session=inference_session,
    frame_idx=ann_frame_idx,
    obj_ids=obj_ids,
    input_points=input_points,
    input_labels=input_labels,
)

# Get masks for both objects on first frame (optional, you can also propagate the masks through the video directly)
outputs = model(
    inference_session=inference_session,
    frame_idx=ann_frame_idx,
)

# Propagate both objects through video
video_segments = {}
for sam3_tracker_video_output in model.propagate_in_video_iterator(inference_session):
    video_res_masks = processor.post_process_masks(
        [sam3_tracker_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
    )[0]
    video_segments[sam3_tracker_video_output.frame_idx] = {
        obj_id: video_res_masks[i]
        for i, obj_id in enumerate(inference_session.obj_ids)
    }

print(f"Tracked {len(inference_session.obj_ids)} objects through {len(video_segments)} frames")
Tracked 2 objects through 180 frames

Streaming Video Inference
For real-time applications, Sam3TrackerVideo supports processing video frames as they arrive:

# Initialize session for streaming
inference_session = processor.init_video_session(
    inference_device=device,
    dtype=torch.bfloat16,
)

# Process frames one by one
for frame_idx, frame in enumerate(video_frames[:10]):  # Process first 10 frames
    inputs = processor(images=frame, device=device, return_tensors="pt")
...
    if frame_idx == 0:
        # Add point input on first frame
        processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=0,
            obj_ids=1,
            input_points=[[[[210, 350], [250, 220]]]],
            input_labels=[[[1, 1]]],
            original_size=inputs.original_sizes[0], # need to be provided when using streaming video inference
        )
...
    # Process current frame
    sam3_tracker_video_output = model(inference_session=inference_session, frame=inputs.pixel_values[0])
...
    video_res_masks = processor.post_process_masks(
        [sam3_tracker_video_output.pred_masks], original_sizes=inputs.original_sizes, binarize=False
    )[0]
    print(f"Frame {frame_idx}: mask shape {video_res_masks.shape}")

Downloads last month
1,987,426
Safetensors
Model size
0.9B params
Tensor type
F32

Files info

Inference Providers
NEW
Mask Generation
This model isn't deployed by any Inference Provider.
🙋
91
Ask for provider support
Model tree for
facebook/sam3
Adapters
1 model
Finetunes
11 models
Merges
1 model
Quantizations
9 models
Spaces using
facebook/sam3
72
⚡
prithivMLmods/SAM3-Gemma4-CUDA
🐸
prithivMLmods/SAM3-Plus-Qwen3.5
🔥
hysts-gradio-custom-html/sam-prompter-demo
🐠
merve/SAM3-video-segmentation
🏀
prithivMLmods/SAM3-Demo
👁
P3ngLiu/SAM3_VLM-FO1
🎯
webml-community/SAM3-Tracker-WebGPU
🔥
hasanbasbunar/SAM3
+ 64 Spaces
Collection including
facebook/sam3
SAM3
Collection
6 items
•
Updated 19 days ago
•
225
System theme
TOS
Privacy
About
Careers
Models
Datasets
Spaces
Pricing
Docs                                                                         