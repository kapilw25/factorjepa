# automate the pipeline for all the 3D environments [.glb]

task1) get screenshots of isolateral view and topdown view from each [.glb] file stored at @data/Generated_3D_Env/Synthatic/Agriculture & Food Production directory [upload it on huggingface usinf HF_TOKEN in @.env and add it to .gitignore]
- current limitation: @Literature/Prev_work3/images.py is getting failed to capture isolateral view and topdown view
- use some VLM as evaluator to confirm the quality fo views.
task1) generate High quality Instructions using Qwen3 model only [no need of Llama] : refer @Literature/Prev_work3/Qwen3VL.ipynb and @src/m04_pipeline_orchestrator.py
task2) generate 2D paths in TOP-DOWN image using @Literature/Prev_work3/gemini_er.py 



Raw notes
```
Scaling the pipeline
🚨 3) @Literature/Prev_work3/gemini_er.py  >> it draws the path on top-down view image >> automate this pipeline to scale it up
get 3k-5k  (3D)environments (.glb files) from Alam >> 
TASK0: automate to get top-down and isolateral views for each  (3D)environment, 
tried @Literature/Prev_work3/images.py , but failed to screenshot - isolateral and topdown views 
TASK1:  GENERATE 9 instructions per environment >>
TASK2: GENERATE  path on top-down view image using gemini_er.py

3.1) get the paths/ files from different models [gpt5, qwen, gemini, …& some open Source models] >> feed it to Judge
```