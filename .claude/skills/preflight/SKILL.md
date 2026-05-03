---
name: preflight
description: Pipeline code review checklist + automated checks (py_compile, AST, ruff) + call-pattern + SIGSEGV + SAM3/Grounded-SAM/m09/orchestrator regression guards. Catches known failures BEFORE GPU run.
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash
argument-hint: [file-path or module-name]
---

# Pipeline Preflight Checklist
CPU-runnable. Guards (B10+) cite `iter/*/errors_N_fixes.md`. `<file>`=target.
## Part A: Automated
- **A1.** `source venv_walkindia/bin/activate && python3 -m py_compile <file>`
- **A2.** `python3 -c "import ast,sys;s=open('<file>').read();t=ast.parse(s);assert 'main' in {n.name for n in t.body if isinstance(n,ast.FunctionDef)};[sys.exit(f'missing {k}') for k in ('--SANITY','--FULL') if k not in s];print('A2 PASS')"`
- **A3.** `ruff check --select F821,F841,F811 <file>`
## Part B: Manual (B1-B9)
- **B1.** tqdm: `tqdm(desc=, unit=)` or `make_pbar(...)`; resume `initial=len(completed)`.
- **B2.** Checkpoint: periodic saves; `verify_or_skip(...)` BEFORE model load.
- **B3.** Tee: docstring `python -u src/*.py --args 2>&1 | tee logs/<name>.log`.
- **B4.** wandb: `add_wandb_args`+`init_wandb`+`log_metrics`+`finish_wandb`; no-op `run=None`.
- **B5.** GPU loud: `check_gpu()` early; no CPU fallback; no `except: pass`.
- **B7.** Repro: seeds (training); `do_sample=False` (m04).
- **B8.** Footguns: `iter_clips_parallel()`→tuple `(queue, event, thread)` not iterable; `verify_or_skip()`→bool (use in `if`); `get_output_dir()`→`Path`.
  `grep -nE "for\s+\w+\s+in\s+iter_clips_parallel|^\s*verify_or_skip\(" <file>` → MUST be 0

**B6.** Config `.get()` ban (CLAUDE.md §5): AST-track vars from `yaml.safe_load`/`yaml.load`/`json.load`/`load_merged_config`/`load_config`/`load_yaml`/`load_*_config`; flag `.get()` on tracked root.
```bash
source venv_walkindia/bin/activate && python3 << 'EOF'
import ast,sys;s=open('<file>').read();t=ast.parse(s)
L={'safe_load','load_yaml','load_merged_config','load_config','load_train_config','load_pipeline_config','load_model_config'}
jl=lambda c:isinstance(c.func,ast.Attribute) and c.func.attr=='load' and isinstance(c.func.value,ast.Name) and c.func.value.id=='json'
tr=set()
for n in ast.walk(t):
 if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call):
  f=n.value.func;nm=f.attr if isinstance(f,ast.Attribute) else (f.id if isinstance(f,ast.Name) else None)
  if nm in L or jl(n.value): tr|={g.id for g in n.targets if isinstance(g,ast.Name)}
def rt(n):
 while isinstance(n,(ast.Subscript,ast.Attribute)): n=n.value
 return n.id if isinstance(n,ast.Name) else None
b=[n.lineno for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='get' and rt(n.func.value) in tr]
print(f'B6 FAIL:{b}' if b else f'B6 PASS:{len(tr)}');sys.exit(1 if b else 0)
EOF
```
**B9.** SIGSEGV smoke (C-exts bypass try/except). Exit 139=SIGSEGV; timeout=hang.
```bash
source venv_walkindia/bin/activate && timeout 30 python3 -c "
import sys,os,tempfile;sys.path.insert(0,'src')
from utils.data_download import iter_clips_parallel
from utils.video_io import decode_video_bytes
q,s,_=iter_clips_parallel('data/val_1k_local');k,b=q.get(timeout=10)
with tempfile.TemporaryDirectory() as d: t=decode_video_bytes(b,d,k,16)
print(f'B9 PASS:{t.shape}' if t is not None else 'B9 FAIL');s.set();os._exit(0)" 2>&1 | tail -3
```
## Part C: iter8 Regression (B10-B15)
**B10.** CLI sys.path (#3): `src/` importing `from utils.X` with `__main__` needs `sys.path.insert`.
```bash
python3 -c "import re,sys;s=open('<file>').read();bad=bool(re.search(r'^\s*from\s+utils\.',s,re.M)) and '__main__' in s and 'sys.path.insert' not in s;print('B10 FAIL' if bad else 'B10 PASS');sys.exit(1 if bad else 0)"
```
**B11.** SAM3 undeclared deps (#5/#6/#7/#18/#19): AST-walk installed `sam3` top-level unconditional imports vs `requirements*.txt`.
```bash
source venv_walkindia/bin/activate && python3 -c "
import ast,os,re,sys,pathlib;s=open('<file>').read()
if 'from sam3' not in s and 'import sam3' not in s: print('B11 SKIP');sys.exit(0)
import sam3;r=pathlib.Path(sam3.__file__).parent;std=set(sys.stdlib_module_names)|{'typing_extensions','pkg_resources'}
def tops(py):
 try: tr=ast.parse(py.read_text())
 except SyntaxError: return set()
 h=set()
 for n in tr.body:
  if isinstance(n,ast.Import): h|={a.name.split('.')[0] for a in n.names}
  elif isinstance(n,ast.ImportFrom) and n.level==0 and n.module: h.add(n.module.split('.')[0])
 return h
OFF={'train','agent','tests','test','benchmarks','examples'}
ex={tp for py in r.rglob('*.py') if not any(x in OFF for x in py.relative_to(r).parts) for tp in tops(py) if tp not in std and tp not in {'sam3','__future__'}}
dc=set()
for rf in ['requirements_gpu.txt','requirements.txt']:
 if os.path.exists(rf):
  for l in open(rf):
   l=l.split('#')[0].strip()
   if l and not l.startswith('-'): dc.add(re.split(r'[<>=!\s]',l)[0].lower().replace('-','_'))
OK={'torch','torchvision','numpy','PIL','matplotlib','tqdm','pandas','cv2','huggingface_hub','regex','scipy','skimage','sklearn','yaml','requests','psutil','triton'}
m=sorted({x for x in ex if x.lower().replace('-','_') not in dc and x not in OK})
print(f'B11 FAIL:{m}' if m else f'B11 PASS:{len(ex)}');sys.exit(1 if m else 0)"
```
**B12.** SAM3 FA3 (#9): `build_sam3_predictor(..., use_fa3=False)`.
```bash
python3 -c "import ast,sys;t=ast.parse(open('<file>').read());b=[];n=0
for x in ast.walk(t):
 if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='build_sam3_predictor':
  n+=1
  if not any(k.arg=='use_fa3' and isinstance(k.value,ast.Constant) and k.value.value is False for k in x.keywords): b.append(x.lineno)
print(f'B12 FAIL:{b}' if b else (f'B12 PASS:{n}' if n else 'B12 SKIP'));sys.exit(1 if b else 0)"
```
**B13.** SAM3 async-exit (#14/#16): `os._exit(0)` at main() end + `__main__ try/except os._exit(1)`.
```bash
python3 -c "import re,sys;s=open('<file>').read()
if 'import sam3' not in s and 'from sam3' not in s: print('B13 SKIP');sys.exit(0)
if 'os._exit(0)' not in s: sys.exit('B13 FAIL: no os._exit(0)')
if not re.search(r'if\s+__name__\s*==\s*[\"\\']__main__[\"\\']:\s*\n\s*try:.*?os\._exit\(1\)',s,re.DOTALL): sys.exit('B13 FAIL: __main__ try/except missing')
print('B13 PASS')"
```
**B14.** SAM3 unified entry (#8): no `build_sam3_multiplex_video_predictor`.
```bash
python3 -c "import re,sys;b=bool(re.search(r'build_sam3_multiplex_video_predictor\s*\(',open('<file>').read()));print('B14 FAIL' if b else 'B14 PASS');sys.exit(1 if b else 0)"
```
**B15.** torchcodec disabled (#10): `video_io._USE_TORCHCODEC=False`.
```bash
python3 -c "import sys;sys.path.insert(0,'src');from utils.video_io import _USE_TORCHCODEC;print('B15 FAIL' if _USE_TORCHCODEC else 'B15 PASS');sys.exit(1 if _USE_TORCHCODEC else 0)"
```
## Part D: transformers 5.x (B16-B20)
**B16.** `torch_dtype=` deprecated (#37):
```bash
source venv_walkindia/bin/activate && python3 -c "import ast,sys;t=ast.parse(open('<file>').read());b=[n.lineno for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='from_pretrained' for k in n.keywords if k.arg=='torch_dtype'];print(f'B16 FAIL:{b}' if b else 'B16 PASS');sys.exit(1 if b else 0)"
```
**B17.** DINO fp32-only (#37):
```bash
source venv_walkindia/bin/activate && python3 -c "
import ast,sys;t=ast.parse(open('<file>').read());b=[]
for n in ast.walk(t):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='from_pretrained':
  r=n.func.value
  if isinstance(r,ast.Name) and r.id=='AutoModelForZeroShotObjectDetection':
   for k in n.keywords:
    if k.arg in ('dtype','torch_dtype') and isinstance(k.value,ast.Attribute) and k.value.attr in ('float16','half','bfloat16'): b.append((n.lineno,k.value.attr))
print(f'B17 FAIL:{b}' if b else 'B17 PASS');sys.exit(1 if b else 0)"
```
**B18.** Sam3TrackerVideoProcessor box depth=3 (#38):
```bash
python3 -c "import re,sys;b=[i+1 for i,l in enumerate(open('<file>').read().split(chr(10))) if re.search(r'\[\s*\[\s*\[\s*b\s*\]\s+for\s+b\s+in\s+',l)];print(f'B18 FAIL:{b}' if b else 'B18 PASS');sys.exit(1 if b else 0)"
```
**B19.** Session-reset on session (#39):
```bash
python3 -c "import re,sys;b=[(i,m.group(1),m.group(2)) for i,l in enumerate(open('<file>').read().split(chr(10)),1) for m in [re.search(r'(\w*processor\w*)\.(reset_inference_session|reset_tracking_data|clear_all|remove_point_inputs|remove_mask_inputs)\s*\(',l)] if m];print(f'B19 FAIL:{b}' if b else 'B19 PASS');sys.exit(1 if b else 0)"
```
**B20.** Sam3TrackerVideoSegmentationOutput attrs (#40): only `object_ids`/`pred_masks`/`object_score_logits`/`frame_idx`.
```bash
source venv_walkindia/bin/activate && python3 << 'PY'
import re,sys;b=[(i,l.strip()) for i,l in enumerate(open('<file>').read().split('\n'),1) if re.search(r'\boutput\.(iou_scores|mask_logits|out_obj_ids|out_binary_masks|out_probs)\b',l) or re.search(r"""getattr\(output,\s*['"]iou_scores['"]""",l)]
print(f'B20 FAIL:{b}' if b else 'B20 PASS');sys.exit(1 if b else 0)
PY
```
## Part E: Grounded-SAM (B21-B25)
**B21.** `load_dotenv()` before sam3/transformers (#21):
```bash
python3 -c "import re,sys;s=open('<file>').read()
if not re.search(r'(^|\n)\s*(import sam3\b|from sam3\b|from transformers import.*Auto[A-Z]\w*)',s): print('B21 SKIP');sys.exit(0)
ls=s.split(chr(10))
fi=next((i for i,l in enumerate(ls) if re.search(r'(import sam3\b|from sam3\b|from transformers import.*Auto[A-Z]\w*)',l)),None)
fl=next((i for i,l in enumerate(ls) if 'load_dotenv(' in l and not l.lstrip().startswith('#')),None)
bad=fl is None or fl>fi;print(f'B21 FAIL:line{fi+1}' if bad else 'B21 PASS');sys.exit(1 if bad else 0)"
```
**B22.** DINO post-process rename (#24): `box_threshold=` → `threshold=`.
```bash
python3 -c "import ast,sys;t=ast.parse(open('<file>').read());bk=[];bl=[]
for n in ast.walk(t):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='post_process_grounded_object_detection': bk+=[n.lineno for k in n.keywords if k.arg=='box_threshold']
 if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and n.slice.value=='labels' and isinstance(n.value,ast.Name) and 'result' in n.value.id.lower(): bl.append(n.lineno)
if bk: sys.exit(f'B22 FAIL:{bk}')
print(f'B22 WARN:{bl}' if bl else 'B22 PASS')"
```
**B23.** Raw SAM3 text+boxes (#27): `add_prompt(bounding_boxes=...)` needs `text=`.
```bash
python3 -c "import ast,sys;t=ast.parse(open('<file>').read());b=[]
for n in ast.walk(t):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_prompt':
  kw={k.arg for k in n.keywords}
  if 'bounding_boxes' in kw and 'text' not in kw: b.append(n.lineno)
print(f'B23 FAIL:{b}' if b else 'B23 PASS');sys.exit(1 if b else 0)"
```
**B24.** Propagate empty + 'No points' (#30/#31):
```bash
python3 -c "import re,sys;s=open('<file>').read()
if 'propagate_in_video' not in s: print('B24 SKIP');sys.exit(0)
e=bool(re.search(r'(n_frame0_objs|len\(.*out_obj_ids.*\))\s*==\s*0',s));r=bool(re.search(r'except\s+RuntimeError',s)) and 'No points are provided' in s
f=(['empty'] if not e else [])+(['no-points'] if not r else [])
print(f'B24 FAIL:{f}' if f else 'B24 PASS');sys.exit(1 if f else 0)"
```
**B25.** Box clamp before xywh normalize (#28):
```bash
python3 -c "import re,sys;s=open('<file>').read()
if 'bounding_boxes=' not in s: print('B25 SKIP');sys.exit(0)
n=bool(re.search(r'/\s*[WH]\b|/\s*float\([WH]\)',s));c=bool(re.search(r'max\s*\(\s*0[.,]?\s*,\s*min\s*\(\s*[WH]',s)) or 'clip(' in s or '.clamp(' in s
hf='add_inputs_to_inference_session' in s or 'Sam3TrackerVideoProcessor' in s;bad=n and not c and not hf
print('B25 FAIL' if bad else 'B25 PASS');sys.exit(1 if bad else 0)"
```
## Part F: raw vs HF SAM3 (B26-B27)
**B26.** `max_frame_num_to_track=` banned on raw sam3 (#33/#35):
```bash
python3 -c "import re,sys;s=open('<file>').read()
raw=bool(re.search(r'(build_sam3_predictor|handle_stream_request)',s));hf='propagate_in_video_iterator' in s
if not raw: print('B26 SKIP');sys.exit(0)
b=[i+1 for i,l in enumerate(s.split(chr(10))) if 'max_frame_num_to_track=' in l and not l.lstrip().startswith('#')]
bad=b and not hf;print(f'B26 FAIL:{b}' if bad else 'B26 PASS');sys.exit(1 if bad else 0)"
```
**B27.** `Sam3VideoProcessor` kwargs/keys (#41): `text=` not `prompts=`; `post['masks']` not `pred_masks`.
```bash
python3 -c "import re,sys;s=open('<file>').read()
if 'Sam3VideoProcessor' not in s and 'Sam3VideoModel' not in s: print('B27 SKIP');sys.exit(0)
bk=[i+1 for i,l in enumerate(s.split(chr(10))) if re.search(r'add_text_prompt\s*\([^)]*prompts\s*=',l)]
bp=[i+1 for i,l in enumerate(s.split(chr(10))) if re.search(r'\bpost\[[\\x22\\x27]pred_masks[\\x22\\x27]\]',l)]
f=(f'prompts={bk}' if bk else '')+(f'pred={bp}' if bp else '');print(f'B27 FAIL:{f}' if f else 'B27 PASS');sys.exit(1 if f else 0)"
```
## Part G: V-JEPA + config (B28-B31)
**B28.** m05 fp16 hardcode (#45):
```bash
python3 -c "import os,re,sys;t='<file>'
if os.path.basename(t)!='m05_vjepa_embed.py': print('B28 SKIP');sys.exit(0)
s=open(t).read();b=[i for i,l in enumerate(s.split(chr(10)),1) if re.search(r'dtype\s*=\s*torch\.float16\b',l) and any(c in s[max(0,s.find(l)-500):s.find(l)+500] for c in ['get_batch_embeddings','autocast','warmup'])]
print(f'B28 FAIL:{b}' if b else 'B28 PASS');sys.exit(1 if b else 0)"
```
**B29.** `_ensure_loaded_2_1` finally restores all (#50):
```bash
python3 -c "import os,re,sys;t='<file>'
if os.path.basename(t)!='vjepa2_imports.py': print('B29 SKIP');sys.exit(0)
for fb in re.findall(r'finally:\s*\n(.*?)(?=\n\S)',open(t).read(),re.S):
 if re.search(r'for\s+\w+\s*(,\s*\w+)?\s+in\s+saved_modules',fb) or 'sys.modules.update' in fb: print('B29 PASS');sys.exit(0)
sys.exit('B29 FAIL')"
```
**B30.** `cfg['data'][patch/crop/tubelet]` schema (#51):
```bash
python3 -c "import re,sys;b=[i+1 for i,l in enumerate(open('<file>').read().split(chr(10)),1) if re.search(r'(data_cfg|cfg\[[\\x22\\x27]data[\\x22\\x27]\])\s*\[\s*[\\x22\\x27](crop_size|patch_size|tubelet_size)[\\x22\\x27]\s*\]',l)];print(f'B30 FAIL:{b}' if b else 'B30 PASS');sys.exit(1 if b else 0)"
```
**B31.** `max_epochs[mode_key]` double-subscript (#52):
```bash
python3 << 'EOF'
import ast,sys;t=ast.parse(open('<file>').read())
sk=lambda n:n.slice.value if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) else None
mk=lambda n:isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Name) and n.slice.id=='mode_key' and sk(n.value)=='max_epochs'
h=[n for n in ast.walk(t) if mk(n)];r={id(n.value) for n in ast.walk(t) if isinstance(n,ast.Assign) and len(n.targets)==1 and sk(n.targets[0])=='max_epochs' and mk(n.value)}
b=sorted({x.lineno for x in h if id(x) not in r});print(f'B31 FAIL:{b}' if b else 'B31 PASS');sys.exit(1 if b else 0)
EOF
```
## Part H: m09 split (B32-B36)
**B32.** `expandable_segments:True` in m09* (#53):
```bash
python3 -c "import os,re,sys;t='<file>'
if not re.match(r'm09[abc]_.*\.py$',os.path.basename(t)): print('B32 SKIP');sys.exit(0)
ok=bool(re.search(r'os\.environ\.setdefault\([\\x22\\x27]PYTORCH_CUDA_ALLOC_CONF[\\x22\\x27]\s*,\s*[\\x22\\x27]expandable_segments:True',open(t).read()))
print('B32 PASS' if ok else 'B32 FAIL');sys.exit(0 if ok else 1)"
```
**B33.** m09c loss pre-init (#54): `jepa_val`/`masked_val`/`context_val`=0.0.
```bash
python3 -c "import os,re,sys;t='<file>'
if os.path.basename(t)!='m09c_surgery.py': print('B33 SKIP');sys.exit(0)
s=open(t).read();b=[v for v in ('jepa_val','masked_val','context_val') if not re.search(rf'\b{v}\s*=\s*0\.0\b',s)]
print(f'B33 FAIL:{b}' if b else 'B33 PASS');sys.exit(1 if b else 0)"
```
**B34.** m09b/c retry + 0-step raise (#55):
```bash
python3 -c "import os,re,sys;t='<file>';bn=os.path.basename(t)
if not re.match(r'm09[bc]_.*\.py$',bn): print('B34 SKIP');sys.exit(0)
s=open(t).read();f=[]
if not re.search(r'while\s+not\s+step_succeeded',s): f.append('no-retry')
if not re.search(r'train_sizer\.size\s*<=\s*train_sizer\.min_size',s): f.append('no-min-raise')
if 'm09c' in bn and not re.search(r'if\s+global_step\s*==\s*0\s*:\s*\n\s*raise\s+RuntimeError',s,re.M): f.append('no-0step-raise')
print(f'B34 FAIL:{f}' if f else 'B34 PASS');sys.exit(1 if f else 0)"
```
**B35.** AdaptiveBatchSizer + yaml memory_cap (#47):
```bash
python3 -c "import os,re,sys;t='<file>'
N={'m04_vlm_tag.py','m04d_motion_features.py','m05_vjepa_embed.py','m05b_baselines.py','m05c_true_overlap.py','m09a_pretrain.py','m09b_explora.py','m09c_surgery.py'}
if os.path.basename(t) not in N: print('B35 SKIP');sys.exit(0)
s=open(t).read();f=[]
if 'AdaptiveBatchSizer' not in s and '_train_step_grad_accum' not in s: f.append('sizer')
if not (re.search(r'memory_cap\s*=\s*[\w\.\[\]\\x22\\x27]*gpu_memory_target',s) or re.search(r'memory_cap\s*=.*yaml',s,re.I)): f.append('memory_cap')
print(f'B35 FAIL:{f}' if f else 'B35 PASS');sys.exit(1 if f else 0)"
```
**B36.** `transformers>=5.x` pinned (#36):
```bash
source venv_walkindia/bin/activate && python3 -c "import re,sys,os
if not os.path.exists('requirements_gpu.txt'): print('B36 SKIP');sys.exit(0)
m=re.search(r'^\s*transformers\s*([<>=~!]+)\s*([\d.]+)',open('requirements_gpu.txt').read(),re.M)
if not m: sys.exit('B36 FAIL:unpinned')
if int(m.group(2).split('.')[0])<5: sys.exit(f'B36 FAIL:{m.group(1)}{m.group(2)}')
print(f'B36 PASS:{m.group(1)}{m.group(2)}')"
```
## Part I: durability + yaml + env (B37-B42)
**B37.** vjepa2 RoPE patch durability (#44/#59):
```bash
python3 -c "import os
mf='deps/vjepa2/app/vjepa_2_1/models/utils/modules.py'
p=os.path.exists(mf) and 'q = q.to(v.dtype)' in open(mf).read() and 'k = k.to(v.dtype)' in open(mf).read()
su=os.path.exists('setup_env_uv.sh') and 'q = q.to(v.dtype)' in open('setup_env_uv.sh').read() and 'vjepa2 RoPE dtype patch' in open('setup_env_uv.sh').read()
if not p and not su: raise SystemExit('B37 FAIL:missing')
print('B37 WARN:disk-only' if not su else 'B37 PASS')"
```
**B38.** `ch11_surgery.yaml` max_epochs (#60): poc≥10, full≥poc.
```bash
source venv_walkindia/bin/activate && python3 -c "import sys,os,yaml;y='configs/train/ch11_surgery.yaml'
if not os.path.exists(y): print('B38 SKIP');sys.exit(0)
me=yaml.safe_load(open(y)).get('optimization',{}).get('max_epochs',{})
m=[k for k in ('sanity','poc','full') if k not in me]
if m: sys.exit(f'B38 FAIL:{m}')
if me['poc']<10 or me['full']<me['poc']: sys.exit(f'B38 FAIL:{me}')
print(f'B38 PASS:{me}')"
```
**B39.** `ch11_surgery.yaml` warmup_pct (#61):
```bash
source venv_walkindia/bin/activate && python3 -c "import sys,os,yaml;y='configs/train/ch11_surgery.yaml'
if not os.path.exists(y): print('B39 SKIP');sys.exit(0)
sr=yaml.safe_load(open(y)).get('surgery',{});p=sr.get('warmup_pct');f=[]
if p is None: f.append('miss')
elif not (0.01<=p<=0.5): f.append(f'={p}')
for st in sr.get('stages',[]):
 if 'warmup_steps' in st: f.append(st.get('name'))
print(f'B39 FAIL:{f}' if f else f'B39 PASS:{p}');sys.exit(1 if f else 0)"
```
**B40.** m05 `_resolve_model` (#42):
```bash
python3 -c "import os,re,sys;t='<file>'
if os.path.basename(t)!='m05_vjepa_embed.py': print('B40 SKIP');sys.exit(0)
s=open(t).read();f=[]
if 'def _resolve_model(' not in s: f.append('helper')
if re.search(r'args\.model\s*=\s*VJEPA_CHECKPOINT_PATH',s): f.append('reassign')
b=[i+1 for i,l in enumerate(s.split(chr(10)),1) if re.search(r'Path\(\s*args\.model\s*\)',l) and 'resolve_model' not in l]
if b: f.append(f'bare@{b}')
print(f'B40 FAIL:{f}' if f else 'B40 PASS');sys.exit(1 if f else 0)"
```
**B41.** `.env` multi-word quoted (#1):
```bash
python3 -c "import os,re,sys
if not os.path.exists('.env'): print('B41 SKIP');sys.exit(0)
b=[]
for i,l in enumerate(open('.env'),1):
 s=l.strip()
 if not s or s.startswith('#') or '=' not in s: continue
 if s.startswith('export '): s=s[7:]
 k,_,v=s.partition('=')
 if not v or v[0] in ('\"',\"'\"): continue
 if re.search(r'\s',v): b.append((i,k.strip()))
print(f'B41 FAIL:{b}' if b else 'B41 PASS');sys.exit(1 if b else 0)"
```
**B42.** `.gitignore` excludes `logs/` (#4):
```bash
python3 -c "import os,sys
if not os.path.exists('.gitignore'): sys.exit('B42 FAIL')
ok=any(p in open('.gitignore').read() for p in ['logs/','logs/*','logs/*.log'])
print('B42 PASS' if ok else 'B42 FAIL');sys.exit(0 if ok else 1)"
```
## Part J: agnostic silent-default (B43-B47)
**B43.** Silent `except` (`.py`): `pass`/`continue`/`break`/`return None` with no `raise`.
```bash
python3 -c "
import ast,sys;t='<file>'
if not t.endswith('.py'): print('B43 SKIP');sys.exit(0)
tr=ast.parse(open(t).read())
b=[(n.lineno,ast.unparse(n.type) if n.type else 'bare') for n in ast.walk(tr) if isinstance(n,ast.ExceptHandler) and n.body and all(isinstance(s,(ast.Pass,ast.Continue,ast.Break)) or (isinstance(s,ast.Return) and (s.value is None or (isinstance(s.value,ast.Constant) and s.value.value is None))) for s in n.body) and not any(isinstance(sub,ast.Raise) for s in n.body for sub in ast.walk(s))]
print(f'B43 FAIL:{b}' if b else 'B43 PASS');sys.exit(1 if b else 0)"
```
**B44.** Shell silent-defaults (`.sh`/`.env`/rc): `${VAR:-}`/`${VAR:=}`/`|| true`/`|| :`/`|| continue`. `${VAR:?msg}` allowed.
```bash
python3 -c "
import re,sys;t='<file>'
if not t.endswith(('.sh','.env','.bashrc','.zshrc','.bash_profile')): print('B44 SKIP');sys.exit(0)
f=[]
for i,l in enumerate(open(t),1):
 s=l.lstrip()
 if s.startswith('#') or not s.strip(): continue
 for m in re.finditer(r'\$\{(\w+)(?::?[-=])([^}?][^}]*)\}',l):
  if f'\${{{m.group(1)}:?' not in l: f.append((i,'def'))
 if re.search(r'\|\|\s*(true\b|:\s*(\||\$)|:\$|continue\b)',l): f.append((i,'pipe'))
if f: print(f'B44 FAIL:{f[:10]}');sys.exit(1)
print('B44 PASS')"
```
**B45.** YAML null values (`.yaml`): WARN only — each null → `None` → silent skip risk.
```bash
source venv_walkindia/bin/activate && python3 -c "
import sys,yaml;t='<file>'
if not t.endswith(('.yaml','.yml')): print('B45 SKIP');sys.exit(0)
try: d=yaml.safe_load(open(t))
except yaml.YAMLError as e: sys.exit(f'B45 FAIL:{e}')
def w(n,p=''):
 r=[]
 if isinstance(n,dict):
  for k,v in n.items(): pp=f'{p}.{k}' if p else str(k);r.append(pp) if v is None else r.extend(w(v,pp))
 elif isinstance(n,list):
  for i,it in enumerate(n): r+=w(it,f'{p}[{i}]')
 return r
ns=w(d) if d else [];print(f'B45 WARN:{len(ns)}' if ns else 'B45 PASS')"
```
**B46.** `.get(k,<non-None>)` sweep (`.py`): None default OK; non-None = silent. Whitelist: `os`/`kwargs`/`request`/`headers`/`environ`/`response`.
```bash
python3 -c "
import ast,sys;t='<file>'
if not t.endswith('.py'): print('B46 SKIP');sys.exit(0)
tr=ast.parse(open(t).read());OK={'os','kwargs','request','headers','environ','response'}
def rt(n):
 while isinstance(n,(ast.Subscript,ast.Attribute)): n=n.value
 return n.id if isinstance(n,ast.Name) else None
b=[(n.lineno,rt(n.func.value) or '?') for n in ast.walk(tr) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='get' and len(n.args)>=2 and not (isinstance(n.args[1],ast.Constant) and n.args[1].value is None) and rt(n.func.value) not in OK]
if b: print(f'B46 FAIL:{b[:15]}');sys.exit(1)
print('B46 PASS')"
```
**B47.** Parallel-index step-count dispatch (`.py`): function with ≥2 `build_*_index(es)`/`*_dataset`/`*_loader` assignments — any bare `total_clips = len(X)` / `steps_per_epoch = len(X)` NOT inside `if-elif-else` or ternary is a silent-collapse bug.
```bash
python3 -c "
import re,sys;s=open('<file>').read()
if not '<file>'.endswith('.py'): print('B47 SKIP');sys.exit(0)
LHS='(total_clips|n_total|total_steps|steps_per_epoch|n_train|n_steps|dataset_size|num_train|n_train_clips|total_train)'
bad=[]
for fn in re.findall(r'def\s+\w+\([^)]*\):(.*?)(?=\n(?:def|class|\S)|\Z)',s,re.DOTALL):
 if len(re.findall(r'=\s*(?:build|make)_\w+_(?:ind(?:ex|ices)|dataset|loader)\(',fn))<2: continue
 for m in re.finditer(rf'(?:^|\n)\s+({LHS})\s*=\s*len\([^)]+\)(?!\s+(?:if|else))',fn):
  bad.append(m.group(1))
print(f'B47 FAIL:{bad}' if bad else 'B47 PASS');sys.exit(1 if bad else 0)"
```
## Part K: orchestrator (B48-B49, `scripts/*.sh`)
**B48.** `&&` chain between `python -u src/m*.py` (iter10 2026-04-22): use `;`.
```bash
source venv_walkindia/bin/activate && python3 -c "import re,sys
fp=sys.argv[1] if len(sys.argv)>1 else 'scripts/legacy2/run_paired_eval_10k.sh';s=open(fp).read()
p=re.compile(r'python\s+-u\s+src/m[0-9]\w*\.py.*?&&\s*\\\\?\s*\n.*?python\s+-u\s+src/m[0-9]\w*\.py',re.DOTALL)
if p.search(s): sys.exit(f'B48 FAIL:{len(p.findall(s))} in {fp}')
print(f'B48 PASS:{fp}')" scripts/legacy2/run_paired_eval_10k.sh
```
**B49.** `outputs_versioned/` writer↔reader contract (iter10 2026-04-22):
```bash
source venv_walkindia/bin/activate && python3 -c "import re
from pathlib import Path
w,r=[],[]
for fp in sorted(Path('scripts').glob('*.sh')):
 s=fp.read_text()
 for m in re.finditer(r'(outputs/(?:full|poc|sanity)/[a-z0-9_]+/)([a-zA-Z0-9_./*{}-]+)',s): w.append(m.group(1)+m.group(2))
 for m in re.finditer(r'(outputs_versioned/[a-zA-Z0-9_]+/)([a-zA-Z0-9_./-]+)',s): r.append((fp.name,m.group(1)+m.group(2)))
o=[(n,p) for n,p in r if not any(p.split('/',2)[-1] in wp for wp in w)]
if o: print(f'B49 FAIL:{len(o)}');[print(f' {n}:{p}') for n,p in o[:5]]
else: print('B49 PASS')"
```
## Part L: silent selector-drop + brittle-queue (B50-B51)
**B50.** Silent drop of yaml-requested selector keys (iter10 v15c): loop over a `mixture`/`policy`/`weights`/`recipe`/`schedule`/`ratios` dict with `if <lookup>[key]: append(...)` silently skips keys whose data is absent → post-loop renorm runs a different distribution than yaml. Post-loop `if not accumulator: raise` only fires on TOTAL emptiness. Fix: per-key `raise` inside loop OR assert before `append`.
```bash
python3 -c "
import ast,re,sys;t='<file>'
if not t.endswith('.py'): print('B50 SKIP');sys.exit(0)
tr=ast.parse(open(t).read());MIX=re.compile(r'(mixture|policy|recipe|schedule|weights|blend|ratios?)\$',re.I);b=[]
for stmt in ast.walk(tr):
 if not (isinstance(stmt,ast.For) and isinstance(stmt.iter,ast.Call) and isinstance(stmt.iter.func,ast.Attribute) and stmt.iter.func.attr=='items'): continue
 try: nm=ast.unparse(stmt.iter.func.value).split('.')[-1]
 except Exception: continue
 if not MIX.search(nm): continue
 for ch in ast.walk(stmt):
  if not (isinstance(ch,ast.If) and isinstance(ch.test,ast.Subscript)): continue
  ap=any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='append' for bd in ch.body for n in ast.walk(bd))
  er=any(isinstance(n,ast.Raise) for bd in ch.orelse for n in ast.walk(bd))
  if ap and not er: b.append((ch.lineno,nm))
if b: print(f'B50 FAIL:{b}; add else-raise');sys.exit(1)
print('B50 PASS')"
```
**B51.** Brittle queue under strict mode (iter10 v3 2026-04-22 `paired_eval_frozen_m05` cascade): `.sh` with `set -e` + `trap … ERR` + ≥2 long `python -u *.py` calls where none has a per-call guard (`|| {…}`, `set +e`-window, or `if ! cmd`) — any single non-zero exit nukes the whole queue (here: m05 partial-completion FATAL killed 16h paired-eval). Applies to `.sh` only.
```bash
python3 -c "
import re,sys;t='<file>'
if not t.endswith('.sh'): print('B51 SKIP');sys.exit(0)
s=open(t).read()
strict=bool(re.search(r'^\s*set\s+-\w*e\w*',s,re.M));trap=bool(re.search(r'^\s*trap\b[^\n]*\bERR\b',s,re.M))
calls=[m.start() for m in re.finditer(r'python\s+-u\s+[\w/.-]+\.py',s)]
if not (strict and trap and len(calls)>=2): print('B51 SKIP');sys.exit(0)
g=0
for pos in calls:
 seg=s[pos:s.find('\n',pos) if s.find('\n',pos)>0 else len(s)]
 back=s[max(0,pos-400):pos]
 fwd=s[max(0,pos-30):pos].rstrip()
 if '||' in seg or (re.search(r'set\s+\+e',back) and 'set -e' not in back.split('set +e')[-1]) or re.search(r'if\s+!\s*\$',fwd): g+=1
if g==0: print(f'B51 FAIL: {len(calls)} python -u under set-e+trap-ERR with no per-call guard. Add `|| {{ log; continue; }}` or `set +e` window.');sys.exit(1)
print(f'B51 PASS: {g}/{len(calls)} guarded')"
```

## Part M: iter12 regressions (B52-B61) — catch BEFORE GPU spend

> Sources: `iter/iter12_multitask_LOSS/errors_N_fixes.md` #62-#81. Each guard catches a specific bug class that wasted GPU-h or invalidated a result. Run on the file glob in the rightmost column.

**B52.** `data/subset_*.json ∩ data/val_1k.json` test leak (iter12 #69 — 41 % overlap silently invalidated paired-eval). `.json` files only.
```bash
python3 -c "import json,sys,glob;a=set(json.load(open('data/val_1k.json'))['clip_keys']);bad=[(p,len(a&set(json.load(open(p))['clip_keys']))) for p in glob.glob('data/subset_*.json') if 'with_leak' not in p and a&set(json.load(open(p))['clip_keys'])];print(f'B52 FAIL:{bad}' if bad else 'B52 PASS');sys.exit(1 if bad else 0)"
```

**B53.** Shell-level blanket `rm -rf outputs/full/m05*` nukes hidden `.m05_checkpoint*.npz` (iter12 #74 — multi-hour GPU loss). `.sh` only.
```bash
python3 -c "import re,sys,os;t='<file>'
if not t.endswith('.sh'): print('B53 SKIP');sys.exit(0)
s=open(t).read();b=[i for i,l in enumerate(s.split(chr(10)),1) if re.search(r'\brm\s+-rf?\s+\S*outputs/\S*m05[_\w/]*\s*$',l) and '.m05_checkpoint' not in s[max(0,sum(len(x)+1 for x in s.split(chr(10))[:i-1])-200):sum(len(x)+1 for x in s.split(chr(10))[:i])+400]]
print(f'B53 FAIL:{b}' if b else 'B53 PASS');sys.exit(1 if b else 0)"
```

**B54.** `from utils.X import Y` *inside a function* shadowing a top-level import → `UnboundLocalError` if any earlier line in the same function references `Y` (iter12 #68). `src/m*.py`, `src/utils/*.py`.
```bash
source venv_walkindia/bin/activate && python3 << 'PY'
import ast,sys;t=ast.parse(open('<file>').read())
top={a.name for n in t.body if isinstance(n,ast.ImportFrom) for a in n.names}
b=[]
for fn in [n for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]:
    locals_imp={a.name:imp.lineno for imp in ast.walk(fn) if isinstance(imp,ast.ImportFrom) and imp is not fn for a in imp.names if a.name in top}
    for nm,lno in locals_imp.items():
        uses=[u.lineno for u in ast.walk(fn) if isinstance(u,ast.Name) and u.id==nm and u.lineno<lno]
        if uses: b.append((fn.name,nm,uses[0],lno))
print(f'B54 FAIL:{b}' if b else 'B54 PASS');sys.exit(1 if b else 0)
PY
```

**B55.** YAML dead-cap field — `n_clips`/`clip_limit`/`poc_simplified`/`sanity_clips` silently downscope `--subset` (iter12 #62). `.yaml` only.
```bash
source venv_walkindia/bin/activate && python3 -c "
import yaml,sys;y='<file>'
if not y.endswith(('.yaml','.yml')): print('B55 SKIP');sys.exit(0)
d=yaml.safe_load(open(y)) or {};b=[]
def w(n,p=''):
    if isinstance(n,dict):
        for k,v in n.items():
            pp=f'{p}.{k}' if p else str(k)
            if k in ('n_clips','clip_limit','poc_simplified','sanity_clips'): b.append(pp)
            else: w(v,pp)
w(d);print(f'B55 FAIL:dead-cap {b}' if b else 'B55 PASS');sys.exit(1 if b else 0)"
```

**B56.** `tempfile.mkdtemp()` outside per-stage/per-epoch loop OR inside loop without matching `shutil.rmtree` (iter12 #80 — tmpfs ENOENT after ~2M write/unlink). `src/utils/training.py` + `src/m*.py`.
```bash
python3 << 'PY'
import ast,sys;t='<file>'
if not t.endswith('.py'): print('B56 SKIP');sys.exit(0)
tr=ast.parse(open(t).read());b=[]
for fn in [n for n in ast.walk(tr) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and any(x in n.name.lower() for x in ('producer','stream','decode'))]:
    mk=[c for c in ast.walk(fn) if isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='mkdtemp']
    if not mk: continue
    loops=[l for l in ast.walk(fn) if isinstance(l,(ast.For,ast.While))]
    for c in mk:
        in_loop=any(c.lineno>=l.lineno and c.lineno<=getattr(l,'end_lineno',c.lineno) for l in loops)
        has_rmtree=any('rmtree' in (ast.unparse(s) if hasattr(ast,'unparse') else '') for s in ast.walk(fn))
        if not in_loop or not has_rmtree: b.append((fn.name,c.lineno))
print(f'B56 FAIL:{b}' if b else 'B56 PASS');sys.exit(1 if b else 0)
PY
```

**B57.** `m06_faiss_metrics.py` hardcodes `data/val_1k_local/tags.json` default → eval_10k workflow reads wrong tags (iter12 #77).
```bash
python3 -c "import os,re,sys;t='<file>'
if os.path.basename(t)!='m06_faiss_metrics.py': print('B57 SKIP');sys.exit(0)
s=open(t).read()
bad=bool(re.search(r'[\\x22\\x27]data/val_1k_local/tags\.json[\\x22\\x27]',s)) and not re.search(r'Path\(args\.subset\)\.parent.*tags',s)
print('B57 FAIL: hardcoded val_1k tags default; auto-derive from --subset instead' if bad else 'B57 PASS');sys.exit(1 if bad else 0)"
```

**B58.** `m05_vjepa_embed.py` checkpoint filename keyed only by encoder-name → cross-variant collision (iter12 #75). Must include model-content fingerprint.
```bash
python3 -c "import os,re,sys;t='<file>'
if os.path.basename(t)!='m05_vjepa_embed.py': print('B58 SKIP');sys.exit(0)
s=open(t).read();has_fp=bool(re.search(r'_checkpoint_fingerprint|_compute_m05_fp',s))
bad=bool(re.search(r'\.m05_checkpoint_\{embed_suffix\}\.npz',s)) and not has_fp
print('B58 FAIL: m05 ckpt path lacks fingerprint — variants will collide' if bad else 'B58 PASS');sys.exit(1 if bad else 0)"
```

**B59.** Mode-gated yaml dict (`{sanity:..., poc:..., full:...}`) declared but never flattened by reader (iter12 #57/#67) → `TypeError: 'dict' is not subscriptable`. `.yaml` only; cross-checks `src/m09c_surgery.py`.
```bash
source venv_walkindia/bin/activate && python3 -c "
import yaml,re,sys,pathlib;y='<file>'
if not y.endswith(('.yaml','.yml')): print('B59 SKIP');sys.exit(0)
d=yaml.safe_load(open(y)) or {}
def gated(n): return isinstance(n,dict) and set(n.keys())>={'sanity','poc','full'} and all(not isinstance(v,(dict,list)) for v in n.values())
gk=[]
def w(n,p=''):
    if isinstance(n,dict):
        for k,v in n.items():
            pp=f'{p}.{k}' if p else str(k)
            if gated(v): gk.append(pp)
            else: w(v,pp)
w(d)
src=pathlib.Path('src/m09c_surgery.py')
if not src.exists(): print(f'B59 PASS:{len(gk)} gated (m09c missing — skip cross-check)');sys.exit(0)
py=src.read_text();missed=[k for k in gk if k.split('.')[-1] not in py]
print(f'B59 WARN:gated keys not flattened in m09c: {missed}' if missed else f'B59 PASS:{len(gk)} gated, all flattened');sys.exit(0)"
```

**B60.** `m11_factor_datasets.py` under `--streaming` flags `has_D_L=True` without disk-exists guard, OR pbar `total=len(segments)` ignores `--subset` count (iter12 #67/#70).
```bash
python3 -c "import os,re,sys;t='<file>'
if os.path.basename(t)!='m11_factor_datasets.py': print('B60 SKIP');sys.exit(0)
s=open(t).read();f=[]
if '--streaming' in s or 'streaming' in s:
    if not re.search(r'\.exists\(\)|_has_materialized_dl|materialized|streaming.*has_D_L\s*=\s*False',s): f.append('manifest:has_D_L without disk-exists guard')
if re.search(r'target\s*=\s*len\(segments\)',s) and 'subset_keys' in s and not re.search(r'(target|n_to_process)\s*=\s*len\(subset_keys',s): f.append('pbar:target ignores subset')
print(f'B60 FAIL:{f}' if f else 'B60 PASS');sys.exit(1 if f else 0)"
```

**B61.** Dead yaml field — `lr_schedule`/`schedule_type`/`optimizer_type`/`sampler_type` declared but read by zero `.py` (iter12 #78 — operator believed one schedule, code ran another). `.yaml` only; greps `src/**/*.py`.
```bash
source venv_walkindia/bin/activate && python3 -c "
import yaml,sys,pathlib,re;y='<file>'
if not y.endswith(('.yaml','.yml')): print('B61 SKIP');sys.exit(0)
d=yaml.safe_load(open(y)) or {}
SUSPECT={'lr_schedule','schedule_type','optimizer_type','sampler_type'}
found=set()
def w(n):
    if isinstance(n,dict):
        for k,v in n.items():
            if k in SUSPECT: found.add(k)
            w(v)
    elif isinstance(n,list):
        for it in n: w(it)
w(d);sus=found&SUSPECT
if not sus: print('B61 PASS');sys.exit(0)
src=''.join(p.read_text() for p in pathlib.Path('src').rglob('*.py'))
dead=[k for k in sus if not re.search(rf'[\\x22\\x27]{k}[\\x22\\x27]',src)]
print(f'B61 FAIL:dead yaml field {dead}' if dead else f'B61 PASS:{len(sus)} live');sys.exit(1 if dead else 0)"
```

## Part N: iter13 regressions (B62-B65) — catch BEFORE GPU spend

> Sources: `iter/iter13_motion_probe_eval/errors_N_fixes.md` #71 + #79 + #82 + `logs/run_src_probe_v1.log` (2026-05-03 cache-policy + np.savez regressions). Each guard catches a bug class B1-B61 misses.
> NOTE: #80 (long-lived `tmp_dir` ENOENT after ~2M cycles) is already covered by **B56** — `producer_thread` mkdtemp at fn-level pre-fix trips B56's `not in_loop` branch when `fn.name` matches `producer/stream/decode`. No additional guard needed.

**B62.** `getattr(<argparse_args>, "<key>", <non_None_default>)` silent fallback. argparse `required=True` already guarantees presence; a `getattr` default suppresses missing flags as `None` instead of crashing at argparse-time, letting bad state propagate into long GPU runs. Fix: drop the default; let `AttributeError` surface immediately. `.py` only; flags only when receiver name looks like an argparse Namespace.
```bash
python3 -c "
import ast,sys;t='<file>'
if not t.endswith('.py'): print('B62 SKIP');sys.exit(0)
tr=ast.parse(open(t).read())
b=[]
for n in ast.walk(tr):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='getattr' and len(n.args)>=3:
  default=n.args[2]
  if isinstance(default,ast.Constant) and default.value is None: continue
  recv=n.args[0].id if isinstance(n.args[0],ast.Name) else None
  if recv and recv.lower() in ('args','ns','namespace','opts','cli','flags'):
   b.append((n.lineno,recv))
print(f'B62 FAIL:{b}; argparse Namespace getattr-default suppresses missing flags — use args.<key> directly (required=True guarantees presence; AttributeError surfaces immediately at use)' if b else 'B62 PASS');sys.exit(1 if b else 0)"
```

**B63.** Bash brace-expanded `ls foo/{a,b,…}.ext` under `set -e + pipefail` is fragile: any missing file in the brace list makes `ls` exit non-zero, which `pipefail` propagates and `-e` turns into a whole-pipeline abort — even when only some of the listed files are required. Fix: per-file `[ -f X ] && ls -lh X` loop, or append `|| true` to the `ls` call only. `.sh` only.
```bash
python3 -c "
import re,sys;t='<file>'
if not t.endswith('.sh'): print('B63 SKIP');sys.exit(0)
s=open(t).read()
strict=bool(re.search(r'^\s*set\s+-\w*e\w*',s,re.M)) and bool(re.search(r'pipefail',s))
if not strict: print('B63 SKIP');sys.exit(0)
b=[]
for i,l in enumerate(s.split(chr(10)),1):
 ls=l.lstrip()
 if ls.startswith('#') or not ls.strip(): continue
 if not re.search(r'\bls\b',l): continue
 if not re.search(r'\{[^}]*,[^}]*\}',l): continue
 if '||' in l: continue
 b.append((i,l.strip()[:90]))
print(f'B63 FAIL:{b}; ls brace-expansion can fail-cascade under set -e+pipefail when any listed file is optional. Use a for-loop with [ -f ] guard, or append || true on JUST the ls call.' if b else 'B63 PASS');sys.exit(1 if b else 0)"
```

**B65.** Atomic-rename anti-pattern: a writer that **silently mutates its path argument** (numpy's `save`/`savez`/`savez_compressed` auto-append `.npy` / `.npz` per [docs](https://numpy.org/doc/stable/reference/generated/numpy.save.html)) followed by `os.replace(<tmp>, final)` — when `<tmp>` doesn't end in the writer's expected suffix, numpy writes to `<tmp>.{ext}` instead, and the rename targets the un-mutated name → `FileNotFoundError` at the first checkpoint save. Detection is **default-to-FAIL**: when static analysis can't *prove* `<tmp>` ends in the expected ext, we flag rather than skip — so the same bug class can't slip through a new binding idiom we haven't seen yet. Add new auto-mangling writers to `WRITERS` to extend coverage. `.py` only.
```bash
python3 << 'PY'
import re,sys
t='<file>'
if not t.endswith('.py'): print('B65 SKIP'); sys.exit(0)
s=open(t).read()

# Writer → suffix that the writer auto-appends when the path arg lacks it.
# Add new entries here when a future writer with the same anti-pattern appears
# (e.g., third-party libs that silently rewrite their path arg).
WRITERS = {'save':'.npy', 'savez':'.npz', 'savez_compressed':'.npz'}

# Regex helpers — direction-aware concat patterns.
# SAFE-by-inheritance:  with_suffix("<literal>" + X.suffix)  → literal prefixes, X.suffix is the last → ext inherited
SAFE_INHERIT = re.compile(r'\.with_suffix\(\s*[\'"](\.[^\'"]*)[\'"]\s*\+\s*\w+\.suffix\b')
# BUG-by-append:        with_suffix(X.suffix + "<literal>")  → X.suffix prefixes, literal appends → final ends in literal
BUG_APPEND = re.compile(r'\.with_suffix\(\s*\w+\.suffix\s*\+\s*[\'"](\.[^\'"]*)[\'"]\s*\)')

bugs = []
for m in re.finditer(r'np\.(save|savez|savez_compressed)\(\s*(\w+)\s*[,\)]', s):
    writer, var = m.group(1), m.group(2)
    expected = WRITERS[writer]
    pos = m.start(); line = s[:pos].count('\n') + 1
    # Must be paired with os.replace(var, ...) within ~800 chars (same fn).
    if not re.search(rf'\bos\.replace\(\s*{var}\s*,', s[pos:pos+800]): continue
    # Find the most recent assignment to var.
    back = s[max(0, pos-1500):pos]
    binds = list(re.finditer(rf'(?:^|\n)\s*{var}\s*=\s*([^\n]+)', back))
    if not binds:
        bugs.append((line, var, writer, expected, '<no binding found — manually verify>'))
        continue
    rhs = binds[-1].group(1).strip()
    # Decision tree:
    if BUG_APPEND.search(rhs):                                         # 1) appended literal → BUG
        sx = BUG_APPEND.search(rhs).group(1)
        if not sx.endswith(expected):
            bugs.append((line, var, writer, expected, f'<X.suffix>+{sx!r}'))
        continue
    if SAFE_INHERIT.search(rhs):                                       # 2) prefix literal + .suffix → safe-by-inheritance
        continue
    literals = re.findall(r'[\'"]([^\'"]*)[\'"]', rhs)                 # 3) any literal ends in expected ext → safe
    if literals and any(lit.endswith(expected) for lit in literals):
        continue
    # 4) DEFAULT-TO-FAIL: nothing in the binding proves safety.
    bugs.append((line, var, writer, expected, f'unverifiable RHS: {rhs[:80]!r}'))

if bugs:
    fmt = '; '.join(f'L{l}: np.{w}({v}) needs tmp ending in {e}, got {sx}' for l,v,w,e,sx in bugs)
    print(f'B65 FAIL: numpy auto-append → silent path mutation → os.replace fails. {fmt}. Fix: bind tmp so a literal ending in the expected ext appears in the RHS (e.g., with_suffix(".tmp{{ext}}")), use safe-inheritance (with_suffix("<lit>" + X.suffix)), OR pass an open file handle to the writer (file objects are not auto-renamed).')
    sys.exit(1)
print('B65 PASS')
PY
```

**B64.** Cache-policy must be gathered UPFRONT and passed per-call (`scripts/delete_protection_for_each_variant.md`). `.sh` orchestrators with ≥2 invocations of `python -u .../m*.py` (whose `.py` all use `add_cache_policy_arg` + `resolve_cache_policy_interactive`) MUST pass `--cache-policy <N>` to **every** call. Otherwise the `.py`-level `input()` prompt fires per-stage during overnight runs and gets eaten by interleaved log noise. Compliant pattern: `declare -A POLICY` + an upfront `_check_and_prompt` loop. `.sh` only.
```bash
python3 << 'PY'
import re,sys
t='<file>'
if not t.endswith('.sh'): print('B64 SKIP');sys.exit(0)
s=open(t).read()
positions=[m.start() for m in re.finditer(r'python\s+-u\s+\S*m\w*\.py',s)]
if len(positions)<2: print(f'B64 SKIP (<2 python -u m*.py calls: {len(positions)})');sys.exit(0)
missing=[]
for pos in positions:
    chunk=s[pos:pos+1000]
    lines=chunk.split('\n')
    cmd_lines=[lines[0]]
    for l in lines[1:]:
        if cmd_lines[-1].rstrip().endswith('\\'):
            cmd_lines.append(l)
        else: break
    cmd=' '.join(cmd_lines)
    if '--cache-policy' not in cmd:
        line_no=s[:pos].count('\n')+1
        missing.append(line_no)
if missing:
    print(f'B64 FAIL: {len(missing)}/{len(positions)} python -u m*.py calls lack --cache-policy at lines {missing}. Gather UPFRONT (declare -A POLICY + _check_and_prompt loop) and pass --cache-policy "$P_X" to each call. See scripts/delete_protection_for_each_variant.md for the contract.')
    sys.exit(1)
print(f'B64 PASS: {len(positions)} python -u m*.py calls, all pass --cache-policy')
PY
```

## Output Format
```
=== PREFLIGHT: <file> ===
AUTO:[A1-A3] GENERIC:[B1-B9] iter8:[B10-B15] TX5:[B16-B20] G-SAM:[B21-B25] SAM3:[B26-B27]
VJEPA/CFG:[B28-B31] m09:[B32-B36] DURABILITY:[B37-B42] AGNOSTIC:[B43-B47] ORCH:[B48-B49] SEL/QUEUE:[B50-B51]
ITER12:[B52-B61] ITER13:[B62-B65]
TOTAL: X/68 passed. Y FAILs.
```
List FAILs with line numbers + fix. Runtime assertions → `src/m*.py`.
