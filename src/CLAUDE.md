1) numbering "m"odules for sorting: [src/m01_<name>.py, src/m02_<name>.py, ] (start with letter "m" to not get into import error with number as prefix)
-- the number m*[01, 02, 3.., 0n] must NOT be REPEATED
2) whem moving form current/previous phase to next phase, move the current python files to @src/legacy/ directory and rename the existing files as per nature of next phase
3) create common / UTILS files here @src/utils
4) note:for both (inference and training) on Nvidia's GPU only. no M1/CPU fallback
keep M1 macbook for only CPU or API based operations
5) in each python script, keep DOCSTRING limited to terminal commands (covering only 2 models --SANITY, --FULL, etc arguments) to be executed 
5.1) each command must have  `python -u src/*.py --args arg_name 2>&1 | tee logs/<log_name>log` format
5.2) start DOCSTRING with  max 2 lines of explanation about the code
6) once all python files are built @src/ directory,
read current versions and then update 
@setup_env.sh , 
@requirements.txt [covering non GPU libraries e.g - numpy, matplotlib, etc], 
@requirements_gpu.txt [covering Nvidia GPU libraries only, e.g - torch, transformers, etc]
6.1) for REPRODUCIBILITY purposes, install everything on "venv_*" via @requirements.txt and/or @requirements_gpu.txt ONLY
no individual isntallations >> which will make REPRODUCIBILITY difficult

7) TEST [`py_compile` && `--help` && `ast` && `import`] using VIRTUAL ENVIRONMENT `source venv_*/bin/activate`
"venv_*" represent to local virtual environment available .e g- here "venv_walkindia"
```
source venv_*/bin/activate     # find   local <venv_*>                                                                       
python -m py_compile src/m01_*.py  # Syntax check                                                      
python src/m01_*.py --help          # Args check                                                       
python -c "import ast; ast.parse(open('src/m01_*.py').read())"  # AST check 
```


8) for each @outputs/plots - both [.png & .pdf] versions should be generated  >> build (src/utils/plotting.py) accordingly