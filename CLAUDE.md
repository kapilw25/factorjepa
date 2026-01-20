# Github
> git config user.name "kapilw25"
  git config user.email "kapilw25@gmail.com" 

#   Simple solution to overwrite everything:
```
git fetch && git reset --hard origin/main
```
What it does:
- git fetch - downloads remote changes
- git reset --hard origin/main - throws away ALL local changes and matches remote exactly

#   TensorBoard via SSH Tunnel
On Cloud Terminal:
```
tensorboard --logdir=tensorboard_logs --port=6006
```

On Local Terminal:
```
ssh -L 6006:localhost:6006 lambda_A100_40GB
```

Open Browser: http://localhost:6006

# NOTES
>> note: be brutally honest. You do not have to agree with me, unless I am correct. But do not LIE/ Halluciante too
>>  Keep explanation SHORT. I cant read verbose explanations
>> Remove all false advertising / STATIC prints


>> note: being Devil's advocate does NOT mean, hallucinate/fake-produce the mistakes which dont exist. If the code is correct , then accept it and move on
>> WEBSEARCH [not always] : if needed to find the universal practices in Ai/ML research world
- never modify @CLAUDE.md