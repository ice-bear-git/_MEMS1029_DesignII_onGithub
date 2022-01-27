# How to use git to clone from github and push changes to the cloud?

## Preparation
* Create token to autorize the connection toward github.
* Replace the password by the link that created by token for the first-time login.

## General Procedure
The terminal command lines for git
```python
cd _desired_folder

git clone _repository_URL

# After making some changes
git status

# If there are files that shown in red font -- they are untracked
# Require to update it [add . means add all]
git add .

# Double check new status
# The files should all be green and wait to be committed
git status

# Add descriptions at committing time
git commit -m "The description of the change [addtional functionality or current progress...]"

# push the local change to the github cloud server 
git push

```
