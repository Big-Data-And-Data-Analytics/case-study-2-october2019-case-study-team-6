1. Pull master to local master
	- git checkout master
	- git pull
	- git status

2. Merge to your branch ('Your_Branch')
	- git checkout  'Your_Branch'
	- git merge master  --- Master merges into 'Your_Branch'
		* if conflicts shows which file to edit
	- once done

3. Merge 'Your_Branch' to master again
	- git checkout master
	- git merge 'Your_Branch'
	- git status
		* will tell you are ahead by commits
	- git push



-If you make a mistake. Delete the commit. Or cherry pick. Please check the link below  for reference
https://christoph.ruegg.name/blog/git-howto-revert-a-commit-already-pushed-to-a-remote-reposit.html
 * Look for Case 2: Delete the second last commit