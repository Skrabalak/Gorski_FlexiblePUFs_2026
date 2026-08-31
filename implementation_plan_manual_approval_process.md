Manual approval process:
1. this will occur after all images in a root directory are completely processed
2. this will occur after the "best" percentage matches are found for all of the red box images in the sample_dir
3. this will present the user with the all_best_percentage image for each of the mm stretches alongside the locations_fixed image for that sample dir. Ensure that the display shows the current mm stretch
4. this will allow the user to select whether or not the all_best_percentage image matches the expected locations fixed image
5. if the user selects YES, then the all_best_percentage image for that mm stretch is correct and the program can move on to the next mm stretch
6. if the user selects NO, then one of the best percentages is incorrect. Present the user with a labeled image for each of the best percentages for that stretch, allowing them to potentially pick more than one. 
7. for each best percentage that was incorrect, do the following:
	1. display the nine best percentages, in a grid formation, each labeled with a number
	2. allow the user to select one of the numbers OR none of the numbers
	3. if the user selects one of the numbers, that should be reused as the correct best percentage match. this should be marked with a percentage of 999, showing a manual override
	4. if the user selects none of the numbers, then redo step one only displaying the next nine best percentages. 
	5. this should allow the user to move forwards or backwards if they choose to do so
	