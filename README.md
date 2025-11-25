# CINEMATCH Hybrid Reccomender system

## Project Overview
This Project uses the Movie Lens 20m dataset in order to provide users robust reccomendations based on their movie tastes. 
In order to do this, cold start functionallity is provided through an onboarding script which selects 15-20 of the most
rated movies and saves users' preferences from that. To provide reccomendations, we use a hybrid approach using a combination of
content based filtering and collaborative filtering. With the hybrid approach we gave the collaborative filtering results a weight of
80% and the content based filtering results a weight of 20%. Using the hybrid approach, the project generates a list of 10 movies
reccomended to the user.

## How to Run the project
To run the project you must first install the requirements from the provided requirements.txt. to do this run the following command after opening a terminal in the folder where requirenents.txt is located:

pip install -r requirements.txt

next run the python script:

cd code  
python3 onboarding.py

On first run the dataset will be downloaded if is not already in the required folder. Once finished, the first thing the program
will ask you for is your username. A profile will be created for you based off of your username. if it is your first time running
the program, you will be asked to rate 15-20 movies of a scale of 1-5. If you have seen the movie, type your rating (1-5) and press
enter to move on to the next movie. If you havent seen the movie, type s then enter to skip. After rating all the movies, our system
will provide you a list of 10 movies reccomended to you based on your tastes. If you want more or different reccomendations,
you can rerun the program with the same user profile to avoid having to go through the onbarding process again.
