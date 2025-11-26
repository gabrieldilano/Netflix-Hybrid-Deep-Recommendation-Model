# CINEMATCH Hybrid Reccomender system

## Project Overview
This Project uses the Movie Lens 20m dataset in order to provide users robust reccomendations based on their movie tastes. 
In order to do this, cold start functionallity is provided through a streamlit UI where a user rates at minimum 15 of the most
rated movies and saves users' preferences from that. To provide reccomendations, we use a hybrid approach using a combination of
content based filtering and collaborative filtering. With the hybrid approach we gave the collaborative filtering results a weight of
80% and the content based filtering results a weight of 20%. Using the hybrid approach, the project generates a list of movies
reccomended to the user.

## How to Run the project
To run the project you must first install the requirements from the provided requirements.txt. to do this run the following command after opening a terminal in the folder where requirenents.txt is located:

pip install -r requirements.txt

next run the streamlit UI:

cd code  
streamlit run streamlit_app.py

On first run the dataset will be downloaded if is not already in the required folder. Once finished, the first thing the UI will display is a welcome screen.
Here you enter a username. A profile will be created for you based off of the username entered if this is the first time said usernmae has been entered. if it is your first time running
the program, you will be asked to rate 15-20 movies of a scale of 1-5. If you have seen the movie, give a rating using the UI. After you have rated at least 15 movies, you can get reccomendations immidiately by
clicking the "Get Reccomendations" Button. If you havent seen the movie, press the skip button. After rating all the movies or choosing to get reccomendations early, our system
will provide you a list of movies reccomended to you based on your tastes. If you want more or different reccomendations, you can login to the same user profile to avoid having to go through the onbarding process again.
