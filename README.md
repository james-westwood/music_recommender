# music_recommender
Music recommender System

Building a music artist recommender system, using collaborative filtering (Matrix Factorisation). 

We start with data of users, and their likes represented in a user artist matrix, which is sparsely populated, as users cannot have an 'opinion' on every artist. The aim is to fill in the missing data with predicted opinions using matrix factorisation. 

The matrix factorisation algorithm being used is the alternating least squares algorithm. The user (listener) artist matrix is decomposed into two smaller matrices, one for the user and one for the artsts. The decomposition is carried out via alternating least squares. 

Collaborative filtering allows us to extract features without i) knowing what the features are, or ii) having to do any feature engineering. 


## How Artists are recommended to users

1) Multiple a user vector with all artist vectors. So each user gets a dot product for artist 1, artist 2.....artist n. 
2) Recommend x artists who are new to the user, with the highest scores.


## Ideas for improving this system

### Improve Data Sources:

Possibly include datasets from Spotify API (https://developer.spotify.com/documentation/web-api), or Million Song Dataset (http://millionsongdataset.com/), which provide richer song features like genre, tempo, and audio properties. This allows for content-based filtering alongside collaborative filtering with ALS.

Incorporate user data: If you can anonymize user data from Last.fm (like playlists, skips, likes etc.), it allows for a hybrid approach. Combine collaborative filtering with content-based filtering for even more personalized recommendations.

### User Interface

- Move beyond command line: Web interface using frameworks like Flask or Django to make the recommender system more user-friendly and accessible.
- Interactive features Allow users to input their favorite artists, genres, or moods to further personalize recommendations. 
- Implement an "explore similar" feature where users click on an artist and get recommendations for similar artists.
- Visualizations Integrate visualizations to display user listening history, artist networks, or recommended playlists. This makes the results more engaging.

### Improve Accuracy

- Model Tuning: Experiment with different ALS parameters (number of latent factors, regularization) to improve recommendation accuracy. Tools like GridSearchCV can help automate this process.
- Evaluation Metrics: Implement metrics like precision-recall or Normalized Discounted 
- Cumulative Gain (NDCG) to evaluate your model's performance and track improvements.
- Incorporate context: Consider adding context like time of day, day of week, or user activity to further refine recommendations based on potential listening habits.

