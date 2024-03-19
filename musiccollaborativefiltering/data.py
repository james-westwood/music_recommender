"""This module features functions and classes to manipulate data for the
collaborative filtering algorithm.
"""

from pathlib import Path

import scipy
import pandas as pd


def load_user_artists(user_artists_file: Path) -> scipy.sparse.csr_matrix:
    """Load the user artists file and return a user-artists matrix in csr
    fromat.
    
    Parameters:
    user_artists_file (Path): The path to the user artists file.

    Returns:
    scipy.sparse.csr_matrix: The user-artists matrix in csr format.

    Example:
    ```
    # Create a 2D array with mostly zero values
data = [
    [0, 0, 1],
    [2, 0, 0],
    [0, 3, 0]
]

# Convert the 2D array to a csr_matrix
matrix = csr_matrix(data)
    ```
    This will output something like:
    ```
      (0, 2)	1.0
      (1, 0)	2.0
      (2, 1)	3.0
    ```
    Each line represents a non-zero element in the matrix. The numbers in parentheses `(i, j)` represent the row index `i` and the column index `j` of the non-zero element, and the number after the tab is the value of that element.
    """
    # Load the user artists file (.dat file) into a Pandas dataframe
    user_artists = pd.read_csv(user_artists_file, sep="\t")
    
    # Set the index of the dataframe to the userID and artistID columns
    user_artists.set_index(["userID", "artistID"], inplace=True)
    
    # Create a COO sparse matrix from the dataframe
    coo = scipy.sparse.coo_matrix(
        (
            user_artists.weight.astype(float),
            (
                user_artists.index.get_level_values(0), # get_level_values returns the values of the index at the specified level
                user_artists.index.get_level_values(1), # get_level_values returns a vector
            ),
        )
    )
    
    # Convert the COO matrix to a CSR matrix
    csr = coo.tocsr()
    
    return csr


class ArtistRetriever:
    """The ArtistRetriever class gets the artist name from the artist ID."""

    def __init__(self):
        self._artists_df = None

    def get_artist_name_from_id(self, artist_id: int) -> str:
        """Return the artist name from the artist ID."""
        return self._artists_df.loc[artist_id, "name"]

    def load_artists(self, artists_file: Path) -> None:
        """Load the artists file and stores it as a Pandas dataframe in a
        private attribute.
        """
        artists_df = pd.read_csv(artists_file, sep="\t")
        artists_df = artists_df.set_index("id")
        self._artists_df = artists_df


if __name__ == "__main__":
    # user_artists_matrix = load_user_artists(
    #     Path("../lastfmdata/user_artists.dat")
    # )
    # print(user_artists_matrix)

    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("../data/artists.dat"))
    artist = artist_retriever.get_artist_name_from_id(1)
    print(artist)