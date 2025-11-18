#!/usr/bin/env python3
"""
Movie Recommendation Engine

Authors: Wiktor Rapacz, Hanna Paczoska

This script implements a simple user-based collaborative filtering system that
recommends movies and TV series for a selected user, based on the ratings of
other users with similar tastes.

Main ideas:
    - Input data: ratings of movies/series stored in a JSON file (movies.json),
      in the form:
          {
              "user0": {"Movie title A": 10, "Movie title B": 7, ...},
              "user1": {...},
              ...
          }

    - Similarity between users is computed using the Pearson correlation
      coefficient, based only on movies that both users have rated.

    - For a selected user (e.g. "user0"), the system:
         1. Finds the most similar users (neighbors).
         2. Predicts ratings for movies that the selected user has not rated yet,
            using a weighted average of neighbors' ratings.
         3. Produces:
             - Top-N recommended movies (highest predicted ratings).
             - Top-N anti-recommendations (lowest predicted ratings).

    - For each recommended / anti-recommended title, the script tries to fetch
      an additional description from Wikipedia using its public REST API.
      It uses a search endpoint and then filters results to prefer pages that
      look like movies/series.

Environment and setup:
    1. Python version:
        - Python 3.10 or newer is recommended
          (the code uses modern type hints like `str | Path`).

    2. Required packages:
        - requests

       You can install it via:
           pip install requests

    3. Data file:
        - Place the file `movies.json` in the same directory as this script.
        - Encoding should be UTF-8.
        - The structure of the JSON file must be:
            {
                "user0": {"Title 1": 10, "Title 2": 8, ...},
                "user1": {...},
                ...
            }

    4. How to run:
        - From a terminal / command line, navigate to the directory with this file
          and movies.json, then run:
              python recommender.py

        - The program will:
            * List all available user IDs (user0, user1, ...).
            * Ask you to choose a user (e.g. "user0" or just "0").
            * Print a formatted report with:
                - the most similar users,
                - top recommendations,
                - top anti-recommendations,
                - short descriptions fetched from Wikipedia (if available).

"""

import json
import math
import requests
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple


Ratings = Dict[str, Dict[str, float]]  # {user: {title: rating}}


# ==========================
# 1. Data loading
# ==========================

def load_ratings(path: str | Path) -> Ratings:
    """
    Load user ratings from a JSON file.

    The expected structure of the JSON file is:
        {
            "user0": {"Movie A": 10, "Movie B": 8, ...},
            "user1": {...},
            ...
        }

    Args:
        path: Path to the JSON file with ratings (str or pathlib.Path).

    Returns:
        A nested dictionary mapping user IDs to dictionaries of
        {movie_title: rating}.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data


# ==========================
# 2. Similarity measure (Pearson)
# ==========================

def pearson_similarity(
    ratings_u: Dict[str, float],
    ratings_v: Dict[str, float],
    min_common: int = 3
) -> float:
    """
    Compute the Pearson correlation coefficient between two users.

    The similarity is computed only on movies that have been rated by both users.
    If there are too few common movies or there is no variance in the ratings,
    the function returns 0.0 (no reliable similarity).

    Args:
        ratings_u: Dictionary of ratings for user U: {title: rating}.
        ratings_v: Dictionary of ratings for user V: {title: rating}.
        min_common: Minimum number of common rated titles required to compute
            the similarity.

    Returns:
        A float value in the range [-1.0, 1.0]:
            - 1.0 means perfect positive correlation (very similar tastes),
            - 0.0 means no linear correlation,
            - -1.0 means perfect negative correlation (opposite tastes).
        If there are too few common items or no variance, 0.0 is returned.
    """
    common_titles = set(ratings_u.keys()) & set(ratings_v.keys())

    if len(common_titles) < min_common:
        return 0.0

    # Collect ratings only for movies rated by both users
    u_vals = [ratings_u[t] for t in common_titles]
    v_vals = [ratings_v[t] for t in common_titles]

    mean_u = sum(u_vals) / len(u_vals)
    mean_v = sum(v_vals) / len(v_vals)

    num = 0.0
    den_u = 0.0
    den_v = 0.0

    for title in common_titles:
        du = ratings_u[title] - mean_u
        dv = ratings_v[title] - mean_v
        num += du * dv
        den_u += du * du
        den_v += dv * dv

    if den_u == 0 or den_v == 0:
        return 0.0

    return num / math.sqrt(den_u * den_v)


def compute_neighbors(
    target_user: str,
    ratings: Ratings,
    min_common: int = 3
) -> List[Tuple[str, float]]:
    """
    Compute similarities between the target user and all other users.

    This function iterates over all users in the ratings dictionary and
    computes the Pearson similarity with the target user. Only users with
    positive similarity (similar tastes) are kept.

    Args:
        target_user: ID of the user for whom we compute neighbors (e.g. "user0").
        ratings: All users' ratings.
        min_common: Minimum number of common movies required to compute
            similarity.

    Returns:
        A list of (user_id, similarity) tuples sorted in descending order of
        similarity.
    """
    target_ratings = ratings[target_user]
    sims: Dict[str, float] = {}

    for user_id, user_ratings in ratings.items():
        if user_id == target_user:
            continue

        sim = pearson_similarity(target_ratings, user_ratings, min_common=min_common)
        if sim > 0:  # we ignore negative similarity (opposite tastes)
            sims[user_id] = sim

    neighbors = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    return neighbors


# ==========================
# 3. Rating prediction
# ==========================

def predict_ratings_for_user(
    target_user: str,
    ratings: Ratings,
    k_neighbors: int = 7,
    min_common: int = 3
) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
    """
    Predict ratings of unseen movies for a given user using neighbor-based CF.

    For the target user, the function:
        1. Finds the most similar users (neighbors).
        2. For each movie not rated by the target user, computes a predicted
           rating as a weighted average of neighbors' ratings, where weights
           are the similarity scores.

    Args:
        target_user: ID of the user for whom we predict ratings.
        ratings: Full rating matrix (all users).
        k_neighbors: Maximum number of top neighbors to use in the prediction.
        min_common: Minimum number of common movies required to compute
            similarity.

    Returns:
        A tuple:
            - predictions: dict mapping movie titles to predicted ratings.
            - neighbors: list of (user_id, similarity) tuples used in prediction.
    """
    neighbors = compute_neighbors(target_user, ratings, min_common=min_common)

    if k_neighbors:
        neighbors = neighbors[:k_neighbors]

    target_ratings = ratings[target_user]

    # Set of all movies rated by any user
    all_titles = {title for user_ratings in ratings.values() for title in user_ratings.keys()}

    # Candidates = movies not yet rated by the target user
    candidates = [t for t in all_titles if t not in target_ratings]

    predictions: Dict[str, float] = {}

    for title in candidates:
        num = 0.0
        den = 0.0
        for user_id, sim in neighbors:
            user_rating = ratings[user_id].get(title)
            if user_rating is None:
                continue
            num += sim * user_rating
            den += abs(sim)

        if den > 0:
            predictions[title] = num / den

    return predictions, neighbors


# ==========================
# 4. Recommendations and anti-recommendations
# ==========================

def recommend_and_antirecommend(
    target_user: str,
    ratings: Ratings,
    n: int = 5,
    k_neighbors: int = 7,
    min_common: int = 3
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Produce top-N recommendations and anti-recommendations for a user.

    Args:
        target_user: ID of the user for whom we generate suggestions.
        ratings: Full rating matrix (all users).
        n: Number of top items to return for recommendations and
           anti-recommendations.
        k_neighbors: Maximum number of neighbors to use in predictions.
        min_common: Minimum number of common movies required to compute
            similarity.

    Returns:
        A tuple of three lists:
            - best:  list of (title, predicted_rating) for recommendations.
            - worst: list of (title, predicted_rating) for anti-recommendations.
            - neighbors: list of (user_id, similarity) used in prediction.
    """
    predictions, neighbors = predict_ratings_for_user(
        target_user, ratings, k_neighbors=k_neighbors, min_common=min_common
    )

    # Highest predicted ratings -> recommendations
    best = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
    # Lowest predicted ratings -> anti-recommendations
    worst = sorted(predictions.items(), key=lambda x: x[1])[:n]

    return best, worst, neighbors


# ==========================
# 5. Movie descriptions – Wikipedia API (search + movie filter)
# ==========================

WIKI_SEARCH_PL = "https://pl.wikipedia.org/w/rest.php/v1/search/title"
WIKI_SEARCH_EN = "https://en.wikipedia.org/w/rest.php/v1/search/title"

WIKI_SUMMARY_PL = "https://pl.wikipedia.org/api/rest_v1/page/summary/"
WIKI_SUMMARY_EN = "https://en.wikipedia.org/api/rest_v1/page/summary/"

MOVIE_KEYWORDS = [
    "film", "movie", "serial", "series", "miniseries",
    "telewizyjny", "television series"
]


def fetch_movie_info_from_wikipedia(title: str) -> dict | None:
    """
    Fetch a short movie/series description from Wikipedia.

    The function uses the Wikipedia REST API in two steps:
        1. It performs a search query for the given title.
        2. Among the search results, it tries to choose a page that looks like
           a movie or TV series (based on keywords in the title or description).
        3. It fetches the summary (short description) for the chosen page.

    The function first tries the Polish Wikipedia, and if nothing appropriate
    is found, it falls back to the English Wikipedia.

    Args:
        title: Title of the movie/series to search for.

    Returns:
        A dictionary with keys:
            - "title": Title of the Wikipedia page,
            - "overview": Short textual summary (description),
        or None if no suitable page could be found.
    """

    def _search_and_summary(search_url: str, summary_base_url: str, query: str) -> dict | None:
        """
        Helper function that performs search and summary retrieval for one wiki.
        """
        params = {"q": query, "limit": 5}
        headers = {
            "User-Agent": "silnik do rekomendacji filmów (mailto:s26834@pjwstk.edu.pl)"
        }

        try:
            # 1. Search pages by title
            r = requests.get(search_url, params=params, headers=headers, timeout=10)
            if r.status_code != 200:
                return None
            data = r.json()
            pages = data.get("pages") or []
            if not pages:
                return None

            # 2. Try to find a page that looks like a MOVIE / SERIES
            chosen_page = None
            for p in pages:
                key = (p.get("key") or "").lower()
                desc = (p.get("description") or "").lower()

                if any(kw in key or kw in desc for kw in MOVIE_KEYWORDS):
                    chosen_page = p
                    break

            # If we did not find anything that clearly looks like a movie,
            # we fall back to the first search result.
            if chosen_page is None:
                chosen_page = pages[0]

            slug = chosen_page.get("key")
            if not slug:
                return None

            # 3. Fetch the summary (short description)
            summary_url = summary_base_url + slug
            s = requests.get(summary_url, headers=headers, timeout=10)
            if s.status_code != 200:
                return None
            summary = s.json()
            extract = summary.get("extract")
            if not extract:
                return None

            return {
                "title": summary.get("title", query),
                "overview": extract
            }

        except (requests.RequestException, ValueError):
            return None

    # First try Polish Wikipedia
    result = _search_and_summary(WIKI_SEARCH_PL, WIKI_SUMMARY_PL, title)
    if result:
        return result

    # If nothing found, try English Wikipedia
    return _search_and_summary(WIKI_SEARCH_EN, WIKI_SUMMARY_EN, title)


# ==========================
# 6. Output / presentation helpers
# ==========================

def shorten(text: str, max_chars: int = 280) -> str:
    """
    Shorten a text to at most `max_chars` characters, preserving whole words.

    The function trims the text, then:
        - If the length is already <= max_chars, returns it unchanged.
        - Otherwise, cuts at max_chars, goes back to the last space,
          and adds "..." at the end.

    Args:
        text: Original text to shorten.
        max_chars: Maximum allowed length of the returned string.

    Returns:
        A shortened version of the input text, possibly ending with "...".
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Try to cut at the last space before the limit
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut + "..."


def print_recommendations_for_user(
    target_user: str,
    ratings: Ratings,
    n: int = 5,
    k_neighbors: int = 7,
) -> None:
    """
    Print a formatted recommendation report for a given user.

    The report contains:
        - a list of most similar users with their Pearson similarity,
        - top-N recommended movies (with predicted ratings),
        - top-N anti-recommended movies (with predicted ratings),
        - short descriptions for each movie, fetched from Wikipedia if possible.

    The textual layout of the report is in Polish, but the function is fully
    documented in English.

    Args:
        target_user: ID of the user for whom the report is generated.
        ratings: Full rating matrix (all users).
        n: Number of recommended and anti-recommended movies to show.
        k_neighbors: Maximum number of neighbors to use for prediction.
    """
    best, worst, neighbors = recommend_and_antirecommend(
        target_user, ratings, n=n, k_neighbors=k_neighbors
    )

    line = "═" * 80
    sep = "─" * 80

    # Header
    print(line)
    print(f"  REKOMENDACJE FILMÓW – dla użytkownika: {target_user}")
    print(line)

    # Recommendations
    print("\n" + sep)
    print("TOP Rekomendacje (filmy, których użytkownik nie oglądał):")
    print(sep)

    if not best:
        print("  (brak filmów do zarekomendowania – użytkownik widział prawie wszystko)")
    else:
        for idx, (title, score) in enumerate(best, start=1):
            print(f"\n  {idx}. ▶ {title}  (przewidywana ocena: {score:.2f}/10)")
            info = fetch_movie_info_from_wikipedia(title)
            if info and info.get("overview"):
                overview = shorten(info["overview"], max_chars=320)
                wrapped = textwrap.fill(overview, width=160)
                print("     Opis:")
                print("     " + wrapped.replace("\n", "\n     "))
            else:
                print("     Opis: brak opisu w Wikipedii.")

    # Anti-recommendations
    print("\n" + sep)
    print("TOP Antyrekomendacje (filmy, których lepiej unikać):")
    print(sep)

    if not worst:
        print("  (brak antyrekomendacji – brak filmów z niską przewidywaną oceną)")
    else:
        for idx, (title, score) in enumerate(worst, start=1):
            print(f"\n  {idx}. ✖ {title}  (przewidywana ocena: {score:.2f}/10)")
            info = fetch_movie_info_from_wikipedia(title)
            if info and info.get("overview"):
                overview = shorten(info["overview"], max_chars=320)
                wrapped = textwrap.fill(overview, width=160)
                print("     Opis:")
                print("     " + wrapped.replace("\n", "\n     "))
            else:
                print("     Opis: brak opisu w Wikipedii.")

    print("\n" + line)
    print("Koniec raportu.\n")


# ==========================
# 7. Script entry point
# ==========================

if __name__ == "__main__":
    # 1. Load ratings from movies.json
    ratings_data = load_ratings("movies.json")

    # 2. List available users (sorted "naturally" by numeric ID)
    def user_key(u: str) -> int:
        """
        Extract the numeric part from a user ID of the form 'userX'.

        Args:
            u: User ID string, e.g. 'user0', 'user15'.

        Returns:
            Integer numeric suffix of the user ID (e.g. 0, 15).
        """
        return int(u.replace("user", ""))

    users = sorted(ratings_data.keys(), key=user_key)

    print("=" * 80)
    print("Silnik rekomendacji filmów")
    print("=" * 80)
    print("\nDostępni użytkownicy:")
    print(", ".join(users))

    # 3. Ask user which ID should be used for generating recommendations
    raw = input(
        "\nPodaj ID użytkownika, dla którego chcesz wygenerować rekomendacje "
        "(np. user0). \nPozostaw puste, aby użyć domyślnego 'user0': "
    ).strip()

    if raw == "":
        selected_user = "user0"
    else:
        # Allow entering just the numeric part, e.g. "0" -> "user0"
        if not raw.startswith("user"):
            selected_user = f"user{raw}"
        else:
            selected_user = raw

    if selected_user not in ratings_data:
        print(f"\n[!] Użytkownik '{selected_user}' nie istnieje w danych.")
        print("    Dostępni użytkownicy to:")
        print("    " + ", ".join(users))
    else:
        print(f"\nGeneruję rekomendacje dla: {selected_user}\n")
        print_recommendations_for_user(selected_user, ratings_data, n=5, k_neighbors=7)
