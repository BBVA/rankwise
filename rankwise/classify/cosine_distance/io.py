from statistics import mean
from functools import lru_cache as cache


def build_distance_function(embedding_functions):
    """Builds a function that calculates the average cosine distance between two texts using multiple embedding models."""

    from sklearn.metrics.pairwise import cosine_similarity  # Heavy import, so we import it here

    embedding_functions = [cache(fn) for fn in embedding_functions]

    def get_cosine_similarity(e1, e2):
        return cosine_similarity([e1], [e2])[0][0]

    def calculate_distance(text1, text2):
        result = mean(
            get_cosine_similarity(get_text_embedding(text1), get_text_embedding(text2))
            for get_text_embedding in embedding_functions
        )
        return result

    return calculate_distance
