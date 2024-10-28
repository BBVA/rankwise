# Copyright 2024 Banco Bilbao Vizcaya Argentaria, S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import lru_cache as cache
from statistics import mean


def build_distance_function(embedding_functions):
    """
    Builds a function that calculates the average cosine distance between two texts using
    multiple embedding models.
    """

    from sklearn.metrics.pairwise import \
        cosine_similarity  # Heavy import, so we import it here

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
