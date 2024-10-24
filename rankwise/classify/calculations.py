from operator import __lt__ as strictly_least
from operator import __le__ as least_among
from operator import __gt__ as strictly_greatest
from operator import __ge__ as greatest_among


__ALL__ = ["is_best", "strictly_least", "least_among", "strictly_greatest", "greatest_among"]


def is_best(distance_fn, comparison_fn, all_documents, question, this_document):
    """
    Returns true if the query is best suited for the document according
    to the distance function and comparison function taking into account
    all the documents in the set.
    """
    other_documents = all_documents - set([this_document])
    distance_with_this_document = distance_fn(question, this_document)
    distance_with_other_documents = (
        distance_fn(question, another_document) for another_document in other_documents
    )
    return all(
        comparison_fn(distance_with_this_document, distance_with_other_document)
        for distance_with_other_document in distance_with_other_documents
    )
