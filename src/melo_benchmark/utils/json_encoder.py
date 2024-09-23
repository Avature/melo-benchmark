import json


class SetsAsListsEncoder(json.JSONEncoder):
    """
    Needed for JSON-encoding sets (as lists)

    See: https://docs.python.org/3/library/json.html
    """

    def default(self, obj):
        if isinstance(obj, set):
            return list(sorted(list(obj)))

        # Base class default method to handle other cases
        return super().default(obj)
