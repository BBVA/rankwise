def calculate_distance(model, query, doc):
    return model.predict([query, doc])
