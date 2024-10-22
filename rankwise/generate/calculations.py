def format_output(document, good_questions, bad_questions):
    """
    Formatea la salida en el formato jsonlines requerido y lo escribe en el archivo de salida.

    Parameters:
    - file_handler: Archivo abierto en modo escritura.
    - document (str): El contenido del documento.
    - good_questions (list): Lista de preguntas buenas.
    - bad_questions (list): Lista de preguntas malas.
    """
    return {
        "document": document,
        "questions": {"good": good_questions, "bad": bad_questions},
    }
