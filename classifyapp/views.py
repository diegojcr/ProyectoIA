from django.shortcuts import render
from .naive_bayes import classify_text, get_model_accuracy

def classify_view(request):
    prediction = None
    input_text = ''
    accuracy = get_model_accuracy() * 100 if get_model_accuracy() is not None else None
    if request.method == "POST":
        input_text = request.POST.get("article", "")
        prediction = classify_text(input_text)
    return render(request, "classifyapp/classify.html", {
        "prediction": prediction,
        "input_text": input_text,
        "accuracy": accuracy
    })
