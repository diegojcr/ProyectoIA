from django.shortcuts import render
from .naive_bayes import classify_text

def classify_view(request):
    prediction = None
    input_text = ''
    if request.method == "POST":
        input_text = request.POST.get("article", "")
        prediction = classify_text(input_text)
    return render(request, "classifyapp/classify.html", {
        "prediction": prediction,
        "input_text": input_text
    })
