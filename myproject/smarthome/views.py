from django.shortcuts import render
from django.http import JsonResponse
from .models import Classification
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View

def classification_list(request):
    classifications = Classification.objects.all().order_by('-timestamp')  # Getting all classifications ordered by timestamp
    return render(request, 'smarthome/main.html', {'classifications': classifications})

@csrf_exempt
def receive_data(request):
    if request.method == 'POST':
        label = request.POST.get('label', 'Unknown')
        media_file = request.FILES.get('media_file')

        classification = Classification.objects.create(
            label=label,
            media_file=media_file
        )
        
        return JsonResponse({'status': 'success', 'id': classification.id})