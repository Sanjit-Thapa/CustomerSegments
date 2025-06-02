from django.shortcuts import render,redirect
from .forms import UploadCSV
from pathlib import Path
from .utils.customer_segmentation import run_analysis
from django.contrib import messages
# Create your views here.

def Home(request):
    if request.method == 'POST':
        csv_file_form = UploadCSV(request.POST,request.FILES)

        if csv_file_form.is_valid():
            file_form = csv_file_form.save(commit=False)
            
            #making sure that the file has been uploaded before accessing the path as it stores temporarily csv_FILE is the field of the form
            uploaded_csv = request.FILES['csv_file']
            if Path(uploaded_csv.name).suffix == '.csv':
                file_form.save()
                csv_path = file_form.csv_file.path
                print(csv_path)
                results =  run_analysis(Path(csv_path))
                if results is None: 
                    messages.error(request, "Data analysis failed. Please check your CSV file format.")
                    return redirect('home')
                return render(request,'FileUpload/Home.html',context={
                    'csv_form': csv_file_form,
                    'top_products': results['top_products'],
                    'rfm_segments': results['rfm_segments'],
                    'rules': results['rules'],
                    'plots': results['plots'],
                    'loyalty_summary': results['loyalty_summary'],
                    'clv': results['clv'],
                    'geo_summary': results['geo_summary'],
                    'day_summary': results['day_summary'],
                    'hour_summary': results['hour_summary'],
                    'churn': results['churn']
             }) 

            else:
                print('Sorry the extension must be the .csv')
        else:
            print("Something is not valid")
    else:
        csv_file_form = UploadCSV()
    return render(request,'FileUpload/Home.html',context={
        'csv_form':csv_file_form
    }) 