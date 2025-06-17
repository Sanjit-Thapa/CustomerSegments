from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import UploadCSV,SignUp,Login
from pathlib import Path
from .utils.customer_segmentation import run_analysis
from django.contrib import messages
from django.conf import settings
import pdfkit
from django.contrib.auth import login

from django.template.loader import render_to_string
from datetime import datetime
import os
import logging

# Setup logging
logger = logging.getLogger(__name__)

def Loginin(request):
    if request.method == 'POST':
        form = Login(request.POST)
        if form.is_valid():
            # Log the user in
            login(request, form.user)
            messages.success(request, "Successfully logged in!")
            return redirect('home')  # Replace 'home' with your actual home URL name
        else:
            messages.error(request, "Invalid email or password. Please try again.")
    else:
        form = Login()  # Initialize an empty form for GET requests

    return render(request, 'FileUpload/Login.html', {'form': form})

def SignedUp(request):
    if request.method == 'POST':
        signup_form = SignUp(request.POST)
        if signup_form.is_valid():
            signup_form.save()
            return redirect('login')  # redirect to login page
        else:
            print(signup_form.errors)  # Debug

    else:
        signup_form = SignUp()

    return render(request, 'FileUpload/Signup.html', {
        'signup': signup_form
    })

def Home(request):
    context = {'csv_form': UploadCSV()}

    # Only plots displayed in UI; sections for PDF via dropdown
    context['available_sections'] = {'Plots': 'plots'}

    if request.method == 'POST':
        if request.POST.get('action') == 'download_single_pdf':
            results = request.session.get('analysis_results')
            section = request.POST.get('pdf_section')
            if not results or not section:
                messages.error(request, "No analysis results or section selected. Please upload a CSV file.")
                logger.warning("PDF generation attempted without results or section")
                return render(request, 'FileUpload/Analysis.html', context)

            # Prepare context for PDF template
            pdf_context = {
                'section_name': section.replace('_', ' ').title(),
                'section_data': results.get(section, []),
                'is_plots': section == 'plots',
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Convert plot URLs to file:/// URLs for wkhtmltopdf

            if section == 'plots':
                pdf_context['section_data'] = []
                for plot in results.get('plots', []):
                    if not plot.startswith('data:image'):
                        plot_filename = os.path.basename(plot)
                        file_path = os.path.normpath(os.path.join(settings.MEDIA_ROOT, 'plots', plot_filename))
                        if os.path.exists(file_path):
                            pdf_context['section_data'].append(f"file:///{file_path.replace(os.sep, '/')}")
                        else:
                            logger.warning(f"Plot file not found: {file_path}")
                    else:
                        pdf_context['section_data'].append(plot)

            try:
                html_content = render_to_string('FileUpload/pdf_report.html', pdf_context)
                pdf = pdfkit.from_string(html_content, False, options={'enable-local-file-access': ''})
                response = HttpResponse(content_type='application/pdf')
                response['Content-Disposition'] = f'attachment; filename="{section}_report.pdf"'
                response.write(pdf)
                return response
            except Exception as e:
                messages.error(request, f"Failed to generate PDF: {str(e)}")
                logger.error(f"PDF generation error: {str(e)}")
                return render(request, 'FileUpload/Analysis.html', context)

        if request.POST.get('action') == 'download_all_pdf':
            results = request.session.get('analysis_results')
            if not results:
                messages.error(request, "No analysis results available. Please upload a CSV file.")
                logger.warning("PDF generation attempted without results")
                return render(request, 'FileUpload/Analysis.html', context)

            # Prepare context for all-sections PDF
            pdf_context = {
                'top_products': results.get('top_products', []),
                'rfm_segments': results.get('rfm_segments', []),
                'rules': results.get('rules', []),
                'loyalty_summary': results.get('loyalty_summary', []),
                'clv': results.get('clv', []),
                'geo_summary': results.get('geo_summary', []),
                'day_summary': results.get('day_summary', []),
                'hour_summary': results.get('hour_summary', []),
                'churn': results.get('churn', []),
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            try:
                html_content = render_to_string('FileUpload/pdf_all_report.html', pdf_context)
                pdf = pdfkit.from_string(html_content, False)
                response = HttpResponse(content_type='application/pdf')
                response['Content-Disposition'] = 'attachment; filename="all_sections_report.pdf"'
                response.write(pdf)
                return response
            except Exception as e:
                messages.error(request, f"Failed to generate all-sections PDF: {str(e)}")
                logger.error(f"PDF generation error: {str(e)}")
                return render(request, 'FileUpload/Analysis.html', context)

        # Handle CSV file upload
        csv_file_form = UploadCSV(request.POST, request.FILES)
        if csv_file_form.is_valid():
            file_form = csv_file_form.save(commit=False)
            uploaded_csv = request.FILES['csv_file']
            if Path(uploaded_csv.name).suffix.lower() != '.csv':
                messages.error(request, "Sorry, the file must have a .csv extension.")
                return render(request, 'FileUpload/Home.html', context)

            file_form.save()
            csv_path = file_form.csv_file.path
            results = run_analysis(Path(csv_path))

            if results is None or ('messages' in results and not any(k in results for k in ['top_products', 'rfm_segments'])):
                for msg in results.get('messages', []):
                    if 'Error' in msg:
                        messages.error(request, msg)
                    elif 'Warning' in msg:
                        messages.warning(request, msg)
                    else:
                        messages.info(request, msg)
                return render(request, 'FileUpload/Home.html', context)

            # Fix plot URLs for UI display
            if 'plots' in results:
                results['plots'] = [
                    f"/media/plots/{os.path.basename(plot)}" if not plot.startswith('data:image') else plot
                    for plot in results.get('plots', [])
                ]

            request.session['analysis_results'] = results
            request.session['csv_path'] = csv_path

            selected_section = request.GET.get('section', 'plots')
            context.update({
            'messages': results.get('messages', []),
            'top_products': results.get('top_products', []),  # List of dicts for Chart.js
            'rfm_segments': results.get('rfm_segments', []),
            'rules': results.get('rules', []),
            'plots': [f"/media/plots/{os.path.basename(plot)}" if not plot.startswith('data:image') else plot for plot in results.get('plots', [])],
            'loyalty_summary': results.get('loyalty_summary', []),
            'clv': results.get('clv', []),
            'geo_summary': results.get('geo_summary', []),
            'day_summary': results.get('day_summary', []),
            'hour_summary': results.get('hour_summary', []),
            'churn': results.get('churn', []),
            'selected_section': selected_section
        })
            return render(request, 'FileUpload/Analysis.html', context)
        else:
            messages.error(request, f"Invalid form submission. Errors: {csv_file_form.errors}")
            logger.error(f"Form errors: {csv_file_form.errors}")
            logger.error(f"POST data: {request.POST}")
            logger.error(f"FILES data: {request.FILES}")
            return render(request, 'FileUpload/Home.html', context)

    elif request.method == 'GET' and 'section' in request.GET:
        selected_section = request.GET.get('section', 'plots')
        results = request.session.get('analysis_results')

        if results:
            # Fix plot URLs for UI display
            if 'plots' in results:
                results['plots'] = [
                    f"/media/plots/{os.path.basename(plot)}" if not plot.startswith('data:image') else plot
                    for plot in results.get('plots', [])
                ]

            context.update({
            'messages': results.get('messages', []),
            'top_products': results.get('top_products', []),
            'rfm_segments': results.get('rfm_segments', []),
            'rules': results.get('rules', []),
            'plots': [f"/media/plots/{os.path.basename(plot)}" if not plot.startswith('data:image') else plot for plot in results.get('plots', [])],
            'loyalty_summary': results.get('loyalty_summary', []),
            'clv': results.get('clv', []),
            'geo_summary': results.get('geo_summary', []),
            'day_summary': results.get('day_summary', []),
            'hour_summary': results.get('hour_summary', []),
            'churn': results.get('churn', []),
            'selected_section': selected_section
        })
            return render(request, 'FileUpload/Analysis.html', context)
        else:
            messages.warning(request, "No analysis results found. Please upload a CSV file.")
            return render(request, 'FileUpload/Home.html', context)

    return render(request, 'FileUpload/Home.html', context)


