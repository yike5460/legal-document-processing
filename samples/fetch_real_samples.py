"""
Fetch real legal document samples from official public sources
"""

import os
import requests
from pathlib import Path
import time

def download_file(url, filename):
    """Download a file from URL"""
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✓ Downloaded {filename}")
            return True
        else:
            print(f"✗ Failed to download {filename}: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")
        return False

def fetch_legal_samples():
    """Fetch various types of legal documents from public sources"""
    
    # Create samples directory if it doesn't exist
    samples_dir = Path(__file__).parent
    os.chdir(samples_dir)
    
    print("Fetching real legal document samples from public sources...")
    print("=" * 60)
    
    # Sample legal documents from public sources
    documents = [
        {
            "name": "court_order_sample.pdf",
            "type": "Court Order",
            "url": "https://www.uscourts.gov/sites/default/files/ao089.pdf",
            "description": "Administrative Office Form - Subpoena"
        },
        {
            "name": "motion_sample.pdf", 
            "type": "Motion",
            "url": "https://www.uscourts.gov/sites/default/files/ao088.pdf",
            "description": "Subpoena to Appear and Testify"
        },
        {
            "name": "complaint_sample.pdf",
            "type": "Complaint",
            "url": "https://www.uscourts.gov/sites/default/files/pro-se-1.pdf",
            "description": "Pro Se 1 - Complaint for a Civil Case"
        },
        {
            "name": "answer_sample.pdf",
            "type": "Answer",
            "url": "https://www.uscourts.gov/sites/default/files/pro-se-3.pdf",
            "description": "Pro Se 3 - Defendant's Answer to the Complaint"
        }
    ]
    
    # Additional sample URLs from other public sources
    additional_samples = [
        {
            "name": "notice_of_hearing.pdf",
            "type": "Notice",
            "url": "https://www.uscourts.gov/sites/default/files/ao450.pdf",
            "description": "Judgment in a Civil Action"
        },
        {
            "name": "discovery_request.pdf",
            "type": "Discovery",
            "url": "https://www.uscourts.gov/sites/default/files/ao088a.pdf",
            "description": "Subpoena to Produce Documents"
        }
    ]
    
    documents.extend(additional_samples)
    
    successful = 0
    failed = 0
    
    for doc in documents:
        print(f"\n{doc['type']}: {doc['description']}")
        print("-" * 40)
        
        if download_file(doc['url'], doc['name']):
            successful += 1
            # Add small delay to be respectful to servers
            time.sleep(1)
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Download Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    # Create fallback samples if needed
    if failed > 0:
        print("\nCreating fallback samples for failed downloads...")
        create_fallback_samples(documents)
    
    print("\nDone! Legal document samples are ready for testing.")
    return successful > 0

def create_fallback_samples(documents):
    """Create simple fallback PDFs for documents that couldn't be downloaded"""
    try:
        from pypdf import PdfWriter
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        import io
        
        for doc in documents:
            filename = doc['name']
            if not Path(filename).exists():
                print(f"Creating fallback for {filename}...")
                
                # Create a simple PDF with document type
                buffer = io.BytesIO()
                c = canvas.Canvas(buffer, pagesize=letter)
                width, height = letter
                
                # Add header
                c.setFont("Helvetica-Bold", 16)
                c.drawCentredString(width/2, height - 50, f"Sample {doc['type']}")
                
                c.setFont("Helvetica", 12)
                c.drawCentredString(width/2, height - 80, doc['description'])
                
                # Add sample content based on type
                c.setFont("Helvetica", 11)
                y = height - 120
                
                if doc['type'] == 'Court Order':
                    c.drawString(100, y, "UNITED STATES DISTRICT COURT")
                    y -= 20
                    c.drawString(100, y, "ORDER")
                    y -= 30
                    c.drawString(100, y, "IT IS HEREBY ORDERED that the motion is GRANTED.")
                    y -= 20
                    c.drawString(100, y, "Defendant shall respond within 30 days.")
                    
                elif doc['type'] == 'Motion':
                    c.drawString(100, y, "MOTION TO DISMISS")
                    y -= 30
                    c.drawString(100, y, "Defendant moves this Court to dismiss the complaint")
                    y -= 20
                    c.drawString(100, y, "pursuant to Rule 12(b)(6).")
                    
                elif doc['type'] == 'Notice':
                    c.drawString(100, y, "NOTICE OF HEARING")
                    y -= 30
                    c.drawString(100, y, "Please take notice that a hearing is scheduled")
                    y -= 20
                    c.drawString(100, y, "for 30 days from the date of this notice.")
                    
                else:
                    c.drawString(100, y, f"This is a sample {doc['type']} document")
                    y -= 20
                    c.drawString(100, y, "for testing the legal document processing system.")
                
                c.save()
                
                # Write to file
                buffer.seek(0)
                with open(filename, "wb") as f:
                    f.write(buffer.read())
                print(f"✓ Created fallback {filename}")
                
    except ImportError:
        print("Cannot create fallback PDFs - reportlab not installed")

if __name__ == "__main__":
    success = fetch_legal_samples()
    exit(0 if success else 1)