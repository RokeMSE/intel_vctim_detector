import io
import tempfile
from datetime import datetime
from fpdf import FPDF
from PIL import Image
import numpy as np
import cv2


class InspectionReport(FPDF):
    """Custom PDF generator for inspection results."""
    
    def __init__(self, mode: str, device: str):
        super().__init__()
        self.mode = mode
        self.device = device
        self.report_time = datetime.now()
        
    def header(self):
        """Page header with logo area and report info."""
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Intel Inspection Report', align='C', new_x='LMARGIN', new_y='NEXT')
        self.set_font('Helvetica', '', 10)
        self.cell(0, 5, f'Mode: {self.mode} | Device: {self.device}', align='C', new_x='LMARGIN', new_y='NEXT')
        self.cell(0, 5, f'Generated: {self.report_time.strftime("%Y-%m-%d %H:%M:%S")}', align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(5)
        
    def footer(self):
        """Page footer with page number."""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')
        
    def add_summary_section(self, total_images: int, total_defects: int, total_passed: int):
        """Add batch summary statistics."""
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Batch Summary', new_x='LMARGIN', new_y='NEXT')
        self.ln(2)
        
        # Summary table
        self.set_font('Helvetica', '', 11)
        col_width = 60
        
        self.set_fill_color(230, 230, 230)
        self.cell(col_width, 8, 'Total Images Processed', border=1, fill=True)
        self.cell(col_width, 8, str(total_images), border=1, new_x='LMARGIN', new_y='NEXT')
        
        self.cell(col_width, 8, 'Total Defects Found', border=1, fill=True)
        self.set_text_color(180, 0, 0) if total_defects > 0 else None
        self.cell(col_width, 8, str(total_defects), border=1, new_x='LMARGIN', new_y='NEXT')
        self.set_text_color(0, 0, 0)
        
        self.cell(col_width, 8, 'Total Passed', border=1, fill=True)
        self.set_text_color(0, 128, 0)
        self.cell(col_width, 8, str(total_passed), border=1, new_x='LMARGIN', new_y='NEXT')
        self.set_text_color(0, 0, 0)
        
        self.ln(10)
        
    def add_image_result(self, filename: str, original_img, result_img, 
                         defects: int, passed: int, unit_id: str = "", 
                         comments: str = "", pin_details: list = None):
        """Add a single image result section."""
        # Check if we need new page
        if self.get_y() > 200:
            self.add_page()
            
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, f'Image: {filename}', new_x='LMARGIN', new_y='NEXT')
        
        # Unit ID and comments
        if unit_id:
            self.set_font('Helvetica', '', 10)
            self.cell(0, 6, f'Unit ID: {unit_id}', new_x='LMARGIN', new_y='NEXT')
            
        if comments:
            self.set_font('Helvetica', 'I', 10)
            self.multi_cell(0, 5, f'Comments: {comments}')
            
        self.ln(3)
        
        # Save images to temp files for embedding
        img_width = 85
        
        # Original image
        orig_path = self._save_temp_image(original_img)
        if orig_path:
            self.image(orig_path, x=10, w=img_width)
            
        # Result image  
        result_path = self._save_temp_image(result_img)
        if result_path:
            self.image(result_path, x=105, y=self.get_y() - img_width * 0.75, w=img_width)
            
        self.ln(5)
        
        # Statistics for this image
        self.set_font('Helvetica', '', 10)
        status_text = "PASS" if defects == 0 else "FAIL"
        status_color = (0, 128, 0) if defects == 0 else (180, 0, 0)
        
        self.cell(40, 6, f'Defects: {defects}', border=1)
        self.cell(40, 6, f'Passed: {passed}', border=1)
        self.set_text_color(*status_color)
        self.cell(40, 6, f'Status: {status_text}', border=1, new_x='LMARGIN', new_y='NEXT')
        self.set_text_color(0, 0, 0)
        
        # Pin details for Socket Pin mode (show first 10 defects)
        if pin_details and self.mode == "Socket Pin Defect":
            defect_pins = [p for p in pin_details if p.get('is_defect', False)]
            if defect_pins:
                self.ln(3)
                self.set_font('Helvetica', 'B', 10)
                self.cell(0, 6, f'Defect Pin Details (showing up to 10):', new_x='LMARGIN', new_y='NEXT')
                self.set_font('Helvetica', '', 9)
                
                for i, pin in enumerate(defect_pins[:10]):
                    self.cell(0, 5, f"  Pin #{pin['id']}: Score {pin['score']:.3f}", new_x='LMARGIN', new_y='NEXT')
                    
                if len(defect_pins) > 10:
                    self.cell(0, 5, f"  ... and {len(defect_pins) - 10} more defects", new_x='LMARGIN', new_y='NEXT')
        
        self.ln(10)
        
    def _save_temp_image(self, img, max_size=400) -> str:
        """Save numpy/PIL image to temp file with compression."""
        try:
            # Convert numpy array to PIL Image
            if isinstance(img, np.ndarray):
                # Images from main.py are already RGB format
                pil_img = Image.fromarray(img)
            elif isinstance(img, Image.Image):
                pil_img = img
            else:
                return None
            
            # Resize to reduce file size (maintain aspect ratio)
            width, height = pil_img.size
            if width > max_size or height > max_size:
                ratio = min(max_size / width, max_size / height)
                new_size = (int(width * ratio), int(height * ratio))
                pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                
            # Save as JPEG with compression
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            pil_img.save(temp_file.name, 'JPEG', quality=70, optimize=True)
            return temp_file.name
        except Exception as e:
            print(f"Error saving temp image: {e}")
            return None
            
    def generate(self) -> bytes:
        """Generate PDF and return as bytes."""
        return bytes(self.output())


def generate_report(mode: str, device: str, image_results: list, 
                    total_defects: int, total_passed: int) -> bytes:
    """
    Generate a PDF report from inspection results.
    
    Args:
        mode: Inspection mode ("VCTIM Detection" or "Socket Pin Defect")
        device: Device used (CPU/GPU)
        image_results: List of dicts with keys:
            - filename: str
            - original_img: numpy array
            - result_img: numpy array
            - defects: int
            - passed: int
            - unit_id: str (optional)
            - comments: str (optional)
            - pin_details: list (optional, for Socket Pin mode)
        total_defects: Total defect count across all images
        total_passed: Total passed count across all images
        
    Returns:
        PDF file as bytes
    """
    pdf = InspectionReport(mode=mode, device=device)
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Add summary
    pdf.add_summary_section(
        total_images=len(image_results),
        total_defects=total_defects,
        total_passed=total_passed
    )
    
    # Add each image result
    for result in image_results:
        pdf.add_image_result(
            filename=result.get('filename', 'Unknown'),
            original_img=result.get('original_img'),
            result_img=result.get('result_img'),
            defects=result.get('defects', 0),
            passed=result.get('passed', 0),
            unit_id=result.get('unit_id', ''),
            comments=result.get('comments', ''),
            pin_details=result.get('pin_details', None)
        )
    
    return pdf.generate()
