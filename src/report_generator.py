import io
import tempfile
from datetime import datetime
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os


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


def generate_jpeg_report(mode: str, device: str, image_results: list, 
                         total_defects: int, total_passed: int, expected_bib: int = None) -> bytes:
    """
    Generate a JPEG report from inspection results, mimicking the PDF layout.
    Returns bytes of the generated JPEG image.
    """
    
    # --- Config & Helpers ---
    width = 800  # Fixed width for the report
    bg_color = (255, 255, 255)
    text_color = (0, 0, 0)
    
    try:
        # Try to load Arial (standard on Windows/Linux usually)
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_med = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
        font_bold = ImageFont.truetype("arialbd.ttf", 12)  # bold if available
    except IOError:
        # Fallback to default
        font_large = ImageFont.load_default()
        font_med = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_bold = ImageFont.load_default()

    def create_text_line(text, font, color=text_color, padding=10):
        # Calculate size
        dummy_img = Image.new('RGB', (10, 10))
        d = ImageDraw.Draw(dummy_img)
        bbox = d.textbbox((0, 0), text, font=font)
        h = bbox[3] - bbox[1] + padding * 2
        
        img = Image.new('RGB', (width, h), bg_color)
        d = ImageDraw.Draw(img)
        # Center text
        text_w = bbox[2] - bbox[0]
        x = (width - text_w) // 2
        d.text((x, padding), text, font=font, fill=color)
        return img

    sections = []
    
    # --- 1. Header ---
    sections.append(create_text_line("Intel Inspection Report", font_large, padding=15))
    sections.append(create_text_line(f"Mode: {mode} | Device: {device}", font_med))
    sections.append(create_text_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", font_small))
    
    # Separator
    sep = Image.new('RGB', (width, 2), (0, 0, 0))
    sections.append(sep)
    
    # --- 2. Summary Section ---
    summary_h = 120
    summary_img = Image.new('RGB', (width, summary_h), bg_color)
    d = ImageDraw.Draw(summary_img)
    
    d.text((20, 20), "Batch Summary", font=font_med, fill=text_color)
    
    # Draw simple table
    start_y = 50
    col_w = 200
    row_h = 40
    
    headers = ["Total Images", "Total Defects", "Total Passed"]
    values = [str(len(image_results)), str(total_defects), str(total_passed)]
    colors = [text_color, (200, 0, 0) if total_defects > 0 else text_color, (0, 128, 0)]
    
    for i, (h, v, c) in enumerate(zip(headers, values, colors)):
        x = 20 + i * col_w
        # Header box
        d.rectangle([x, start_y, x + col_w - 5, start_y + row_h], outline=(100, 100, 100), width=1)
        d.text((x + 10, start_y + 10), h, font=font_bold, fill=text_color)
        
        # Value
        d.text((x + 10 + 120, start_y + 10), v, font=font_bold, fill=c)
        
    sections.append(summary_img)
    sections.append(sep)
    
    # --- 3. Image Results ---
    for result in image_results:
        # Create a container for this result
        # Estimate height: Title (30) + Images (300) + Stats (40) + Comments (variable)
        
        # We'll build it vertically
        res_sections = []
        
        # Title
        title_img = Image.new('RGB', (width, 40), (240, 240, 240))
        d_title = ImageDraw.Draw(title_img)
        d_title.text((20, 10), f"Image: {result.get('filename', 'Unknown')}", font=font_bold, fill=text_color)
        res_sections.append(title_img)
        
        # Info (Unit ID, Comments)
        info_h = 10
        info_lines = []
        if result.get('unit_id'):
            info_lines.append(f"Unit ID: {result['unit_id']}")
        if result.get('comments'):
            info_lines.append(f"Comments: {result['comments']}")
            
        if info_lines:
            info_h += len(info_lines) * 20 + 10
            info_img = Image.new('RGB', (width, info_h), bg_color)
            d_info = ImageDraw.Draw(info_img)
            y_txt = 10
            for line in info_lines:
                d_info.text((20, y_txt), line, font=font_small, fill=text_color)
                y_txt += 20
            res_sections.append(info_img)
            
        # Images (Original vs Result)
        # Resize to fit side-by-side
        img_w = (width - 60) // 2
        
        orig = result.get('original_img')
        res = result.get('result_img')
        
        if orig is not None and res is not None:
             # Convert valid numpy arrays to PIL if needed (usually they are numpy RGB from main)
            if isinstance(orig, np.ndarray):
                pil_orig = Image.fromarray(orig)
            else:
                pil_orig = orig
                
            if isinstance(res, np.ndarray):
                pil_res = Image.fromarray(res)
            else:
                pil_res = res
                
            # Resize keeping aspect ratio
            def resize_contain(img, target_w):
                ratio = target_w / img.width
                new_h = int(img.height * ratio)
                return img.resize((target_w, new_h), Image.Resampling.LANCZOS)
            
            pil_orig = resize_contain(pil_orig, img_w)
            pil_res = resize_contain(pil_res, img_w)
            
            max_h = max(pil_orig.height, pil_res.height)
            
            imgs_container = Image.new('RGB', (width, max_h + 20), bg_color)
            imgs_container.paste(pil_orig, (20, 10))
            imgs_container.paste(pil_res, (width // 2 + 10, 10))
            
            res_sections.append(imgs_container)
            
        # Stats
        stats_h = 50
        stats_img = Image.new('RGB', (width, stats_h), bg_color)
        d_stats = ImageDraw.Draw(stats_img)
        
        defects = result.get('defects', 0)
        passed = result.get('passed', 0)
        status = "PASS" if defects == 0 else "FAIL"
        color = (0, 128, 0) if defects == 0 else (180, 0, 0)
        
        stats_text = f"Defects: {defects}   |   Passed: {passed}   |   Status: {status}"
        d_stats.text((20, 10), stats_text, font=font_bold, fill=color)
        
        # VALIDATION ERROR CHECK
        if expected_bib is not None:
            total_found = defects + passed
            if total_found != expected_bib:
                error_text = f"MISSING VCTIM! DUT COUNT MISMATCH: Expected {expected_bib}, Found {total_found}"
                # Draw error text in red below stats
                d_stats.text((20, 30), error_text, font=font_bold, fill=(255, 0, 0))
        
        res_sections.append(stats_img)
        
        # Combine sections for this result
        total_h = sum(img.height for img in res_sections)
        result_final = Image.new('RGB', (width, total_h), bg_color)
        y_off = 0
        for s in res_sections:
            result_final.paste(s, (0, y_off))
            y_off += s.height
            
        sections.append(result_final)
        
        # Add thin separator betwen images
        sections.append(Image.new('RGB', (width, 1), (200, 200, 200)))

    # --- Combine All ---
    final_height = sum(img.height for img in sections)
    final_report = Image.new('RGB', (width, final_height), bg_color)
    
    current_y = 0
    for section in sections:
        final_report.paste(section, (0, current_y))
        current_y += section.height
        
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    final_report.save(img_byte_arr, format='JPEG', quality=85)
    return img_byte_arr.getvalue()
