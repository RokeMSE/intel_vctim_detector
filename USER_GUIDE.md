# 📖 Front-End User Guide: Unified Industrial Inspector

> **Audience:** End-users (operators, technicians, quality engineers) who interact with the web-based graphical user interface (GUI) only.  
> **System Version:** VCTIM Detection Mode  
> **Interface:** Streamlit Web Application (accessed via a browser)

---

## 📋 Table of Contents

1. [Getting Started](#-getting-started)
   - [Accessing the Application](#accessing-the-application)
   - [Interface Overview](#interface-overview)
2. [🔴 Mandatory Functions](#-mandatory-functions)
   - [Step 1: Upload Inspection Images](#step-1-upload-inspection-images)
   - [Step 2: Review Inspection Results](#step-2-review-inspection-results)
   - [Step 3: Interpret the Validation Alerts](#step-3-interpret-the-validation-alerts)
   - [Step 4: Review the Batch Summary](#step-4-review-the-batch-summary)
3. [🟢 Optional Functions](#-optional-functions)
   - [Adjust Detection Confidence](#1-adjust-detection-confidence)
   - [Set Expected BIB Amount](#2-set-expected-bib-amount)
   - [Select Inference Device (CPU / GPU)](#3-select-inference-device-cpu--gpu)
   - [Use Real-time Webcam Capture](#4-use-real-time-webcam-capture)
   - [Add Report Annotations (Unit ID)](#5-add-report-annotations-unit-id)
   - [Scan Barcodes for Unit ID](#6-scan-barcodes-for-unit-id)
   - [Add Comments / Notes](#7-add-comments--notes)
   - [Download Inspection Report (.JPEG)](#8-download-inspection-report-jpeg)
   - [Leverage Smart Caching](#9-leverage-smart-caching)
4. [📊 Understanding the Output](#-understanding-the-output)
   - [Visual Color Legend](#visual-color-legend)
   - [Metrics Explained](#metrics-explained)
   - [Report File Contents](#report-file-contents)
5. [❓ FAQ & Troubleshooting](#-faq--troubleshooting)
6. [⚡ Quick Reference Card](#-quick-reference-card)

---

## 🚀 Getting Started

### Accessing the Application

1. Open any modern web browser (Chrome, Edge, or Firefox recommended).
2. Navigate to the URL provided by your system administrator.  
   - Default local URL: **`http://localhost:8501`**
   - If deployed via Docker: `http://<server-ip>:8501`
3. Wait for the page to fully load. You will see the application title **"Inspection System"** at the top, and a **sidebar** on the left-hand side.

> **Tip:** Bookmark the URL for quick access during your shift.

### Interface Overview

The interface is divided into **two main areas**:

| Area | Location | Purpose |
|------|----------|---------|
| **Sidebar** (Configuration Panel) | Left side of the screen | Contains all settings, file upload, and configuration controls. |
| **Main Content Area** | Center / right of the screen | Displays inspection results, images, metrics, validation alerts, batch summary, and the report download button. |

**Sidebar sections in order (top to bottom):**

1. **System Settings** — Device selection (CPU/GPU)
2. **Configuration** — Detection settings and parameters
3. **VCTIM Settings** — Detection Confidence slider, Expected BIB Amount, Webcam toggle
4. **Upload Images** — File uploader for batch inspection

---

## 🔴 Mandatory Functions

> These steps form the **minimum required workflow** for performing an inspection. You **must** complete these steps every time you run an inspection session.

---

### Step 1: Upload Inspection Images

This is the **primary method** for providing input to the system.

**Where to find it:** Sidebar → bottom section → **"Upload Images (JPG/PNG)"**

**How to do it:**

1. Locate the **"Upload Images (JPG/PNG)"** section at the bottom of the sidebar.
2. Click the **"Browse files"** button.
3. In the file dialog that opens, navigate to the folder containing your inspection images.
4. Select one or more image files. Supported formats are:
   - `.jpg`
   - `.jpeg`
   - `.png`
5. Click **Open** to confirm your selection.

**Alternative method:** You can also **drag and drop** image files directly onto the uploader area in the sidebar.

**What happens next:**
- A **progress bar** will immediately appear in the main content area.
- The system automatically starts processing each uploaded image through the AI model.
- Once complete, results for each image populate the main content area (see Step 2).

> ⚠️ **Important Notes:**
> - If no images are uploaded, the main content area will display: *"Upload one or more images to begin bulk inspection."*
> - The system supports **bulk processing** — you can upload many images at once, and they will be processed sequentially with a progress bar tracking overall batch completion.
> - Image files should be clear, well-lit photographs of the circuit board surface. Poor image quality may affect detection accuracy.

---

### Step 2: Review Inspection Results

Once the AI has finished processing, results are displayed in the main content area as a series of **expandable panels** — one per uploaded image.

**For each image, you will see:**

#### 2a. Expander Panel Header
- Labeled as: **`Image: <filename>`** (e.g., `Image: board_01.jpg`)
- Click the panel header to **expand** or **collapse** it.
- By default, panels are **expanded** after processing.

#### 2b. Side-by-Side Image Comparison
Inside each expanded panel, two images are displayed in adjacent columns:

| Left Column | Right Column |
|-------------|-------------|
| **Original** — The raw image you uploaded, exactly as-is. | **Result** — The AI-annotated image showing detection bounding boxes with confidence scores. |

**Reading the Result Image:**

The AI draws **bounding boxes** (rectangles) around each detected component. Each box includes a label and a confidence score:

- **🟥 Red Box** → **`missing_vctim`** — The AI has identified this location as having a **missing** VCTIM component.
- **🟦 Blue Box** → **`normal`** — The AI has identified this as a **present/normal** VCTIM component.

Each label includes a confidence value between `0.00` and `1.00` (e.g., `missing_vctim 0.87`). Higher values mean the AI is more certain about its detection.

#### 2c. Metrics Panel
Directly below the image pair, two numeric metrics are displayed:

| Metric | Description |
|--------|-------------|
| **Missing** | Count of `missing_vctim` detections found by the AI in this image. Displayed with an inverse (⚠️ warning) color when > 0. |
| **Normal** | Count of `normal` components detected by the AI in this image. |

---

### Step 3: Interpret the Validation Alerts

Directly beneath the metrics for each image, the system performs an **automatic validation check**. It compares the total number of detected items (Missing + Normal) against the **Expected BIB Amount** you set in the sidebar.

| Alert | Appearance | Meaning |
|-------|------------|---------|
| ✅ **Count Matches** (Green box) | `✅ Count Matches: <N>` | The AI found exactly the expected number of components. No anomaly in component count. |
| ⚠️ **Mismatch** (Red box) | `⚠️ MISSING VCTIM! BIB Count Mismatch! Expected <X>, Found <Y>` | The total detected components **do not match** the expected count. This may indicate that the AI missed a detection, or there is a genuine component issue. **Investigate this unit further.** |

> **Why this matters:** Even if zero `missing_vctim` labels are detected, the count validation catches scenarios where the AI may have entirely failed to detect a location — neither as `normal` nor as `missing`. A mismatch is a **red flag** requiring manual review.

---

### Step 4: Review the Batch Summary

After **all** uploaded images have been processed and displayed, scroll to the bottom of the page to find the **Batch Summary** section.

**What it shows:**

| Metric | Description |
|--------|-------------|
| **Total Defects** | The cumulative sum of all **Missing** detections across every image in the current upload batch. |
| **Total Passed** | The cumulative sum of all **Normal** detections across every image in the current upload batch. |

**How to use it:**
- Use this as a **quick dashboard** to assess the overall pass/fail rate for a batch of units without scrolling through each individual result.
- If **Total Defects > 0**, at least one image contained missing components — scroll up to identify which specific image(s) triggered the alert.

---

## 🟢 Optional Functions

> These features allow you to **customize**, **annotate**, and **export** your inspection results. None of these steps are required for basic inspection, but they enhance traceability and operational flexibility.

---

### 1. Adjust Detection Confidence

**Where:** Sidebar → **VCTIM Settings** → **"Detection Confidence"** slider

**What it does:** Controls the minimum confidence score the AI requires before it reports a detection. This directly affects sensitivity.

| Setting | Range | Default |
|---------|-------|---------|
| Detection Confidence | `0.00` to `1.00` | **`0.25`** |

**How to adjust:**
1. Click and drag the slider to your desired value.
2. Results will **automatically refresh** with the new threshold applied.

**Interpretation guide:**

| Direction | Effect | When to use |
|-----------|--------|-------------|
| **Increase** (e.g., `0.4` → `0.6`) | **More strict.** Reduces false positives (fewer spurious detections), but may miss some genuine defects. | When the AI is flagging too many false alarms on good units. |
| **Decrease** (e.g., `0.25` → `0.1`) | **More sensitive.** Catches more potential defects, but may increase false positives. | When the AI is failing to flag known bad units. |

> **Recommendation:** The default value of `0.25` is optimized for general use. Adjust only if you notice consistent false positives or missed detections across multiple units.

---

### 2. Set Expected BIB Amount

**Where:** Sidebar → **VCTIM Settings** → **"Expected BIB Amount"** number input

**What it does:** Defines the total number of VCTIM positions (both Normal + Missing) expected on each unit. The system uses this number to validate the AI's results.

| Setting | Range | Default |
|---------|-------|---------|
| Expected BIB Amount | `1` to `20` | **`10`** |

**How to adjust:**
1. Use the **`+`** and **`-`** buttons, or type a value directly into the input field.
2. The validation alerts (✅ / ⚠️) will **immediately recalculate** using the new value — even for cached results — without re-running the AI inference.

> **Important:** Set this value **before** uploading images, or update it at any time to retroactively re-validate previously processed results. No re-inference is needed.

**Example:**
- You are inspecting a board that should have **12** VCTIM positions.
- Set **Expected BIB Amount = 12**.
- If the AI detects 10 Normal + 1 Missing = 11 total, the system will flag: `⚠️ Expected 12, Found 11`.

---

### 3. Select Inference Device (CPU / GPU)

**Where:** Sidebar → **System Settings** → **"Inference Device"** radio buttons

**Options:**

| Option | Description |
|--------|-------------|
| **CPU** *(default)* | Uses the central processor. Slower but always available. Uses OpenVINO optimization for acceleration. |
| **GPU** | Uses NVIDIA CUDA-capable GPU for faster inference. Requires a compatible GPU and CUDA drivers. |

**How to use:**
1. Select either **CPU** or **GPU** using the radio button.
2. If you select **GPU** but no compatible GPU is detected, the system will display a **warning message** in the sidebar: *"GPU selected but CUDA not available. Falling back to CPU."* and automatically revert to CPU processing.

> **Note:** GPU processing is significantly faster (5–10×) for large batches. If your workstation has a compatible GPU, select this option for improved throughput.

---

### 4. Use Real-time Webcam Capture

**Where:** Sidebar → **VCTIM Settings** → **"Use Webcam (Real-time)"** checkbox

Instead of uploading pre-captured images, you can use a connected camera for **live, single-shot inspections**.

**How to use:**

1. Connect a webcam or USB camera to your workstation.
2. Check the **"Use Webcam (Real-time)"** checkbox in the sidebar.
3. The main content area will display a **camera viewfinder** with a "Take a snapshot or use live view" label.
4. Position the circuit board in the camera's field of view with adequate lighting.
5. Click the **camera button** (📷) to capture a snapshot.
6. The system will **immediately process** the captured image and display:
   - The annotated result image with bounding boxes.
   - **Missing** and **Normal** count metrics.
   - The validation alert (✅ or ⚠️) comparing against the Expected BIB Amount.

> **⚠️ Limitations:**
> - Webcam mode processes **one image at a time** (no bulk processing).
> - Results from webcam captures are **not included** in the batch summary or downloadable report. They are for immediate, live feedback only.
> - Image quality depends heavily on camera resolution and lighting conditions. For production-grade inspection, pre-captured high-resolution images are recommended.

---

### 5. Add Report Annotations (Unit ID)

**Where:** Inside each image's expanded result panel → **"📝 Report Annotations (Optional)"** section

**What it does:** Allows you to tag each inspected image with a unique identifier for traceability.

**How to use:**

1. After the image has been processed, look for the **"📝 Report Annotations (Optional)"** section below the validation alert.
2. In the **"Unit ID"** text field, type the unit's tracking number (e.g., `UNIT-001`, `SN-2024-0099`).
3. The ID is automatically saved as you type and will be included in the downloadable report.

> **Tip:** The Unit ID also determines the **report filename**. If a Unit ID is provided, the downloaded report file will be named `<UnitID>_<timestamp>.jpg` instead of a generic name. This makes filing and searching through inspection records much easier.

---

### 6. Scan Barcodes for Unit ID

**Where:** Inside each image's expanded result panel → **"📷 Scan"** button (next to the Unit ID field)

Instead of manually typing the Unit ID, you can use a **physical barcode scanner** to populate it automatically.

**How to use:**

1. Click the **"📷 Scan"** button next to the Unit ID input field.
2. A **popup dialog** titled **"Scan Unit ID"** will appear.
3. The dialog contains a text input field with the message: *"Please scan the barcode now..."*
4. Click inside the scanner input field in the dialog to ensure it is active/focused.
5. Point your **physical barcode scanner** at the unit's barcode label and trigger the scan.
6. The scanned value will automatically populate the input field.
7. The dialog will **close automatically** and the Unit ID field will be updated with the scanned value.

> **Requirements:**
> - A USB or Bluetooth barcode scanner that emulates keyboard input (HID mode).
> - The scanner must be connected and recognized by your workstation's operating system.

---

### 7. Add Comments / Notes

**Where:** Inside each image's expanded result panel → **"Comments"** text area

**What it does:** Provides a free-text field for operator notes, observations, or disposition decisions.

**How to use:**

1. Locate the **"Comments"** text area below the Unit ID field in each image's result panel.
2. Click inside the text area and type your notes.
3. Examples of useful comments:
   - `"Visual confirmation: PASS — AI false alarm on corner reflection"`
   - `"FAIL — confirmed 2 missing VCTIMs, escalated to supervisor"`
   - `"Re-inspect: Image was blurry, retaking photo"`
4. Comments are automatically saved and included in the downloadable report.

---

### 8. Download Inspection Report (.JPEG)

**Where:** Bottom of the main content area → **"📄 Generate Report"** section

After processing images (and optionally adding annotations), you can export a visual summary report of the entire inspection session.

**How to use:**

1. Scroll to the very bottom of the page.
2. Locate the **"📄 Generate Report"** section.
3. Click the **"🖨️ Download Report (.jpeg)"** button (displayed in blue/primary style).
4. Your browser will automatically download a `.jpg` file.

**Report contents:**

The downloaded JPEG report image includes:

| Section | Content |
|---------|---------|
| **Header** | "Intel Inspection Report", Mode, Device, and Generation Timestamp |
| **Batch Summary Table** | Total Images processed, Total Defects, Total Passed |
| **Per-Image Results** | For each image: filename, Unit ID (if provided), Comments (if provided), side-by-side Original vs. Result images, Defects/Passed/Status (PASS or FAIL), and Mismatch warnings |

**Report filename logic:**

| Condition | Filename |
|-----------|----------|
| Unit ID **is** provided (for at least one image) | `<UnitID>_<YYYYMMDD_HHMMSS>.jpg` |
| Unit ID **not** provided | `inspection_report_VCTIM_Detection_<YYYYMMDD_HHMMSS>.jpg` |

> **Tip:** The report uses the **first** Unit ID found among all processed images for the filename. Assign the Unit ID to the first image if you want predictable naming.

---

### 9. Leverage Smart Caching

**What it does:** The system automatically caches inference results. If you re-open the same set of uploaded images without changing detection settings, the system will **instantly display** previously computed results instead of re-running the AI.

**How it works:**

- The cache is active when **all** of the following conditions are unchanged:
  1. The same set of uploaded files (matched by filename + file size)
  2. The same detection mode (VCTIM Detection)
  3. The same inference device (CPU/GPU)
  4. The same detection confidence threshold

- When cached results are loaded, a **blue info banner** appears: `📋 Results loaded from cache. Upload new files or change settings to re-run inspection.`

**What you can still do with cached results:**
- ✅ Change the **Expected BIB Amount** — validation alerts recalculate in real-time without re-inference.
- ✅ Edit **Unit IDs** and **Comments** — annotations are independent of the cache.
- ✅ **Download the report** — the report reflects the latest annotations.

**How to force a fresh re-inspection:**
- Change the **Detection Confidence** slider to any different value (even slightly), OR
- Upload a different set of image files, OR
- Change the **Inference Device** (CPU ↔ GPU).

> **Why this is useful:** If you need to adjust annotations or re-validate with a different Expected BIB count, caching eliminates the need to wait for AI re-processing, saving significant time during shifts.

---

## 📊 Understanding the Output

### Visual Color Legend

| Element | Color | Meaning |
|---------|-------|---------|
| Bounding Box | 🟦 **Blue** | **Normal** component detected |
| Bounding Box | 🟥 **Red** | **Missing** component detected |
| Box Label | White text on colored background | Class name + confidence score |
| Validation Alert | 🟩 **Green box** | Count matches expected BIB amount |
| Validation Alert | 🟥 **Red box** | Count mismatch — investigate |
| Cache Banner | 🟦 **Blue info box** | Results loaded from cache |
| GPU Warning | 🟨 **Yellow warning** | GPU unavailable, using CPU fallback |

### Metrics Explained

| Metric | Where it appears | What it counts |
|--------|-----------------|----------------|
| **Missing** (per image) | Inside each image expander | Number of `missing_vctim` detections in that specific image |
| **Normal** (per image) | Inside each image expander | Number of `normal` detections in that specific image |
| **Total Defects** (batch) | Batch Summary section | Sum of all Missing counts across all images |
| **Total Passed** (batch) | Batch Summary section | Sum of all Normal counts across all images |

### Report File Contents

The downloadable JPEG report is a **single image file** (800px wide) formatted as a vertical document with the following structure:

```
┌──────────────────────────────────────┐
│   Intel Inspection Report  (Header)  │
│   Mode | Device | Timestamp          │
├──────────────────────────────────────┤
│   Batch Summary Table                │
│   Total Images | Defects | Passed    │
├──────────────────────────────────────┤
│   Image 1: <filename>               │
│   Unit ID: <if provided>            │
│   Comments: <if provided>           │
│   ┌────────────┐  ┌────────────┐    │
│   │  Original  │  │   Result   │    │
│   └────────────┘  └────────────┘    │
│   Defects: X | Passed: Y | PASS/FAIL│
│   ⚠️ Mismatch warning (if any)      │
├──────────────────────────────────────┤
│   Image 2: ...                       │
│   (repeats for each image)           │
└──────────────────────────────────────┘
```

---

## ❓ FAQ & Troubleshooting

### General Questions

**Q: What image formats are supported?**  
A: `.jpg`, `.jpeg`, and `.png` files are supported.

**Q: Is there a limit on how many images I can upload at once?**  
A: There is no hard limit, but processing time scales linearly with the number of images. For very large batches (50+ images), processing may take several minutes on CPU.

**Q: Can I inspect images from different products in the same batch?**  
A: Yes, but note that the Expected BIB Amount applies to **all** images in the batch. If different products have different BIB counts, inspect them in separate batches for accurate validation.

**Q: Do I need to keep the browser tab open during processing?**  
A: Yes. Closing or refreshing the browser tab during processing will **cancel** the current inspection. You will need to re-upload the images.

### Settings & Configuration

**Q: I changed the detection confidence but results didn't update.**  
A: If files were already processed, the system may be serving cached results for the *previous* threshold value. Changing the confidence slider **does** trigger a fresh re-inference because the configuration signature changes. If you still see stale results, try re-uploading the files.

**Q: What value should I set for 'Expected BIB Amount'?**  
A: Set this to the total number of VCTIM positions on the board you are inspecting — count both locations that should have components AND any known empty locations. Consult your product specification sheet for the correct value.

**Q: The sidebar shows "GPU selected but CUDA not available." What should I do?**  
A: This means your workstation either does not have a compatible NVIDIA GPU, or the CUDA drivers are not installed. The system automatically falls back to CPU. Contact your IT administrator if GPU acceleration is expected to be available.

### Results & Validation

**Q: I see a ⚠️ mismatch alert, but the board looks fine. What happened?**  
A: Possible causes:
1. **Wrong Expected BIB Amount** — Verify the number matches your product spec.
2. **Obscured components** — Part of the board may be out-of-frame or occluded.
3. **Image quality** — Poor lighting, blur, or glare can cause missed detections.
4. **Low confidence threshold** — If set too high, detections may be filtered out.

**Q: The AI detected a 'missing_vctim' where a normal component is clearly present.**  
A: This is a **false positive**. Try:
1. Increasing the **Detection Confidence** slider (e.g., from `0.25` to `0.35`).
2. Ensuring the image has good lighting and is in-focus.
3. Noting it in the **Comments** field as `"False positive — component visually confirmed present"`.

**Q: Can I re-run inspection on just one image from the batch?**  
A: No. The system processes all uploaded images as a single batch. To re-inspect a specific image, upload only that image by itself.

### Reports

**Q: Where is the report saved?**  
A: The report is downloaded to your browser's default download folder (usually `C:\Users\<username>\Downloads\` on Windows).

**Q: Can I get a PDF report instead of JPEG?**  
A: The current version generates JPEG reports only. PDF generation capability exists in the backend but is not exposed in the current interface.

**Q: The report shows "FAIL" for an image, but I annotated it as OK in my comments.**  
A: The PASS/FAIL status on the report is determined **purely by the AI's detection results** (FAIL if any `missing_vctim` is detected, PASS if zero). Your comments are a separate field for human notes and do not override the AI's classification.

---

## ⚡ Quick Reference Card

> **Print this section and post it at your workstation for quick reference.**

### Minimum Workflow (Mandatory Steps)

```
1. Open browser → Go to http://localhost:8501
2. Sidebar → "Upload Images (JPG/PNG)" → Select files
3. Wait for progress bar to complete
4. Review results: Original vs. Result images
5. Check validation alerts: ✅ or ⚠️
6. Review Batch Summary at bottom
```

### Optional Enhancements

```
• Sidebar → Adjust "Detection Confidence" (0.0 - 1.0)
• Sidebar → Set "Expected BIB Amount" (1 - 20)
• Sidebar → Select CPU or GPU
• Sidebar → ☑ "Use Webcam (Real-time)" for live inspection
• Per-image → Enter "Unit ID" or click "📷 Scan"
• Per-image → Add "Comments"
• Bottom → Click "🖨️ Download Report (.jpeg)"
```

### Color Legend

```
🟥 Red Box    = Missing VCTIM (defect)
🟦 Blue Box   = Normal VCTIM (good)
🟩 Green Alert = Count matches expected
🟥 Red Alert   = Count mismatch — investigate!
```

### Detection Confidence Cheat Sheet

```
← 0.0 -------- 0.25 (Default) -------- 1.0 →
   More sensitive                More strict
   More false positives          May miss defects
```
