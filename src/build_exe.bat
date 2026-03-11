@echo off
echo ==========================================
echo    Building Streamlit App for Windows     
echo ==========================================

echo Installing dependencies...
pip install pyinstaller fpdf2

echo Starting build process...
pyinstaller --noconfirm --onefile --windowed --name "InspectionSystem_exeonly" ^
    --add-data "main.py;." ^
    --add-data "report_generator.py;." ^
    --add-data ".env;." ^
    --add-data "models;models" ^
    --copy-metadata streamlit ^
    --collect-all streamlit ^
    --collect-all ultralytics ^
    --hidden-import "albumentations" ^
    --hidden-import "ultralytics" ^
    --hidden-import "dotenv" ^
    --hidden-import "fpdf" ^
    --collect-all "cv2" ^
    --exclude-module "langchain" ^
    run_app.py

echo ==========================================
echo Build complete! 
echo Your offline application is located in the 'dist\InspectionSystem' folder.
echo To run the app, double-click 'InspectionSystem.exe' inside that folder.
echo ==========================================
pause
