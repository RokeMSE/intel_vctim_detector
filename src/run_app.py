import os
import sys
import streamlit.web.cli as stcli

def resolve_path(path):
    # PyInstaller creates a temp folder and stores path in _MEIPASS if using --onefile
    # But since we use --onedir, our assets are in the same folder as the exe.
    if getattr(sys, 'frozen', False):
        application_path = sys._MEIPASS
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(application_path, path)

if __name__ == "__main__":
    # Streamlit requires the exact path to main.py
    main_script_path = resolve_path("main.py")
    
    sys.argv = [
        "streamlit",
        "run",
        main_script_path,
        "--global.developmentMode=false",
    ]
    
    sys.exit(stcli.main())
