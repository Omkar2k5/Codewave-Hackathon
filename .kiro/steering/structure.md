# Project Structure

## Root Organization
```
├── frontend/           # Next.js web application
├── backend/            # Python ML/AI processing
├── .kiro/              # Kiro IDE configuration and steering
├── .vscode/            # VS Code settings
└── .zencoder/          # Additional tooling configuration
```

## Frontend Structure (`frontend/`)
```
├── app/                # Next.js App Router pages
├── components/         # Reusable React components
├── hooks/              # Custom React hooks
├── lib/                # Utility functions and configurations
├── public/             # Static assets
├── styles/             # Global styles and Tailwind config
├── package.json        # Dependencies and scripts
└── next.config.mjs     # Next.js configuration
```

## Backend Structure (`backend/`)
```
├── core/               # Core ML/AI modules
├── data/               # Training data and model weights
├── deep_sort/          # DeepSORT tracking implementation
├── exported_models/    # Exported model files
├── logic/              # Business logic and algorithms
├── simple_deployment/  # Deployment configurations
├── tools/              # Utility scripts and tools
├── Report/             # Documentation and reports
├── requirements.txt    # Python dependencies
└── *.py               # Main detection and processing scripts
```

## Key Files
- **Frontend Entry**: `frontend/app/page.tsx` (landing page)
- **Main Components**: `frontend/components/landing-page.tsx`, `navbar.tsx`
- **Backend Main**: `backend/enhanced_crowd_detection.py` (primary detection)
- **Simple Detection**: `backend/simple_crowd_detection.py` (basic implementation)
- **Object Tracking**: `backend/object_tracker.py` (YOLO + DeepSORT)

## Naming Conventions
- **Frontend**: kebab-case for files, PascalCase for components
- **Backend**: snake_case for Python files and functions
- **Components**: Use descriptive names (e.g., `landing-page.tsx`, `crowd-detection.py`)
- **Directories**: lowercase with hyphens or underscores as appropriate

## Configuration Files
- `frontend/tailwind.config.js` - Tailwind CSS configuration
- `frontend/next.config.mjs` - Next.js build configuration
- `backend/requirements.txt` - Python package dependencies
- `.gitignore` - Version control exclusions