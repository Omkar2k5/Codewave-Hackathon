---
description: Repository Information Overview
alwaysApply: true
---

# Drishti (Real-Time-Crowd-Detection) Information

## Summary
Drishti is an intelligent system designed to detect, prevent, and manage stampede-like situations in crowded environments. Using real-time data, predictive analytics, and alert mechanisms, it empowers authorities and event organizers to maintain crowd safety and respond swiftly to potential risks.

## Structure
- **backend/**: Python-based crowd detection and analysis system
- **frontend/**: Next.js web application for visualization and monitoring
- **backend/deep_sort/**: Object tracking implementation
- **backend/core/**: Core ML model implementation
- **backend/exported_models/**: Exported model files for deployment
- **backend/simple_deployment/**: Simplified deployment package

## Projects

### Backend (ML System)
**Configuration File**: requirements.txt

#### Language & Runtime
**Language**: Python
**Version**: Python 3.x
**Build System**: Native Python
**Package Manager**: pip

#### Dependencies
**Main Dependencies**:
- tensorflow>=2.8.0
- opencv-python>=4.5.0
- numpy>=1.21.0
- pillow>=8.3.0
- easydict>=1.9
- absl-py>=1.0.0
- tqdm>=4.62.0

#### Build & Installation
```bash
pip install -r backend/requirements.txt
python backend/setup_environment.py
```

#### Main Files
**Entry Points**:
- backend/yolo_crowd_detection.py
- backend/advanced_crowd_detection.py
- backend/enhanced_crowd_detection.py
- backend/simple_deployment.py

**Model Files**:
- backend/yolov8m.pt (excluded from git)
- backend/data/yolov4.weights (excluded from git)

### Frontend (Dashboard)
**Configuration File**: package.json

#### Language & Runtime
**Language**: TypeScript/JavaScript
**Version**: Node.js
**Build System**: Next.js
**Package Manager**: npm/pnpm

#### Dependencies
**Main Dependencies**:
- next: 14.2.30
- react: 18.2.0
- react-dom: 18.2.0
- framer-motion
- tailwindcss
- shadcn components
- three.js: 0.178.0

**Development Dependencies**:
- typescript: ^5
- postcss: ^8.5
- tailwindcss: ^3.4.17

#### Build & Installation
```bash
cd frontend
npm install
npm run dev   # Development
npm run build # Production build
```

#### Main Files
**Entry Points**:
- frontend/app/page.tsx
- frontend/app/dashboard/
- frontend/components/dashboard-page.tsx
- frontend/components/landing-page.tsx