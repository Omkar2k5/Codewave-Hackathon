# Technology Stack

## Frontend
- **Framework**: Next.js 14 with TypeScript
- **Styling**: Tailwind CSS with custom animations
- **UI Components**: Radix UI primitives with shadcn/ui
- **3D Graphics**: Three.js with React Three Fiber
- **Animations**: Framer Motion and GSAP
- **State Management**: React hooks and context

## Backend
- **Language**: Python 3.x
- **ML/AI**: TensorFlow, OpenCV, Deepsort Crowd Detection, Custom Trained
- **Computer Vision**: HOG descriptors, background subtraction, cascade classifiers
- **Clustering**: scikit-learn (DBSCAN for crowd grouping)
- **Model Formats**: YOLOv8 with .pt weights and Custom Deep Sort Algorithm

## Development Commands

### Frontend
```bash
cd frontend
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
```

### Backend
```bash
cd backend
python -m pip install -r requirements.txt  # Install dependencies
python setup_environment.py                # Setup environment
python enhanced_crowd_detection.py         # Run enhanced detection
python simple_crowd_detection.py           # Run basic detection
python object_tracker.py --video <path>    # Run object tracking
```

## Key Libraries
- **Frontend**: React 18, Next.js 14, Tailwind CSS, Framer Motion, Three.js
- **Backend**: TensorFlow 2.8+, OpenCV 4.5+, NumPy, scikit-learn
- **Development**: TypeScript, ESLint, PostCSS