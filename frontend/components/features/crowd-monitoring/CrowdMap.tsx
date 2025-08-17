"use client"

import React, { useState, useEffect, useCallback } from 'react';
import {
    GoogleMap,
    useJsApiLoader,
    HeatmapLayerF,
    MarkerF,
    InfoWindowF,
    CircleF,
    PolygonF,
    OverlayViewF
} from '@react-google-maps/api';

// --- TYPES ---
interface Camera {
    id: string;
    name: string;
    lat: number;
    lng: number;
    status: string;
    fov: number; // Field of view in degrees
    direction: number; // Direction in degrees (0-360)
    fovRadius: number; // FOV triangle radius in meters
}

interface CrowdStats {
    totalPeople: number;
    activeCameras: number;
}

interface CameraCoverageCircle {
    center: { lat: number; lng: number };
    radius: number;
}

interface FOVTriangle {
    cameraId: string;
    points: google.maps.LatLng[];
    center: google.maps.LatLng;
    rotation: number;
    isDragging: boolean;
    isRotating: boolean;
}

interface FOVControlsProps {
    camera: Camera;
    onUpdate: (updates: Partial<Camera>) => void;
    onClose: () => void;
}

interface CrowdMapProps {
    cameraPlacementMode: boolean;
    onCloseCameraPlacementMode: () => void;
    mapType: 'heatmap' | 'escape-routes';
    selectedCameraPositions: Camera[];
    onCameraPositionsUpdate: (cameras: Camera[]) => void;
    cameraCoverageCircle: CameraCoverageCircle | null;
    onCameraCoverageUpdate: (coverage: CameraCoverageCircle | null) => void;
}

// --- STYLES & CONSTANTS ---
const containerStyle = {
  width: '100%',
  height: '100%'
};

const center = {
  lat: 18.5204,
  lng: 73.8567
};

// FOV triangle constants
const DEFAULT_FOV = 90; // degrees
const DEFAULT_FOV_RADIUS = 100; // meters
const FOV_TRIANGLE_COLOR = '#4ECDC4';
const FOV_TRIANGLE_BORDER_COLOR = '#ffffff';

// Dark mode map options
const darkMapOptions = {
  mapTypeControl: false,
  streetViewControl: false,
  fullscreenControl: false,
  backgroundColor: '#1a1a1a',
  styles: [
    { elementType: 'geometry', stylers: [{ color: '#242f3e' }] },
    { elementType: 'labels.text.stroke', stylers: [{ color: '#242f3e' }] },
    { elementType: 'labels.text.fill', stylers: [{ color: '#746855' }] },
    {
      featureType: 'administrative.locality',
      elementType: 'labels.text.fill',
      stylers: [{ color: '#d59563' }]
    },
    {
      featureType: 'poi',
      elementType: 'labels.text.fill',
      stylers: [{ color: '#d59563' }]
    },
    {
      featureType: 'poi.park',
      elementType: 'geometry',
      stylers: [{ color: '#263c3f' }]
    },
    {
      featureType: 'poi.park',
      elementType: 'labels.text.fill',
      stylers: [{ color: '#6b9a76' }]
    },
    {
      featureType: 'road',
      elementType: 'geometry',
      stylers: [{ color: '#38414e' }]
    },
    {
      featureType: 'road',
      elementType: 'geometry.stroke',
      stylers: [{ color: '#212a37' }]
    },
    {
      featureType: 'road',
      elementType: 'labels.text.fill',
      stylers: [{ color: '#9ca5b3' }]
    },
    {
      featureType: 'road.highway',
      elementType: 'geometry',
      stylers: [{ color: '#746855' }]
    },
    {
      featureType: 'road.highway',
      elementType: 'geometry.stroke',
      stylers: [{ color: '#1f2835' }]
    },
    {
      featureType: 'road.highway',
      elementType: 'labels.text.fill',
      stylers: [{ color: '#f3d19c' }]
    },
    {
      featureType: 'transit',
      elementType: 'geometry',
      stylers: [{ color: '#2f3948' }]
    },
    {
      featureType: 'transit.station',
      elementType: 'labels.text.fill',
      stylers: [{ color: '#d59563' }]
    },
    {
      featureType: 'water',
      elementType: 'geometry',
      stylers: [{ color: '#17263c' }]
    },
    {
      featureType: 'water',
      elementType: 'labels.text.fill',
      stylers: [{ color: '#515c6d' }]
    },
    {
      featureType: 'water',
      elementType: 'labels.text.stroke',
      stylers: [{ color: '#17263c' }]
    }
  ]
};

// --- MAIN COMPONENT ---
export default function CrowdMap({ 
    cameraPlacementMode, 
    onCloseCameraPlacementMode, 
    mapType,
    selectedCameraPositions,
    onCameraPositionsUpdate,
    cameraCoverageCircle,
    onCameraCoverageUpdate
}: CrowdMapProps) {
    // --- STATE MANAGEMENT ---
    const [map, setMap] = useState<google.maps.Map | null>(null);
    const [heatMapData, setHeatMapData] = useState<google.maps.LatLng[]>([]);
    const [selectedCamera, setSelectedCamera] = useState<Camera | null>(null);
    const [crowdStats, setCrowdStats] = useState<CrowdStats>({
        totalPeople: 0,
        activeCameras: 0
    });
    
    // Camera placement mode state
    const [cameraPlacementMap, setCameraPlacementMap] = useState<google.maps.Map | null>(null);
    
    // FOV triangles state
    const [fovTriangles, setFovTriangles] = useState<FOVTriangle[]>([]);
    const [selectedFOVTriangle, setSelectedFOVTriangle] = useState<string | null>(null);
    
    // Shared zoom state for both maps
    const [sharedZoom, setSharedZoom] = useState(13);
    const [sharedCenter, setSharedCenter] = useState(center);

    // --- GOOGLE MAPS API LOADER ---
    const { isLoaded } = useJsApiLoader({
        id: 'google-map-script',
        googleMapsApiKey: "AIzaSyA_sHaZiCurw9zPjUIBcmtuYrte90UHX-g", // Replace with your API key
        libraries: ['visualization', 'geometry', 'places'],
    });

    // --- FOV TRIANGLE FUNCTIONS ---
    const calculateFOVTriangle = useCallback((camera: Camera): google.maps.LatLng[] => {
        if (!isLoaded) return [];
        
        const center = new window.google.maps.LatLng(camera.lat, camera.lng);
        const halfFOV = camera.fov / 2;
        const startAngle = camera.direction - halfFOV;
        const endAngle = camera.direction + halfFOV;
        
        // Convert meters to degrees (approximate)
        const radiusInDegrees = camera.fovRadius / 111000; // 1 degree ≈ 111km
        
        const points: google.maps.LatLng[] = [center];
        
        // Calculate triangle points
        for (let angle = startAngle; angle <= endAngle; angle += 10) {
            const rad = (angle * Math.PI) / 180;
            const lat = center.lat() + radiusInDegrees * Math.cos(rad);
            const lng = center.lng() + radiusInDegrees * Math.sin(rad);
            points.push(new window.google.maps.LatLng(lat, lng));
        }
        
        return points;
    }, [isLoaded]);

    const updateFOVTriangle = useCallback((cameraId: string, updates: Partial<FOVTriangle>) => {
        setFovTriangles(prev => prev.map(triangle => 
            triangle.cameraId === cameraId 
                ? { ...triangle, ...updates }
                : triangle
        ));
    }, []);

    const createFOVTriangle = useCallback((camera: Camera): FOVTriangle => {
        const points = calculateFOVTriangle(camera);
        return {
            cameraId: camera.id,
            points,
            center: new window.google.maps.LatLng(camera.lat, camera.lng),
            rotation: camera.direction,
            isDragging: false,
            isRotating: false
        };
    }, [calculateFOVTriangle]);

    // Update FOV triangles when cameras change
    useEffect(() => {
        const newTriangles = selectedCameraPositions.map(camera => createFOVTriangle(camera));
        setFovTriangles(newTriangles);
    }, [selectedCameraPositions, createFOVTriangle]);

    // --- FOV INTERACTION HANDLERS ---
    const onFOVTriangleClick = useCallback((cameraId: string) => {
        setSelectedFOVTriangle(cameraId);
    }, []);

    const onFOVTriangleDragStart = useCallback((cameraId: string) => {
        updateFOVTriangle(cameraId, { isDragging: true });
    }, [updateFOVTriangle]);

    const onFOVTriangleDragEnd = useCallback((cameraId: string, newCenter: google.maps.LatLng) => {
        updateFOVTriangle(cameraId, { 
            isDragging: false, 
            center: newCenter 
        });
        
        // Update camera position
        const updatedCameras = selectedCameraPositions.map(camera => 
            camera.id === cameraId 
                ? { ...camera, lat: newCenter.lat(), lng: newCenter.lng() }
                : camera
        );
        onCameraPositionsUpdate(updatedCameras);
    }, [updateFOVTriangle, selectedCameraPositions, onCameraPositionsUpdate]);

    const onFOVTriangleRotate = useCallback((cameraId: string, newRotation: number) => {
        updateFOVTriangle(cameraId, { rotation: newRotation });
        
        // Update camera direction
        const updatedCameras = selectedCameraPositions.map(camera => 
            camera.id === cameraId 
                ? { ...camera, direction: newRotation }
                : camera
        );
        onCameraPositionsUpdate(updatedCameras);
    }, [updateFOVTriangle, selectedCameraPositions, onCameraPositionsUpdate]);

    const onFOVTriangleResize = useCallback((cameraId: string, newRadius: number) => {
        // Update camera FOV radius
        const updatedCameras = selectedCameraPositions.map(camera => 
            camera.id === cameraId 
                ? { ...camera, fovRadius: newRadius }
                : camera
        );
        onCameraPositionsUpdate(updatedCameras);
    }, [selectedCameraPositions, onCameraPositionsUpdate]);

    // Handle FOV triangle rotation via mouse events
    const handleFOVRotation = useCallback((cameraId: string, event: google.maps.MapMouseEvent) => {
        if (!event.latLng) return;
        
        const camera = selectedCameraPositions.find(cam => cam.id === cameraId);
        if (!camera) return;
        
        const cameraLatLng = new window.google.maps.LatLng(camera.lat, camera.lng);
        const clickLatLng = event.latLng;
        
        // Calculate angle between camera center and click point
        const deltaLat = clickLatLng.lat() - cameraLatLng.lat();
        const deltaLng = clickLatLng.lng() - cameraLatLng.lng();
        const angle = Math.atan2(deltaLng, deltaLat) * 180 / Math.PI;
        
        // Convert to 0-360 range
        const normalizedAngle = (angle + 360) % 360;
        
        onFOVTriangleRotate(cameraId, normalizedAngle);
    }, [selectedCameraPositions, onFOVTriangleRotate]);

    // --- DATA SIMULATION LOGIC ---
    const getCrowdLevel = (cameraName: string): { level: string; color: string } => {
        // Default to medium level - will be replaced by actual data
        return { level: 'Medium', color: '#ffaa44' };
    };

    const generateCrowdData = useCallback(() => {
        if (!isLoaded || selectedCameraPositions.length === 0) return;
        const newHeatmapData: google.maps.LatLng[] = [];
        
        // Generate heatmap points around camera positions
        // This will be replaced by actual crowd detection data
        selectedCameraPositions.forEach(camera => {
            const pointsPerCamera = 3; // Default points per camera
            for (let i = 0; i < pointsPerCamera; i++) {
                newHeatmapData.push(new window.google.maps.LatLng(
                    camera.lat + (Math.random() - 0.5) * 0.003,
                    camera.lng + (Math.random() - 0.5) * 0.003
                ));
            }
        });
        
        setHeatMapData(newHeatmapData);
        setCrowdStats({
            totalPeople: 0, // Will be replaced by actual data
            activeCameras: selectedCameraPositions.length
        });
    }, [isLoaded, selectedCameraPositions]);

    // --- CAMERA PLACEMENT LOGIC ---
    const onCameraPlacementMapClick = (event: google.maps.MapMouseEvent) => {
        if (cameraPlacementMode && event.latLng) {
            // Check if we're clicking on an existing FOV triangle for rotation
            if (selectedFOVTriangle) {
                handleFOVRotation(selectedFOVTriangle, event);
                setSelectedFOVTriangle(null);
                return;
            }
            
            // Otherwise, place a new camera
            const newCamera: Camera = {
                id: `cam_${Date.now()}`,
                name: `Camera ${selectedCameraPositions.length + 1}`,
                lat: event.latLng.lat(),
                lng: event.latLng.lng(),
                status: 'active',
                fov: DEFAULT_FOV,
                direction: 0,
                fovRadius: DEFAULT_FOV_RADIUS
            };
            const updatedCameras = [...selectedCameraPositions, newCamera];
            onCameraPositionsUpdate(updatedCameras);
        }
    };

    const removeCamera = (cameraId: string) => {
        const updatedCameras = selectedCameraPositions.filter(cam => cam.id !== cameraId);
        onCameraPositionsUpdate(updatedCameras);
    };

    const submitCameraPositions = () => {
        if (selectedCameraPositions.length === 0) {
            alert("Please add at least one camera position.");
            return;
        }
        
        // Calculate the bounding circle for all cameras
        calculateCameraCoverageCircle();
        
        // Close placement mode and return to main view
        onCloseCameraPlacementMode();
        
        // Zoom both maps to show all cameras
        zoomToCameras();
    };

    const calculateCameraCoverageCircle = () => {
        if (selectedCameraPositions.length === 0) return;

        // Calculate center point of all cameras
        const centerLat = selectedCameraPositions.reduce((sum, cam) => sum + cam.lat, 0) / selectedCameraPositions.length;
        const centerLng = selectedCameraPositions.reduce((sum, cam) => sum + cam.lng, 0) / selectedCameraPositions.length;
        
        // Find the maximum distance from center to any camera
        let maxDistance = 0;
        selectedCameraPositions.forEach(camera => {
            const distance = window.google.maps.geometry.spherical.computeDistanceBetween(
                new window.google.maps.LatLng(centerLat, centerLng),
                new window.google.maps.LatLng(camera.lat, camera.lng)
            );
            maxDistance = Math.max(maxDistance, distance);
        });
        
        // Add 20% margin to ensure cameras are well inside the circle
        const radiusWithMargin = maxDistance * 1.2;
        
        const newCoverage = {
            center: { lat: centerLat, lng: centerLng },
            radius: radiusWithMargin
        };
        
        onCameraCoverageUpdate(newCoverage);
    };

    // --- MAP LOAD HANDLER ---
    const onMapLoad = useCallback((mapInstance: google.maps.Map) => setMap(mapInstance), []);
    
    // Function to zoom both maps to show all cameras
    const zoomToCameras = useCallback(() => {
        if (selectedCameraPositions.length === 0) return;
        
        const bounds = new window.google.maps.LatLngBounds();
        selectedCameraPositions.forEach(camera => {
            bounds.extend(new window.google.maps.LatLng(camera.lat, camera.lng));
        });
        
        // Calculate center and zoom level
        const center = bounds.getCenter();
        const newCenter = { lat: center.lat(), lng: center.lng() };
        
        // Update shared state
        setSharedCenter(newCenter);
        
        // Apply to both maps
        if (map) {
            map.fitBounds(bounds as any);
            const zoom = map.getZoom() || 13;
            setSharedZoom(Math.min(zoom, 15));
        }
        if (cameraPlacementMap) {
            cameraPlacementMap.fitBounds(bounds as any);
            const zoom = cameraPlacementMap.getZoom() || 13;
            setSharedZoom(Math.min(zoom, 15));
        }
    }, [selectedCameraPositions, map, cameraPlacementMap]);

    // --- ICONS ---
    const createCameraIcon = (color: string) => ({
        url: `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(`<svg width="30" height="30" viewBox="0 0 30 30" xmlns="http://www.w3.org/2000/svg"><circle cx="15" cy="15" r="12" fill="${color}" stroke="white" stroke-width="2"/><text x="15" y="20" text-anchor="middle" fill="white" font-size="14" font-weight="bold">📹</text></svg>`)}`,
        scaledSize: new window.google.maps.Size(30, 30)
    });

    // --- EFFECTS ---
    useEffect(() => {
        // Only generate crowd data for heatmap view and when cameras are present
        if (mapType === 'heatmap' && selectedCameraPositions.length > 0) {
            const interval = setInterval(generateCrowdData, 3000);
            return () => clearInterval(interval);
        }
    }, [generateCrowdData, mapType, selectedCameraPositions]);

    // Keyboard shortcuts
    useEffect(() => {
        const handleKeyPress = (event: KeyboardEvent) => {
            if (!cameraPlacementMode) return;
            
            switch (event.key) {
                case 'Escape':
                    if (selectedFOVTriangle) {
                        setSelectedFOVTriangle(null);
                    } else {
                        onCloseCameraPlacementMode();
                    }
                    break;
                case 'r':
                case 'R':
                    if (selectedFOVTriangle) {
                        // Reset camera direction to 0
                        const updatedCameras = selectedCameraPositions.map(cam => 
                            cam.id === selectedFOVTriangle 
                                ? { ...cam, direction: 0 }
                                : cam
                        );
                        onCameraPositionsUpdate(updatedCameras);
                    }
                    break;
                case 'f':
                case 'F':
                    if (selectedFOVTriangle) {
                        // Reset FOV radius to default
                        const updatedCameras = selectedCameraPositions.map(cam => 
                            cam.id === selectedFOVTriangle 
                                ? { ...cam, fovRadius: DEFAULT_FOV_RADIUS }
                                : cam
                        );
                        onCameraPositionsUpdate(updatedCameras);
                    }
                    break;
            }
        };

        if (cameraPlacementMode) {
            document.addEventListener('keydown', handleKeyPress);
            return () => document.removeEventListener('keydown', handleKeyPress);
        }
    }, [cameraPlacementMode, selectedFOVTriangle, selectedCameraPositions, onCameraPositionsUpdate, onCloseCameraPlacementMode]);

    // Only zoom when exiting camera placement mode
    // useEffect(() => {
    //     if (map && selectedCameraPositions.length > 0 && !cameraCoverageCircle) {
    //         zoomToCameras();
    //     }
    // }, [map, selectedCameraPositions, cameraCoverageCircle, zoomToCameras]);
    
    // Auto-zoom whenever camera positions change (add/remove cameras) - DISABLED during placement
    // useEffect(() => {
    //     if (map && selectedCameraPositions.length > 0) {
    //         zoomToCameras();
    //     }
    // }, [selectedCameraPositions.length, zoomToCameras]);

    // --- RENDER LOGIC ---
    if (!isLoaded) {
        return (
            <div className="flex justify-center items-center h-full bg-gray-800/40 text-white">
                <h2>Loading Emergency Response System...</h2>
            </div>
        );
    }

    // Camera Placement Mode View
    if (cameraPlacementMode) {
        return (
            <div className="relative w-full h-full">
                {/* Header */}
                <div className="absolute top-3 left-3 z-10 bg-gray-800/95 backdrop-blur-sm p-4 rounded-lg shadow-lg max-w-xs border border-gray-600">
                    <h3 className="text-lg font-semibold text-white mb-2">📹 Camera Placement Mode</h3>
                    <p className="text-sm text-gray-300 mb-3">
                        Click on the map to place cameras. Each click adds a new camera position.
                    </p>
                    
                    {/* FOV Configuration */}
                    <div className="mb-3 p-3 bg-gray-700 rounded border border-gray-600 max-h-96 overflow-y-auto">
                        <h4 className="text-sm font-medium text-white mb-2">🔺 FOV Configuration</h4>
                        
                        {/* Rotation Mode Indicator */}
                        {selectedFOVTriangle && (
                            <div className="mb-3 p-2 bg-yellow-600/20 border border-yellow-500/50 rounded text-xs text-yellow-300">
                                🎯 <strong>Rotation Mode Active</strong><br/>
                                Click anywhere on the map to set camera direction
                                <button
                                    onClick={() => setSelectedFOVTriangle(null)}
                                    className="ml-2 px-2 py-1 bg-yellow-600 hover:bg-yellow-700 text-white text-xs rounded transition-colors"
                                >
                                    Cancel
                                </button>
                            </div>
                        )}
                        
                        <div className="space-y-2">
                            <div>
                                <label className="text-xs text-gray-300 block">FOV Radius (meters)</label>
                                <input
                                    type="range"
                                    min="50"
                                    max="500"
                                    defaultValue={DEFAULT_FOV_RADIUS}
                                    className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                                    onChange={(e) => {
                                        const newRadius = parseInt(e.target.value);
                                        const updatedCameras = selectedCameraPositions.map(cam => ({
                                            ...cam,
                                            fovRadius: newRadius
                                        }));
                                        onCameraPositionsUpdate(updatedCameras);
                                    }}
                                />
                                <span className="text-xs text-gray-400">{DEFAULT_FOV_RADIUS}m</span>
                            </div>
                            <div>
                                <label className="text-xs text-gray-300 block">Default Direction (degrees)</label>
                                <input
                                    type="range"
                                    min="0"
                                    max="360"
                                    defaultValue="0"
                                    className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                                    onChange={(e) => {
                                        const newDirection = parseInt(e.target.value);
                                        const updatedCameras = selectedCameraPositions.map(cam => ({
                                            ...cam,
                                            direction: newDirection
                                        }));
                                        onCameraPositionsUpdate(updatedCameras);
                                    }}
                                />
                                <span className="text-xs text-gray-400">0°</span>
                            </div>
                        </div>
                        
                        {/* Legend */}
                        <div className="mt-3 p-2 bg-gray-600 rounded text-xs text-gray-300">
                            <div className="font-medium mb-2">🎨 Visual Legend:</div>
                            <div className="space-y-1">
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 bg-teal-400 rounded-sm"></div>
                                    <span>Normal FOV Triangle</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 bg-yellow-400 rounded-sm"></div>
                                    <span>Selected FOV Triangle</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 bg-red-500 rounded-sm"></div>
                                    <span>Camera Direction Line</span>
                                </div>
                            </div>
                        </div>
                        
                        <div className="mt-3 p-2 bg-gray-600 rounded text-xs text-gray-300">
                            <div className="font-medium mb-1">🎯 Interaction Guide:</div>
                            <div>• <strong>Click & Drag</strong> FOV triangle to move camera</div>
                            <div>• <strong>Click FOV triangle</strong> to open inline controls</div>
                            <div>• <strong>Use sliders</strong> to adjust radius & direction</div>
                            <div>• <strong>Double-click</strong> camera marker for quick access</div>
                            <div>• <strong>Inline controls</strong> appear above each camera</div>
                        </div>
                        
                        {/* Keyboard Shortcuts */}
                        <div className="mt-3 p-2 bg-gray-600 rounded text-xs text-gray-300">
                            <div className="font-medium mb-1">⌨️ Keyboard Shortcuts:</div>
                            <div>• <strong>ESC</strong> - Cancel rotation / Exit mode</div>
                            <div>• <strong>R</strong> - Reset camera direction to 0°</div>
                            <div>• <strong>F</strong> - Reset radius to {DEFAULT_FOV_RADIUS}m</div>
                        </div>
                    </div>
                    
                    {/* Inline FOV Controls - Now available directly on the map for each camera */}

                    <div className="mb-3">
                        <strong className="text-white">Cameras placed: {selectedCameraPositions.length}</strong>
                    </div>
                    
                    {selectedCameraPositions.length > 0 && (
                        <div className="max-h-32 overflow-y-auto mb-3">
                            {selectedCameraPositions.map((camera) => (
                                <div key={camera.id} className="flex justify-between items-center p-2 bg-gray-700 rounded mb-1">
                                    <div className="text-sm text-gray-200">
                                        <div>{camera.name}</div>
                                        <div className="text-xs text-gray-400">
                                            FOV: {camera.fov}° | Dir: {camera.direction}° | R: {camera.fovRadius}m
                                        </div>
                                    </div>
                                    <button 
                                        onClick={() => removeCamera(camera.id)}
                                        className="bg-red-500 hover:bg-red-600 text-white text-xs px-2 py-1 rounded transition-colors"
                                    >
                                        Remove
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                    
                    <div className="flex gap-2">
                        <button 
                            onClick={submitCameraPositions}
                            className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded font-medium transition-colors"
                        >
                            Submit & Return
                        </button>
                        <button 
                            onClick={onCloseCameraPlacementMode}
                            className="flex-1 px-3 py-2 bg-red-500 hover:bg-red-600 text-white rounded transition-colors"
                        >
                            Cancel
                        </button>
                    </div>
                </div>

                {/* Camera Placement Map */}
                <GoogleMap
                    mapContainerStyle={containerStyle}
                    center={sharedCenter}
                    zoom={sharedZoom}
                    onLoad={setCameraPlacementMap}
                    onClick={onCameraPlacementMapClick}
                    options={darkMapOptions}
                >
                    {/* Show existing cameras */}
                    {selectedCameraPositions.map(camera => (
                        <MarkerF
                            key={camera.id}
                            position={{ lat: camera.lat, lng: camera.lng }}
                            icon={createCameraIcon('#4ecdc4')}
                            title={camera.name}
                            onDblClick={() => onFOVTriangleClick(camera.id)}
                        />
                    ))}
                    
                    {/* Show FOV triangles */}
                    {fovTriangles.map(triangle => (
                        <React.Fragment key={triangle.cameraId}>
                            <PolygonF
                                paths={triangle.points}
                                options={{
                                    fillColor: selectedFOVTriangle === triangle.cameraId ? '#FFD700' : FOV_TRIANGLE_COLOR,
                                    fillOpacity: selectedFOVTriangle === triangle.cameraId ? 0.5 : 0.3,
                                    strokeColor: selectedFOVTriangle === triangle.cameraId ? '#FFD700' : FOV_TRIANGLE_BORDER_COLOR,
                                    strokeWeight: selectedFOVTriangle === triangle.cameraId ? 3 : 2,
                                    strokeOpacity: selectedFOVTriangle === triangle.cameraId ? 1 : 0.8,
                                    clickable: true,
                                    draggable: true
                                }}
                                onClick={() => onFOVTriangleClick(triangle.cameraId)}
                                onDragStart={() => onFOVTriangleDragStart(triangle.cameraId)}
                                onDragEnd={(e) => {
                                    if (e.latLng) {
                                        onFOVTriangleDragEnd(triangle.cameraId, e.latLng);
                                    }
                                }}
                            />
                            
                            {/* Direction indicator line */}
                            {(() => {
                                const camera = selectedCameraPositions.find(cam => cam.id === triangle.cameraId);
                                if (!camera) return null;
                                
                                const center = new window.google.maps.LatLng(camera.lat, camera.lng);
                                const directionRad = (camera.direction * Math.PI) / 180;
                                const radiusInDegrees = camera.fovRadius / 111000;
                                
                                const endLat = center.lat() + radiusInDegrees * 0.7 * Math.cos(directionRad);
                                const endLng = center.lng() + radiusInDegrees * 0.7 * Math.sin(directionRad);
                                
                                return (
                                    <PolygonF
                                        paths={[
                                            center,
                                            new window.google.maps.LatLng(endLat, endLng)
                                        ]}
                                        options={{
                                            strokeColor: '#FF0000',
                                            strokeWeight: 3,
                                            strokeOpacity: 0.8,
                                            fillOpacity: 0
                                        }}
                                    />
                                );
                            })()}
                            
                            {/* Inline FOV Controls for each camera */}
                            {(() => {
                                const camera = selectedCameraPositions.find(cam => cam.id === triangle.cameraId);
                                if (!camera) return null;
                                
                                // Position controls above the camera
                                const controlLat = camera.lat + 0.0005; // Offset above camera
                                const controlLng = camera.lng;
                                
                                return (
                                                                         <OverlayViewF
                                         position={{ lat: controlLat, lng: controlLng }}
                                         mapPaneName="overlayMouseTarget"
                                     >
                                        <FOVControls
                                            camera={camera}
                                            onUpdate={(updates) => {
                                                const updatedCameras = selectedCameraPositions.map(cam => 
                                                    cam.id === camera.id 
                                                        ? { ...cam, ...updates }
                                                        : cam
                                                );
                                                onCameraPositionsUpdate(updatedCameras);
                                            }}
                                            onClose={() => setSelectedFOVTriangle(null)}
                                        />
                                    </OverlayViewF>
                                );
                            })()}
                        </React.Fragment>
                    ))}
                </GoogleMap>
            </div>
        );
    }

    // Main Map View
    return (
        <div className="relative w-full h-full">
            {/* --- UI OVERLAYS --- */}
            <div className="absolute top-3 left-3 z-10 bg-gray-800/95 backdrop-blur-sm p-4 rounded-lg shadow-lg max-w-xs border border-gray-600">
                <h3 className="text-lg font-semibold text-white mb-2">
                    {mapType === 'heatmap' ? 'Crowd Detection Dashboard' : 'Escape Routes Dashboard'}
                </h3>
                <div className="text-sm text-gray-300 mb-1">
                    {mapType === 'heatmap' ? (
                        <>👥 Total People Detected: <strong className="text-white">{crowdStats.totalPeople}</strong></>
                    ) : (
                        <>🛣️ Total Routes Detected: <strong className="text-white">{crowdStats.totalPeople}</strong></>
                    )}
                </div>
                <div className="text-sm text-gray-300 mb-3">📹 Active Cameras: <strong className="text-white">{selectedCameraPositions.length}</strong></div>
                
                {mapType === 'escape-routes' && (
                    <div className="text-xs text-gray-400 pt-2 border-t border-gray-600">
                        💡 Place cameras here to set up crowd monitoring zones
                    </div>
                )}
                
                {selectedCameraPositions.length > 0 && (
                    <div className="text-xs text-gray-400 pt-2 border-t border-gray-600">
                        🎯 <strong>FOV Controls:</strong> Click any FOV triangle to open inline controls
                    </div>
                )}
            </div>

            {/* --- GOOGLE MAP COMPONENT --- */}
            <GoogleMap
                mapContainerStyle={containerStyle}
                center={sharedCenter}
                zoom={sharedZoom}
                onLoad={onMapLoad}
                options={darkMapOptions}
            >
                {/* Heatmap is only visible in heatmap view */}
                {mapType === 'heatmap' && <HeatmapLayerF data={heatMapData} />}

                {/* Dynamic Camera Markers (from user placement) */}
                {selectedCameraPositions.map(camera => (
                    <MarkerF
                        key={camera.id}
                        position={{ lat: camera.lat, lng: camera.lng }}
                        icon={createCameraIcon(getCrowdLevel(camera.name).color)}
                        onClick={() => setSelectedCamera(camera)}
                        onDblClick={() => onFOVTriangleClick(camera.id)}
                    />
                ))}

                {/* InfoWindow for selected camera */}
                {selectedCamera && (
                    <InfoWindowF
                        position={{ lat: selectedCamera.lat, lng: selectedCamera.lng }}
                        onCloseClick={() => setSelectedCamera(null)}
                    >
                        <div className="text-gray-800">
                            <h4 className="font-semibold">{selectedCamera.name}</h4>
                            <p className="text-sm">Status: {selectedCamera.status}</p>
                            <p className="text-sm">Crowd Level: {getCrowdLevel(selectedCamera.name).level}</p>
                        </div>
                    </InfoWindowF>
                )}
                
                {/* Camera Coverage Circle - encompasses all cameras with margin */}
                {cameraCoverageCircle && (
                    <CircleF
                        center={cameraCoverageCircle.center}
                        radius={cameraCoverageCircle.radius}
                        options={{
                            strokeColor: '#4ECDC4',
                            strokeOpacity: 0.8,
                            strokeWeight: 3,
                            fillColor: '#4ECDC4',
                            fillOpacity: 0.1
                        }}
                    />
                )}
                
                {/* Show FOV triangles */}
                {fovTriangles.map(triangle => (
                    <React.Fragment key={triangle.cameraId}>
                        <PolygonF
                            paths={triangle.points}
                            options={{
                                fillColor: selectedFOVTriangle === triangle.cameraId ? '#FFD700' : FOV_TRIANGLE_COLOR,
                                fillOpacity: selectedFOVTriangle === triangle.cameraId ? 0.4 : 0.2,
                                strokeColor: selectedFOVTriangle === triangle.cameraId ? '#FFD700' : FOV_TRIANGLE_BORDER_COLOR,
                                strokeWeight: selectedFOVTriangle === triangle.cameraId ? 2.5 : 1.5,
                                strokeOpacity: selectedFOVTriangle === triangle.cameraId ? 0.9 : 0.6,
                                clickable: true
                            }}
                            onClick={() => onFOVTriangleClick(triangle.cameraId)}
                        />
                        
                        {/* Direction indicator line */}
                        {(() => {
                            const camera = selectedCameraPositions.find(cam => cam.id === triangle.cameraId);
                            if (!camera) return null;
                            
                            const center = new window.google.maps.LatLng(camera.lat, camera.lng);
                            const directionRad = (camera.direction * Math.PI) / 180;
                            const radiusInDegrees = camera.fovRadius / 111000;
                            
                            const endLat = center.lat() + radiusInDegrees * 0.7 * Math.cos(directionRad);
                            const endLng = center.lng() + radiusInDegrees * 0.7 * Math.sin(directionRad);
                            
                            return (
                                <PolygonF
                                    paths={[
                                        center,
                                        new window.google.maps.LatLng(endLat, endLng)
                                    ]}
                                    options={{
                                        strokeColor: '#FF0000',
                                        strokeWeight: 2,
                                        strokeOpacity: 0.6,
                                        fillOpacity: 0
                                    }}
                                />
                            );
                        })()}
                        
                        {/* Inline FOV Controls for each camera in main view */}
                        {selectedFOVTriangle === triangle.cameraId && (() => {
                            const camera = selectedCameraPositions.find(cam => cam.id === triangle.cameraId);
                            if (!camera) return null;
                            
                            // Position controls above the camera
                            const controlLat = camera.lat + 0.0005; // Offset above camera
                            const controlLng = camera.lng;
                            
                            return (
                                <OverlayViewF
                                    position={{ lat: controlLat, lng: controlLng }}
                                    mapPaneName="overlayMouseTarget"
                                >
                                    <FOVControls
                                        camera={camera}
                                        onUpdate={(updates) => {
                                            const updatedCameras = selectedCameraPositions.map(cam => 
                                                cam.id === camera.id 
                                                    ? { ...cam, ...updates }
                                                    : cam
                                            );
                                            onCameraPositionsUpdate(updatedCameras);
                                        }}
                                        onClose={() => setSelectedFOVTriangle(null)}
                                    />
                                </OverlayViewF>
                            );
                        })()}
                    </React.Fragment>
                ))}
            </GoogleMap>
        </div>
    );
}

// --- FOV CONTROLS COMPONENT ---
const FOVControls: React.FC<FOVControlsProps> = ({ camera, onUpdate, onClose }) => {
    return (
        <div className="bg-gray-800/95 backdrop-blur-sm p-3 rounded-lg shadow-lg border border-gray-600 min-w-[200px] max-w-[220px]">
            <div className="text-xs font-medium text-white mb-2 text-center">
                {camera.name} Controls
            </div>
            
            <div className="space-y-3">
                {/* FOV Radius Control */}
                <div>
                    <label className="text-xs text-gray-300 block mb-1">Radius (m)</label>
                    <div className="flex items-center gap-2">
                        <input
                            type="range"
                            min="50"
                            max="500"
                            value={camera.fovRadius}
                            className="flex-1 h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                            onChange={(e) => {
                                const newRadius = parseInt(e.target.value);
                                onUpdate({ fovRadius: newRadius });
                            }}
                        />
                        <span className="text-xs text-white min-w-[2.5rem]">{camera.fovRadius}m</span>
                    </div>
                </div>
                
                {/* Direction Control */}
                <div>
                    <label className="text-xs text-gray-300 block mb-1">Direction (°)</label>
                    <div className="flex items-center gap-2">
                        <input
                            type="range"
                            min="0"
                            max="360"
                            value={camera.direction}
                            className="flex-1 h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                            onChange={(e) => {
                                const newDirection = parseInt(e.target.value);
                                onUpdate({ direction: newDirection });
                            }}
                        />
                        <span className="text-xs text-white min-w-[2.5rem]">{camera.direction}°</span>
                    </div>
                </div>
                
                {/* Quick Actions */}
                <div className="flex gap-1 pt-1">
                    <button
                        onClick={() => onUpdate({ direction: 0 })}
                        className="flex-1 px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded transition-colors"
                        title="Reset direction to 0°"
                    >
                        Reset Dir
                    </button>
                    <button
                        onClick={() => onUpdate({ fovRadius: DEFAULT_FOV_RADIUS })}
                        className="flex-1 px-2 py-1 bg-green-600 hover:bg-green-700 text-white text-xs rounded transition-colors"
                        title="Reset radius to default"
                    >
                        Reset R
                    </button>
                </div>
                
                {/* Close Button */}
                <button
                    onClick={onClose}
                    className="w-full px-2 py-1 bg-gray-600 hover:bg-gray-700 text-white text-xs rounded transition-colors"
                >
                    Close
                </button>
            </div>
        </div>
    );
};
