declare global {
  interface Window {
    google: typeof google;
  }
}

declare namespace google.maps {
  class LatLng {
    constructor(lat: number, lng: number);
    lat(): number;
    lng(): number;
    equals(other: LatLng | null): boolean;
    toJSON(): any;
    toUrlValue(precision?: number): string;
  }

  class LatLngBounds {
    constructor();
    extend(point: LatLng): void;
    contains(point: LatLng): boolean;
    equals(other: LatLngBounds | LatLngBoundsLiteral | null): boolean;
    getCenter(): LatLng;
    getNorthEast(): LatLng;
    getSouthWest(): LatLng;
    isEmpty(): boolean;
    toJSON(): any;
    toUrlValue(precision?: number): string;
    intersects(other: LatLngBounds): boolean;
    toSpan(): LatLng;
    union(other: LatLngBounds): LatLngBounds;
  }

  interface LatLngBoundsLiteral {
    east: number;
    north: number;
    south: number;
    west: number;
  }

  class Size {
    constructor(width: number, height: number);
    equals(other: Size | null): boolean;
    height: number;
    width: number;
  }

  class Map {
    constructor(mapDiv: Element, opts?: MapOptions);
    fitBounds(bounds: LatLngBounds | LatLngBoundsLiteral): void;
    setZoom(zoom: number): void;
    getZoom(): number;
  }

  interface MapOptions {
    center?: LatLng;
    zoom?: number;
    mapTypeControl?: boolean;
    streetViewControl?: boolean;
    fullscreenControl?: boolean;
    styles?: any[];
    backgroundColor?: string;
  }

  interface MapMouseEvent {
    latLng?: LatLng;
  }

  interface Icon {
    url: string;
    scaledSize?: Size;
    anchor?: Point;
    origin?: Point;
  }

  class Point {
    constructor(x: number, y: number);
    x: number;
    y: number;
    equals(other: Point | null): boolean;
  }

  namespace geometry {
    namespace spherical {
      function computeDistanceBetween(from: LatLng, to: LatLng): number;
    }
  }
}

export {};
