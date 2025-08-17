# Frontend Components Structure

This directory contains all React components organized by their purpose and functionality.

## Directory Structure

```
components/
├── ui/                    # Reusable UI components (shadcn/ui)
├── features/              # Feature-specific components
│   ├── crowd-monitoring/  # Crowd detection and monitoring
│   ├── dashboard/         # Dashboard and analytics
│   └── landing/          # Landing page and marketing
├── layout/                # Layout and navigation components
├── common/                # Shared/reusable components
└── __tests__/             # Component tests
```

## Component Categories

### UI Components (`/ui`)
- **shadcn/ui components**: Button, Card, Input, etc.
- **Purpose**: Basic, reusable UI elements
- **Usage**: Import directly from `@/components/ui/[component-name]`

### Feature Components (`/features`)
- **crowd-monitoring**: Components specific to crowd detection functionality
  - `CrowdMap.tsx` - Main crowd monitoring map component
- **dashboard**: Dashboard and analytics components
  - `DashboardPage.tsx` - Main dashboard page component
- **landing**: Landing page and marketing components
  - `LandingPage.tsx` - Main landing page component
  - `WorkingPage.tsx` - How it works page component

### Layout Components (`/layout`)
- **Navigation and structure components**
  - `Navbar.tsx` - Main navigation bar
  - `Footer.tsx` - Site footer
  - `ThemeProvider.tsx` - Theme context provider

### Common Components (`/common`)
- **Shared/reusable components across features**
  - `HorizontalSlides.tsx` - Horizontal sliding component
  - `SplitText.tsx` - Text animation component
  - `DarkVeil.tsx` - Background overlay component
  - `SectionTransition.tsx` - Section transition animations

## Import Patterns

### Using Index Files (Recommended)
```typescript
import { LandingPage, DashboardPage, CrowdMap } from "@/components";
```

### Direct Imports
```typescript
import { LandingPage } from "@/components/features";
import { Navbar } from "@/components/layout";
import { SplitText } from "@/components/common";
```

## Naming Conventions

- **Components**: PascalCase (e.g., `CrowdMap.tsx`)
- **Directories**: kebab-case (e.g., `crowd-monitoring/`)
- **Files**: PascalCase for components, camelCase for utilities

## Adding New Components

1. **Feature Components**: Place in appropriate `/features/[category]/` directory
2. **Layout Components**: Place in `/layout/` directory
3. **Common Components**: Place in `/common/` directory
4. **UI Components**: Place in `/ui/` directory
5. **Update index files**: Export new components from appropriate index files

## Benefits of This Structure

- **Clear separation of concerns**
- **Easier to find and maintain components**
- **Better scalability for large applications**
- **Consistent import patterns**
- **Logical grouping by functionality**
