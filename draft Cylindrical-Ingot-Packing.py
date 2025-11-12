"""
Random Ingot Packing Simulation - Drop and Stick Algorithm
For Rawwater Industrial Project
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as mpatches

# ============================================================================
# PARAMETERS
# ============================================================================

# Chamber dimensions
CHAMBER_RADIUS = 50.0  # mm
CHAMBER_HEIGHT = 100.0  # mm

# Ingot dimensions (cylindrical)
INGOT_RADIUS = 10.0  # mm
INGOT_HEIGHT = 30.0  # mm

# Simulation parameters
NUM_INGOTS = 10
DROP_STEP_SIZE = 0.5  # mm (smaller = more accurate but slower)
MAX_PLACEMENT_ATTEMPTS = 200  # How many random positions to try per ingot

# ============================================================================
# INGOT CLASS
# ============================================================================

class CylindricalIngot:
    """Represents a cylindrical ingot with position"""
    
    def __init__(self, x, y, z, radius, height):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.height = height
    
    def get_position(self):
        """Return center position as tuple"""
        return (self.x, self.y, self.z)
    
    def get_bottom_y(self):
        """Return Y coordinate of bottom surface"""
        return self.y - self.height / 2
    
    def get_top_y(self):
        """Return Y coordinate of top surface"""
        return self.y + self.height / 2
    
    def __repr__(self):
        return f"Ingot(x={self.x:.1f}, y={self.y:.1f}, z={self.z:.1f})"

# ============================================================================
# COLLISION DETECTION
# ============================================================================

def check_collision_with_ingot(new_ingot, existing_ingot):
    """
    Check if two cylindrical ingots collide
    
    Args:
        new_ingot: CylindricalIngot to test
        existing_ingot: CylindricalIngot already placed
    
    Returns:
        True if collision detected, False otherwise
    """
    # Horizontal distance between centers (in XZ plane)
    horizontal_distance = np.sqrt(
        (new_ingot.x - existing_ingot.x)**2 + 
        (new_ingot.z - existing_ingot.z)**2
    )
    
    # Check if horizontally overlapping
    min_horizontal_separation = new_ingot.radius + existing_ingot.radius
    horizontally_overlapping = horizontal_distance < min_horizontal_separation
    
    if not horizontally_overlapping:
        return False  # Too far apart horizontally
    
    # Check vertical overlap
    new_bottom = new_ingot.get_bottom_y()
    new_top = new_ingot.get_top_y()
    existing_bottom = existing_ingot.get_bottom_y()
    existing_top = existing_ingot.get_top_y()
    
    # Vertical overlap occurs if ranges intersect
    vertically_overlapping = not (new_top < existing_bottom or new_bottom > existing_top)
    
    return horizontally_overlapping and vertically_overlapping


def check_inside_chamber(x, z, chamber_radius, ingot_radius):
    """
    Check if ingot position is inside circular chamber
    
    Args:
        x, z: Horizontal position
        chamber_radius: Radius of chamber
        ingot_radius: Radius of ingot
    
    Returns:
        True if inside chamber (with clearance), False otherwise
    """
    distance_from_center = np.sqrt(x**2 + z**2)
    return distance_from_center <= (chamber_radius - ingot_radius)


def check_collision_with_all(new_ingot, existing_ingots):
    """
    Check if new ingot collides with any existing ingots
    
    Args:
        new_ingot: CylindricalIngot to test
        existing_ingots: List of already-placed ingots
    
    Returns:
        True if any collision detected, False otherwise
    """
    for existing_ingot in existing_ingots:
        if check_collision_with_ingot(new_ingot, existing_ingot):
            return True
    return False

# ============================================================================
# DROP AND STICK ALGORITHM
# ============================================================================

def drop_ingot(x, z, existing_ingots, chamber_height, ingot_radius, 
               ingot_height, step_size):
    """
    Drop an ingot from top of chamber until it hits bottom or another ingot
    
    Args:
        x, z: Horizontal position
        existing_ingots: List of already-placed ingots
        chamber_height: Height of chamber
        ingot_radius: Radius of ingot
        ingot_height: Height of ingot
        step_size: Distance to drop each iteration
    
    Returns:
        Final Y position if successfully placed, None if failed
    """
    # Start at top of chamber
    y = chamber_height - ingot_height / 2
    
    # Minimum Y position (sitting on chamber bottom)
    min_y = ingot_height / 2
    
    # Drop until we hit something
    while y > min_y:
        # Create test ingot at current height
        test_ingot = CylindricalIngot(x, y, z, ingot_radius, ingot_height)
        
        # Check collision with existing ingots
        if check_collision_with_all(test_ingot, existing_ingots):
            # Collision! Move back up one step and place there
            y += step_size
            return y
        
        # No collision, continue dropping
        y -= step_size
    
    # Hit the bottom
    return min_y


def generate_random_packing(num_ingots, chamber_radius, chamber_height,
                            ingot_radius, ingot_height, step_size,
                            max_attempts=200, random_seed=None):
    """
    Generate a random packing configuration using drop-and-stick algorithm
    
    Args:
        num_ingots: Number of ingots to place
        chamber_radius: Radius of chamber (mm)
        chamber_height: Height of chamber (mm)
        ingot_radius: Radius of ingots (mm)
        ingot_height: Height of ingots (mm)
        step_size: Drop increment (mm)
        max_attempts: Maximum attempts to place each ingot
        random_seed: Seed for reproducibility
    
    Returns:
        List of successfully placed CylindricalIngot objects
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    placed_ingots = []
    
    for i in range(num_ingots):
        placed = False
        
        for attempt in range(max_attempts):
            # Generate random horizontal position
            # Use uniform distribution in circle
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, chamber_radius - ingot_radius)
            
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            
            # Alternative: uniform in square (commented out)
            # x = np.random.uniform(-(chamber_radius - ingot_radius), 
            #                       (chamber_radius - ingot_radius))
            # z = np.random.uniform(-(chamber_radius - ingot_radius), 
            #                       (chamber_radius - ingot_radius))
            
            # Check if inside chamber
            if not check_inside_chamber(x, z, chamber_radius, ingot_radius):
                continue
            
            # Drop from top
            y = drop_ingot(x, z, placed_ingots, chamber_height, 
                          ingot_radius, ingot_height, step_size)
            
            if y is not None:
                # Successfully placed!
                new_ingot = CylindricalIngot(x, y, z, ingot_radius, ingot_height)
                placed_ingots.append(new_ingot)
                print(f"Ingot {i+1}/{num_ingots} placed at ({x:.1f}, {y:.1f}, {z:.1f})")
                placed = True
                break
        
        if not placed:
            print(f"WARNING: Could not place ingot {i+1} after {max_attempts} attempts")
    
    return placed_ingots

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_packing_density(ingots, chamber_radius, chamber_height):
    """Calculate packing density (volume fraction)"""
    chamber_volume = np.pi * chamber_radius**2 * chamber_height
    
    ingot_volume = 0
    for ingot in ingots:
        ingot_volume += np.pi * ingot.radius**2 * ingot.height
    
    packing_density = ingot_volume / chamber_volume
    return packing_density


def calculate_total_surface_area(ingots):
    """Calculate total exposed surface area of all ingots"""
    total_area = 0
    for ingot in ingots:
        # Cylindrical surface + two circular ends
        lateral_area = 2 * np.pi * ingot.radius * ingot.height
        end_area = 2 * np.pi * ingot.radius**2
        total_area += lateral_area + end_area
    
    return total_area


def count_contacts(ingots, tolerance=1.0):
    """
    Count number of contact points between ingots
    
    Args:
        ingots: List of ingots
        tolerance: Distance threshold for considering contact (mm)
    """
    contact_count = 0
    
    for i, ingot1 in enumerate(ingots):
        for ingot2 in ingots[i+1:]:
            # Check if they're touching
            horizontal_distance = np.sqrt(
                (ingot1.x - ingot2.x)**2 + 
                (ingot1.z - ingot2.z)**2
            )
            
            # Vertically aligned?
            vertical_gap = min(
                abs(ingot1.get_bottom_y() - ingot2.get_top_y()),
                abs(ingot2.get_bottom_y() - ingot1.get_top_y())
            )
            
            # Check horizontal contact
            horizontal_touching = horizontal_distance < (ingot1.radius + ingot2.radius + tolerance)
            
            # Check vertical contact
            vertical_touching = vertical_gap < tolerance
            
            if horizontal_touching or vertical_touching:
                contact_count += 1
    
    return contact_count

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_packing_3d(ingots, chamber_radius, chamber_height, title="Random Packing"):
    """Create 3D visualization of packing"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw chamber (wireframe cylinder)
    theta = np.linspace(0, 2*np.pi, 50)
    z_chamber = np.linspace(0, chamber_height, 50)
    Theta, Z = np.meshgrid(theta, z_chamber)
    X_chamber = chamber_radius * np.cos(Theta)
    Y_chamber = chamber_radius * np.sin(Theta)
    
    ax.plot_surface(X_chamber, Y_chamber, Z, alpha=0.1, color='blue', edgecolor='blue')
    
    # Draw chamber base
    theta_base = np.linspace(0, 2*np.pi, 50)
    r_base = np.linspace(0, chamber_radius, 20)
    Theta_base, R_base = np.meshgrid(theta_base, r_base)
    X_base = R_base * np.cos(Theta_base)
    Y_base = R_base * np.sin(Theta_base)
    Z_base = np.zeros_like(X_base)
    
    ax.plot_surface(X_base, Y_base, Z_base, alpha=0.2, color='gray')
    
    # Draw ingots
    for i, ingot in enumerate(ingots):
        # Create cylinder for each ingot
        theta_ingot = np.linspace(0, 2*np.pi, 20)
        z_ingot = np.array([ingot.get_bottom_y(), ingot.get_top_y()])
        Theta_ingot, Z_ingot = np.meshgrid(theta_ingot, z_ingot)
        X_ingot = ingot.radius * np.cos(Theta_ingot) + ingot.x
        Y_ingot = ingot.radius * np.sin(Theta_ingot) + ingot.z
        
        ax.plot_surface(X_ingot, Y_ingot, Z_ingot, 
                       alpha=0.7, color=plt.cm.tab10(i % 10))
        
        # Draw top and bottom caps
        r_cap = np.linspace(0, ingot.radius, 10)
        Theta_cap, R_cap = np.meshgrid(theta_ingot, r_cap)
        X_cap = R_cap * np.cos(Theta_cap) + ingot.x
        Y_cap = R_cap * np.sin(Theta_cap) + ingot.z
        
        # Bottom cap
        Z_bottom = np.ones_like(X_cap) * ingot.get_bottom_y()
        ax.plot_surface(X_cap, Y_cap, Z_bottom, alpha=0.7, color=plt.cm.tab10(i % 10))
        
        # Top cap
        Z_top = np.ones_like(X_cap) * ingot.get_top_y()
        ax.plot_surface(X_cap, Y_cap, Z_top, alpha=0.7, color=plt.cm.tab10(i % 10))
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_zlabel('Y (mm)')
    ax.set_title(title)
    ax.set_box_aspect([1,1,2])  # Aspect ratio
    
    plt.tight_layout()
    return fig


def plot_packing_top_view(ingots, chamber_radius, title="Top View"):
    """Create top-down 2D view of packing"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw chamber circle
    chamber = Circle((0, 0), chamber_radius, fill=False, 
                    edgecolor='blue', linewidth=2, label='Chamber')
    ax.add_patch(chamber)
    
    # Draw ingots
    for i, ingot in enumerate(ingots):
        circle = Circle((ingot.x, ingot.z), ingot.radius, 
                       alpha=0.6, color=plt.cm.tab10(i % 10))
        ax.add_patch(circle)
        
        # Add label
        ax.text(ingot.x, ingot.z, str(i+1), 
               ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_xlim(-chamber_radius*1.1, chamber_radius*1.1)
    ax.set_ylim(-chamber_radius*1.1, chamber_radius*1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_packing_side_view(ingots, chamber_radius, chamber_height, title="Side View"):
    """Create side view of packing"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw chamber outline
    chamber_rect = Rectangle((-chamber_radius, 0), 2*chamber_radius, chamber_height,
                            fill=False, edgecolor='blue', linewidth=2, label='Chamber')
    ax.add_patch(chamber_rect)
    
    # Draw ingots (as rectangles in side view)
    for i, ingot in enumerate(ingots):
        rect = Rectangle((ingot.x - ingot.radius, ingot.get_bottom_y()),
                        2*ingot.radius, ingot.height,
                        alpha=0.6, color=plt.cm.tab10(i % 10))
        ax.add_patch(rect)
        
        # Add label
        ax.text(ingot.x, ingot.y, str(i+1),
               ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_xlim(-chamber_radius*1.1, chamber_radius*1.1)
    ax.set_ylim(-5, chamber_height*1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    return fig

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_to_excel(ingots, filename='ingot_positions.xlsx'):
    """Export ingot positions to Excel file"""
    data = {
        'Ingot_ID': range(1, len(ingots) + 1),
        'X_mm': [ingot.x for ingot in ingots],
        'Y_mm': [ingot.y for ingot in ingots],
        'Z_mm': [ingot.z for ingot in ingots],
        'Radius_mm': [ingot.radius for ingot in ingots],
        'Height_mm': [ingot.height for ingot in ingots]
    }
    
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
    print(f"\nPositions exported to: {filename}")
    return df


def export_to_csv(ingots, filename='ingot_positions.csv'):
    """Export ingot positions to CSV file"""
    data = {
        'Ingot_ID': range(1, len(ingots) + 1),
        'X_mm': [ingot.x for ingot in ingots],
        'Y_mm': [ingot.y for ingot in ingots],
        'Z_mm': [ingot.z for ingot in ingots],
        'Radius_mm': [ingot.radius for ingot in ingots],
        'Height_mm': [ingot.height for ingot in ingots]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Positions exported to: {filename}")
    return df

# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def run_monte_carlo_study(num_simulations, num_ingots, chamber_radius, 
                          chamber_height, ingot_radius, ingot_height, 
                          step_size):
    """
    Run multiple random packing simulations for statistical analysis
    
    Args:
        num_simulations: Number of Monte Carlo iterations
        (other parameters same as generate_random_packing)
    
    Returns:
        Dictionary with results and statistics
    """
    print(f"\n{'='*60}")
    print(f"MONTE CARLO SIMULATION: {num_simulations} iterations")
    print(f"{'='*60}\n")
    
    all_packings = []
    packing_densities = []
    surface_areas = []
    contact_counts = []
    num_placed = []
    
    for sim in range(num_simulations):
        print(f"\n--- Simulation {sim+1}/{num_simulations} ---")
        
        # Generate random packing
        ingots = generate_random_packing(
            num_ingots, chamber_radius, chamber_height,
            ingot_radius, ingot_height, step_size,
            random_seed=sim  # Different seed each time
        )
        
        # Calculate metrics
        density = calculate_packing_density(ingots, chamber_radius, chamber_height)
        surface_area = calculate_total_surface_area(ingots)
        contacts = count_contacts(ingots)
        
        # Store results
        all_packings.append(ingots)
        packing_densities.append(density)
        surface_areas.append(surface_area)
        contact_counts.append(contacts)
        num_placed.append(len(ingots))
        
        print(f"  Placed: {len(ingots)}/{num_ingots} ingots")
        print(f"  Density: {density:.3f}")
        print(f"  Surface area: {surface_area:.1f} mm²")
        print(f"  Contacts: {contacts}")
    
    # Statistical analysis
    results = {
        'packings': all_packings,
        'packing_densities': np.array(packing_densities),
        'surface_areas': np.array(surface_areas),
        'contact_counts': np.array(contact_counts),
        'num_placed': np.array(num_placed)
    }
    
    return results


def print_monte_carlo_statistics(results):
    """Print statistical summary of Monte Carlo results"""
    print(f"\n{'='*60}")
    print("MONTE CARLO STATISTICS")
    print(f"{'='*60}\n")
    
    print(f"Number of simulations: {len(results['packing_densities'])}")
    
    print(f"\nPacking Density:")
    print(f"  Mean: {np.mean(results['packing_densities']):.4f}")
    print(f"  Std Dev: {np.std(results['packing_densities']):.4f}")
    print(f"  Min: {np.min(results['packing_densities']):.4f}")
    print(f"  Max: {np.max(results['packing_densities']):.4f}")
    print(f"  Median: {np.median(results['packing_densities']):.4f}")
    
    print(f"\nSurface Area (mm²):")
    print(f"  Mean: {np.mean(results['surface_areas']):.1f}")
    print(f"  Std Dev: {np.std(results['surface_areas']):.1f}")
    print(f"  Min: {np.min(results['surface_areas']):.1f}")
    print(f"  Max: {np.max(results['surface_areas']):.1f}")
    
    print(f"\nContact Points:")
    print(f"  Mean: {np.mean(results['contact_counts']):.1f}")
    print(f"  Std Dev: {np.std(results['contact_counts']):.1f}")
    print(f"  Min: {int(np.min(results['contact_counts']))}")
    print(f"  Max: {int(np.max(results['contact_counts']))}")
    
    print(f"\nIngots Successfully Placed:")
    print(f"  Mean: {np.mean(results['num_placed']):.1f}")
    print(f"  Success Rate: {np.mean(results['num_placed'])/NUM_INGOTS*100:.1f}%")


def plot_monte_carlo_distributions(results):
    """Plot histograms of Monte Carlo results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Packing density
    axes[0, 0].hist(results['packing_densities'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(results['packing_densities']), 
                       color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 0].set_xlabel('Packing Density')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Packing Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Surface area
    axes[0, 1].hist(results['surface_areas'], bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].axvline(np.mean(results['surface_areas']),
                       color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 1].set_xlabel('Total Surface Area (mm²)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Surface Area')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Contact counts
    axes[1, 0].hist(results['contact_counts'], bins=range(int(np.min(results['contact_counts'])),
                                                          int(np.max(results['contact_counts']))+2),
                   edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].axvline(np.mean(results['contact_counts']),
                       color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1, 0].set_xlabel('Number of Contact Points')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Contact Points')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Number placed
    axes[1, 1].hist(results['num_placed'], 
                   bins=range(int(np.min(results['num_placed'])),
                             int(np.max(results['num_placed']))+2),
                   edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].axvline(np.mean(results['num_placed']),
                       color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1, 1].set_xlabel('Number of Ingots Placed')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Successfully Placed Ingots')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def select_representative_cases(results):
    """
    Select representative cases from Monte Carlo results
    
    Returns indices for: worst, median, best packing densities
    """
    densities = results['packing_densities']
    
    worst_idx = np.argmin(densities)
    best_idx = np.argmax(densities)
    median_idx = np.argmin(np.abs(densities - np.median(densities)))
    
    print(f"\n{'='*60}")
    print("REPRESENTATIVE CASES SELECTED")
    print(f"{'='*60}")
    print(f"\nWorst case (lowest density): Simulation #{worst_idx+1}")
    print(f"  Density: {densities[worst_idx]:.4f}")
    
    print(f"\nMedian case: Simulation #{median_idx+1}")
    print(f"  Density: {densities[median_idx]:.4f}")
    
    print(f"\nBest case (highest density): Simulation #{best_idx+1}")
    print(f"  Density: {densities[best_idx]:.4f}")
    
    return worst_idx, median_idx, best_idx

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("="*60)
    print("RANDOM INGOT PACKING SIMULATION")
    print("Drop-and-Stick Algorithm")
    print("="*60)
    print(f"\nChamber: R={CHAMBER_RADIUS}mm, H={CHAMBER_HEIGHT}mm")
    print(f"Ingots: R={INGOT_RADIUS}mm, H={INGOT_HEIGHT}mm")
    print(f"Number to place: {NUM_INGOTS}")
    print(f"Drop step size: {DROP_STEP_SIZE}mm")
    
    # ========================================================================
    # OPTION 1: Single packing simulation
    # ========================================================================
    
    print("\n" + "="*60)
    print("SINGLE PACKING SIMULATION")
    print("="*60)
    
    ingots = generate_random_packing(
        NUM_INGOTS, CHAMBER_RADIUS, CHAMBER_HEIGHT,
        INGOT_RADIUS, INGOT_HEIGHT, DROP_STEP_SIZE,
        random_seed=42  # For reproducibility
    )
    
    # Calculate metrics
    density = calculate_packing_density(ingots, CHAMBER_RADIUS, CHAMBER_HEIGHT)
    surface_area = calculate_total_surface_area(ingots)
    contacts = count_contacts(ingots)
    
    print(f"\nRESULTS:")
    print(f"  Successfully placed: {len(ingots)}/{NUM_INGOTS} ingots")
    print(f"  Packing density: {density:.4f} ({density*100:.2f}%)")
    print(f"  Total surface area: {surface_area:.1f} mm²")
    print(f"  Contact points: {contacts}")
    
    # Export single case
    export_to_csv(ingots, 'single_packing.csv')
    
    # Visualize single case
    fig1 = plot_packing_3d(ingots, CHAMBER_RADIUS, CHAMBER_HEIGHT, "3D View")
    fig2 = plot_packing_top_view(ingots, CHAMBER_RADIUS, "Top View")
    fig3 = plot_packing_side_view(ingots, CHAMBER_RADIUS, CHAMBER_HEIGHT, "Side View")

# Show all plots
    plt.show()
    
    # ========================================================================
    # OPTION 2: Monte Carlo study (comment out if not needed)
    # ========================================================================
    
    run_monte_carlo = input("\nRun Monte Carlo simulation? (y/n): ").lower() == 'y'
    
    if run_monte_carlo:
        num_mc_sims = int(input("Number of Monte Carlo simulations (e.g., 50): "))
        
        mc_results = run_monte_carlo_study(
            num_mc_sims, NUM_INGOTS, CHAMBER_RADIUS, CHAMBER_HEIGHT,
            INGOT_RADIUS, INGOT_HEIGHT, DROP_STEP_SIZE
        )
        
        # Print statistics
        print_monte_carlo_statistics(mc_results)
        
        # Plot distributions
        fig4 = plot_monte_carlo_distributions(mc_results)
        
        # Select representative cases
        worst_idx, median_idx, best_idx = select_representative_cases(mc_results)
        
        # Export representative cases
        export_to_csv(mc_results['packings'][worst_idx], 'config_worst.csv')
        export_to_csv(mc_results['packings'][median_idx], 'config_median.csv')
        export_to_csv(mc_results['packings'][best_idx], 'config_best.csv')
        
        # Visualize representative cases
        fig5 = plot_packing_3d(mc_results['packings'][worst_idx], 
                              CHAMBER_RADIUS, CHAMBER_HEIGHT, 
                              f"Worst Case (Density={mc_results['packing_densities'][worst_idx]:.3f})")
        
        fig6 = plot_packing_3d(mc_results['packings'][median_idx], 
                              CHAMBER_RADIUS, CHAMBER_HEIGHT, 
                              f"Median Case (Density={mc_results['packing_densities'][median_idx]:.3f})")
        
        fig7 = plot_packing_3d(mc_results['packings'][best_idx], 
                              CHAMBER_RADIUS, CHAMBER_HEIGHT, 
                              f"Best Case (Density={mc_results['packing_densities'][best_idx]:.3f})")
    
    # Show all plots
    plt.show()
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - single_packing.csv")
    if run_monte_carlo:
        print("  - config_worst.csv")
        print("  - config_median.csv")
        print("  - config_best.csv")
    print("\nUse these CSV files to import coordinates into SolidWorks!")


# ============================================================================
# RUN THE PROGRAM
# ============================================================================

if __name__ == "__main__":
    main()

