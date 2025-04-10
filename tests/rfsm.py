import numpy as np
import heapq
import warnings

def calculate_flood_depth_from_rainfall(dem_data, rainfall_data):
    """
    Calculates flood depth map based on a DEM and spatially distributed 
    rainfall depth using a priority-flood algorithm.

    Assumes water flows off the edges and fills depressions until equilibrium.

    Args:
        dem_data (np.ndarray): 2D NumPy array of terrain elevations.
        rainfall_data (np.ndarray): 2D NumPy array of the *depth* of rainfall 
                                    added to each cell. Must have the same 
                                    shape as dem_data.

    Returns:
        np.ndarray: 2D NumPy array of the calculated flood depth at each cell, 
                    or None if inputs are invalid. Returns array of zeros if
                    no flooding occurs.
    """

    if not isinstance(dem_data, np.ndarray) or dem_data.ndim != 2:
        raise ValueError("dem_data must be a 2D NumPy array.")
    if not isinstance(rainfall_data, np.ndarray) or rainfall_data.ndim != 2:
        raise ValueError("rainfall_data must be a 2D NumPy array.")
    if dem_data.shape != rainfall_data.shape:
        raise ValueError("dem_data and rainfall_data must have the same shape.")

    rows, cols = dem_data.shape
    
    # --- Input Data Preparation ---
    # Ensure rainfall is non-negative
    rainfall_data = np.maximum(0, rainfall_data) 
    
    # Initial potential water surface if no flow occurred
    potential_water_surface = dem_data + rainfall_data

    # --- Algorithm Initialization ---
    # Final calculated water surface elevation for each cell
    # Initialize with negative infinity, indicating not yet calculated definitively
    # We use negative infinity because we are finding the *lowest* elevation water
    # needs to reach to flow *out*. Any cell not connected to the outflow
    # path will retain its initial potential_water_surface.
    # A slight variation: Initialize with potential_water_surface. The algorithm
    # will then *lower* the water surface where outflow is possible.
    water_surface_elevation = np.copy(potential_water_surface)

    # Priority queue (min-heap) stores tuples: (elevation, row, col)
    # We process cells with lower elevations first.
    pq = []

    # Boolean mask to track cells added to the queue/processed
    # Initialize all to False
    processed = np.zeros((rows, cols), dtype=bool)

    # --- Seed the Queue with Boundary Cells ---
    # Add all boundary cells to the priority queue. Water can potentially flow 
    # out from these cells, so their initial water level is their potential level.
    print("Initializing priority queue with boundary cells...")
    boundary_indices = []
    for r in range(rows):
        boundary_indices.extend([(r, 0), (r, cols - 1)])
    for c in range(1, cols - 1):
        boundary_indices.extend([(0, c), (rows - 1, c)])

    for r, c in boundary_indices:
        if not processed[r, c]:
            elev = water_surface_elevation[r, c] # Use the potential surface here
            heapq.heappush(pq, (elev, r, c))
            processed[r, c] = True
            
    print(f"Initialized queue with {len(pq)} boundary cells.")
    if not pq:
        warnings.warn("DEM has no boundary cells (1x1 or empty?). Returning zero depth.")
        return np.zeros_like(dem_data)

    # --- Main Processing Loop (Priority Flood) ---
    print("Processing cells...")
    processed_count = 0
    while pq:
        current_wse, r, c = heapq.heappop(pq)
        processed_count += 1
        if processed_count % 10000 == 0:
             print(f"  Processed {processed_count} cells...")

        # The actual water surface elevation at (r, c) cannot be lower than
        # the elevation from which we popped it. We might have pushed it
        # multiple times; ensure we use the final popped value.
        # water_surface_elevation[r, c] = current_wse # Update happens in neighbor step

        # Check 8 neighbors (or 4 if preferred)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc

            # Check if neighbor is within bounds
            if 0 <= nr < rows and 0 <= nc < cols:
                
                # Calculate the water surface elevation required at the neighbor
                # It must be at least its own potential level (ground+rain)
                neighbor_potential_wse = potential_water_surface[nr, nc]
                
                # It must also be at least the level of the current cell 
                # (water needs to be at least this high to potentially flow *from* # current to neighbor, or for them to be part of the same pool)
                # This propagates the "spill elevation" inwards.
                neighbor_wse_candidate = max(neighbor_potential_wse, current_wse)

                # If this calculated level is lower than the neighbor's current
                # recorded water surface elevation, it means we found a lower
                # path for water to stabilize or flow out. Update and add to queue.
                if neighbor_wse_candidate < water_surface_elevation[nr, nc]:
                    water_surface_elevation[nr, nc] = neighbor_wse_candidate
                    # Push the neighbor onto the queue with its new, lower elevation
                    heapq.heappush(pq, (neighbor_wse_candidate, nr, nc))
                    # Mark as processed to avoid redundant pushes? No, allow reprocessing if lower path found.
                    # Only mark processed *when* adding to queue initially? Let's stick to the standard.
                    # Correction: Standard priority flood marks processed *when adding*.
                    # However, allowing updates if a lower path is found IS the core mechanic.
                    # Let's refine: only push if not processed OR if a lower path found.
                    # But the `if neighbor_wse_candidate < water_surface_elevation[nr, nc]` check
                    # already handles finding a better path. Let's remove the `processed` check here
                    # and rely solely on the elevation comparison for updates.
                    # The initial boundary seeding handles the initial `processed` state.

    print(f"Processing complete. Processed {processed_count} queue items.")

    # --- Calculate Flood Depth ---
    # Depth is the difference between the final water surface and the ground DEM
    flood_depth = water_surface_elevation - dem_data
    
    # Ensure depth is not negative (due to potential floating point inaccuracies)
    flood_depth = np.maximum(0, flood_depth)

    # Optional: Set very small values to zero
    # tolerance = 1e-6 
    # flood_depth[flood_depth < tolerance] = 0

    return flood_depth

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Create Sample Data (replace with your actual data)
    print("Creating sample DEM and rainfall data...")
    rows, cols = 50, 70
    
    # Simple bowl-shaped DEM
    x = np.linspace(-5, 5, cols)
    y = np.linspace(-5, 5, rows)
    xx, yy = np.meshgrid(x, y)
    dem = (xx**2 + yy**2) * 0.5 # Parabolic bowl
    
    # Add some noise/complexity
    dem += np.random.rand(rows, cols) * 2.0 
    
    # Make edges slightly lower to ensure outflow
    dem[0, :] = dem[1, :] - 1
    dem[-1, :] = dem[-2, :] - 1
    dem[:, 0] = dem[:, 1] - 1
    dem[:, -1] = dem[:, -2] - 1
    
    # Create Rainfall Data
    # Example 1: Truly uniform rainfall depth
    rainfall_amount = 5.0 # Uniform depth of 5.0 units
    rainfall = np.full((rows, cols), rainfall_amount, dtype=np.float32)
    
    # Example 2: Spatially variable rainfall (optional)
    # rainfall = np.random.rand(rows, cols) * 8.0 + 2.0 # Random rainfall 2 to 10

    print(f"DEM shape: {dem.shape}")
    print(f"Rainfall shape: {rainfall.shape}")
    print(f"Uniform rainfall depth applied: {rainfall_amount if 'rainfall_amount' in locals() else 'Variable'}")

    # 2. Calculate Flood Depth
    print("\nCalculating flood depth...")
    flood_depth_map = calculate_flood_depth_from_rainfall(dem.astype(np.float32), rainfall.astype(np.float32))

    if flood_depth_map is not None:
        print("\nCalculation finished.")
        print(f"Flood depth map shape: {flood_depth_map.shape}")
        print(f"Maximum flood depth: {np.nanmax(flood_depth_map):.2f}")
        print(f"Minimum flood depth: {np.nanmin(flood_depth_map):.2f}")
        print(f"Average flood depth (where depth > 0): {np.nanmean(flood_depth_map[flood_depth_map > 1e-6]):.2f}")
        print(f"Percentage of area flooded: {np.count_nonzero(flood_depth_map > 1e-6) / flood_depth_map.size * 100:.2f}%")

        # Optional: Visualization (requires matplotlib)
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            im1 = axes[0].imshow(dem, cmap='terrain')
            axes[0].set_title('DEM')
            plt.colorbar(im1, ax=axes[0], label='Elevation')
            
            # Use a mask to show only flooded areas clearly
            flood_depth_masked = np.ma.masked_where(flood_depth_map <= 1e-6, flood_depth_map)
            cmap_flood = plt.cm.Blues
            cmap_flood.set_bad(color='lightgrey') # Show non-flooded areas in grey

            im2 = axes[1].imshow(flood_depth_masked, cmap=cmap_flood, vmin=0)
            axes[1].set_title('Calculated Flood Depth')
            plt.colorbar(im2, ax=axes[1], label='Depth')
            
            # Show final water surface
            final_wse = dem + flood_depth_map
            im3 = axes[2].imshow(final_wse, cmap='viridis')
            axes[2].set_title('Final Water Surface Elevation')
            plt.colorbar(im3, ax=axes[2], label='Elevation')

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("\nMatplotlib not found. Skipping visualization.")
            print("Install it using: pip install matplotlib")