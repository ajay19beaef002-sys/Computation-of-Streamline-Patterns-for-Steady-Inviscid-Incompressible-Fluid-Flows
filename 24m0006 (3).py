import numpy as np
import matplotlib.pyplot as plt

def solve_stream_function(psi1, psi2, psi3, initial_guess):
    """
    Solves the stream-function equation using the Point Jacobi method.

    Args:
        psi1: Stream function value at the bottom boundary.
        psi2: Stream function value at the inlet.
        psi3: Stream function value at the top boundary.
        initial_guess: Initial guess for the stream function.

    Returns:
        psi: The converged stream function values.
        error_history: List of error values at each iteration.
    """

    # 1. Set up the grid
    nx = 31  # Number of points in x-direction (3m / 0.1m + 1)
    ny = 41  # Number of points in y-direction (4m / 0.1m + 1)
    psi = np.full((ny, nx), initial_guess, dtype=float)  # Initialize psi

    # 2. Apply boundary conditions
    #   Bottom boundary
    psi[0, :] = psi3
    #   Top boundary
    psi[ny - 1, :] = psi3
    #   Inlet and Outlet
    psi[:, 0] = psi2
    psi[:, nx - 1] = psi2

    # left
    psi[:, 0] = psi3
    psi[:, -1] = psi3
    psi[20:40, 15] = psi3  # Custom boundary condition in the middle column
    psi[0:11, 15] = psi1
    psi[11:20, 15] = psi2

    # 3. Point Jacobi Iteration
    error = 1.0
    error_tolerance = 1e-4
    error_history = []  # Initialize the error history
    psi_prev = np.copy(psi)  # Store previous iteration's psi values

    while error > error_tolerance:
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
               if i == 15 and (0 <= j < 11 or 11 <= j < 20 or 20 <= j < 40): 
                   continue  # Skip updating wall points
               psi[j, i] = 0.25 * (psi_prev[j, i + 1] + psi_prev[j, i - 1] +
                            psi_prev[j + 1, i] + psi_prev[j - 1, i])

        psi[0, :] = psi3
    #   Top boundary
        psi[ny - 1, :] = psi3
        #   Inlet and Outlet
        psi[:, 0] = psi2
        psi[:, nx - 1] = psi2
        
        psi[0:11, 15] = psi1
        psi[11:20, 15] = psi2
        psi[20:40, 15] = psi3  # Custom boundary condition in the middle column
        
        # left
        psi[:, 0] = psi3
        psi[:, -1] = psi3

        # bottom
        psi[0, 0:11] = psi3
        psi[0, 11:20] = psi1
        psi[0, 20:] = psi3

        # Calculate the L2 norm of the difference
        diff_norm = np.sqrt(np.sum((psi[1:ny - 1, 1:nx - 1] - psi_prev[1:ny - 1, 1:nx - 1])**2))
        
        # Calculate the L2 norm of the current solution
        psi_norm = np.sqrt(np.sum(psi[1:ny-1, 1:nx-1]**2))

        error = diff_norm / psi_norm
        error_history.append(error)

        # Update psi_prev for the next iteration
        psi_prev = np.copy(psi)

    return psi, error_history

def print_solution_at_x(psi, y_coords, x_values):
    """
    Prints the solution at specified x-locations.

    Args:
        psi: The converged stream function values.
        y_coords: The y-coordinates of the grid points.
        x_values: A list of x-values where the solution should be printed.
    """
    nx = psi.shape[1]
    x_coords = np.linspace(0, 3, nx)  # Create x-coordinates

    for x in x_values:
        x_index = np.argmin(np.abs(x_coords - x))  # Find closest x-index
        print(f"Solution at x = {x}:")
        for j, y in enumerate(y_coords):
            print(f"  y = {y:.1f}, psi = {psi[j, x_index]:.2f}")
        print("-" * 20)
    print(psi[:, 15])

# 4. Main Program
if __name__ == "__main__":
    # Define y-coordinates for printing the solution
    ny = 41
    y_coords = np.linspace(0, 4, ny)

    # Define x-values where the solution should be printed
    x_values_to_print = [0.5, 1.0, 1.5]  # Example x-values where the solution will be printed

    # Test Cases
    test_cases = [
        {"psi1": 100, "psi2": 150, "psi3": 300},
        {"psi1": 100, "psi2": 200, "psi3": 300},
        {"psi1": 100, "psi2": 250, "psi3": 300}
    ]
    
    # Different initial guesses
    initial_guesses = [100, 150, 200]

    for case_num, case in enumerate(test_cases):
        psi1 = case["psi1"]
        psi2 = case["psi2"]
        psi3 = case["psi3"]

        print(f"Test Case {case_num + 1}: psi1 = {psi1}, psi2 = {psi2}, psi3 = {psi3}")
        print("=" * 40)

        for initial_guess in initial_guesses:
            print(f"Initial Guess: {initial_guess}")
            psi, error_history = solve_stream_function(psi1, psi2, psi3, initial_guess)

            # Print stream function values for this test case and initial guess
            print(f"Stream Function Values for Test Case {case_num + 1}, Initial Guess: {initial_guess}")
            print(psi)  # Print the entire psi matrix
            print("-" * 40)

            # Define x and y coordinates for proper visualization
            x_vals = np.linspace(0, 3, psi.shape[1])  # X-coordinates
            y_vals = np.linspace(0, 4, psi.shape[0])  # Y-coordinates
            X, Y = np.meshgrid(x_vals, y_vals)  # Create meshgrid

            # Compute velocity components
            u = np.gradient(psi, axis=0)  # u = ∂ψ/∂y
            v = -np.gradient(psi, axis=1)  # v = -∂ψ/∂x

            # Plot stream function contour
            plt.figure(figsize=(8, 6))
            plt.contourf(X, Y, psi, levels=50)
            plt.colorbar(label="Stream Function ψ")
            plt.title(f"Stream Function - Test Case {case_num + 1}, Initial Guess: {initial_guess}")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

            # Plot velocity streamline profile separately
            plt.figure(figsize=(8, 6))
            plt.streamplot(X, Y, u, v, color='black', density=2, linewidth=1, arrowsize=1)
            plt.title(f"Velocity Streamlines - Test Case {case_num + 1}, Initial Guess: {initial_guess}")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

            # Plot error convergence
            plt.figure(figsize=(8, 6))
            plt.plot(error_history)
            plt.title(f"Convergence History - Test Case {case_num + 1}, Initial Guess: {initial_guess}")
            plt.xlabel("Iteration")
            plt.ylabel("Error")
            plt.yscale('log')
            plt.show()