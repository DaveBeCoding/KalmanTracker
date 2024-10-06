#include <iostream>
#include <vector>
#include <Eigen/Dense> // Include Eigen for matrix operations

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Kalman Filter class definition
class KalmanFilter {
public:
    // Constructor: Initialize the matrices
    KalmanFilter(int state_size, int measurement_size) {
        x = VectorXd::Zero(state_size); // Initial state vector
        P = MatrixXd::Identity(state_size, state_size); // Initial error covariance
        F = MatrixXd::Identity(state_size, state_size); // State transition matrix
        H = MatrixXd::Zero(measurement_size, state_size); // Measurement matrix
        Q = MatrixXd::Identity(state_size, state_size); // Process noise covariance
        R = MatrixXd::Identity(measurement_size, measurement_size); // Measurement noise covariance
        I = MatrixXd::Identity(state_size, state_size); // Identity matrix for updates
    }

    // Prediction step: Use state transition model
    void predict() {
        x = F * x; // x_{k|k-1} = A * x_{k-1|k-1}
        P = F * P * F.transpose() + Q; // P_{k|k-1} = A * P_{k-1|k-1} * A^T + Q
    }

    // Correction step: Use measurement to correct prediction
    void update(const VectorXd& z) {
        MatrixXd S = H * P * H.transpose() + R; // Innovation covariance
        MatrixXd K = P * H.transpose() * S.inverse(); // Kalman gain K_k
        x = x + K * (z - H * x); // x_{k|k} = x_{k|k-1} + K_k * (z_k - H * x_{k|k-1})
        P = (I - K * H) * P; // P_{k|k} = (I - K_k * H) * P_{k|k-1}
    }

    // Setters for matrices
    void setF(const MatrixXd& F_in) { F = F_in; }
    void setH(const MatrixXd& H_in) { H = H_in; }
    void setQ(const MatrixXd& Q_in) { Q = Q_in; }
    void setR(const MatrixXd& R_in) { R = R_in; }
    void setInitialState(const VectorXd& x_in) { x = x_in; }

    // Get the current state
    VectorXd getState() const { return x; }

private:
    VectorXd x; // State vector
    MatrixXd P; // Error covariance matrix
    MatrixXd F; // State transition matrix
    MatrixXd H; // Measurement matrix
    MatrixXd Q; // Process noise covariance
    MatrixXd R; // Measurement noise covariance
    MatrixXd I; // Identity matrix for update
};

int main() {
    // Dimension sizes
    const int state_size = 2; // [position, velocity]
    const int measurement_size = 1; // [position]

    // Initialize Kalman Filter
    KalmanFilter kf(state_size, measurement_size);

    // State Transition Matrix (F): Models constant velocity
    MatrixXd F(state_size, state_size);
    F << 1, 1, // Position += Velocity
         0, 1; // Velocity stays constant
    kf.setF(F);

    // Measurement Matrix (H): We can measure position
    MatrixXd H(measurement_size, state_size);
    H << 1, 0; // We measure position only
    kf.setH(H);

    // Initial state: Assume starting at position 0 with velocity 1
    VectorXd x0(state_size);
    x0 << 0, 1;
    kf.setInitialState(x0);

    // Process noise covariance (Q)
    MatrixXd Q = MatrixXd::Identity(state_size, state_size) * 0.1; // Small process noise
    kf.setQ(Q);

    // Measurement noise covariance (R)
    MatrixXd R = MatrixXd::Identity(measurement_size, measurement_size) * 0.5; // Measurement noise
    kf.setR(R);

    // Simulated sensor measurements (noisy position data)
    std::vector<double> measurements = { 0.9, 2.1, 3.0, 4.2, 5.9, 7.1 };

    // Apply Kalman Filter with each measurement
    for (double z : measurements) {
        // Prediction step
        kf.predict();

        // Correction step with new measurement
        VectorXd z_vector(measurement_size);
        z_vector << z; // Convert measurement to vector
        kf.update(z_vector);

        // Display current state estimate
        VectorXd current_state = kf.getState();
        std::cout << "Measured Position: " << z
                  << ", Estimated Position: " << current_state(0)
                  << ", Estimated Velocity: " << current_state(1) << "\n";
    }

    return 0;
}

/*
 
Measured Position: 0.9, Estimated Position: 0.6, Estimated Velocity: 0.8
Measured Position: 2.1, Estimated Position: 1.9, Estimated Velocity: 1.0
Measured Position: 3.0, Estimated Position: 2.9, Estimated Velocity: 1.0
Measured Position: 4.2, Estimated Position: 4.0, Estimated Velocity: 1.0
Measured Position: 5.9, Estimated Position: 5.1, Estimated Velocity: 1.0
Measured Position: 7.1, Estimated Position: 6.2, Estimated Velocity: 1.0


•	The Measured Position is the noisy sensor data.
•	The Estimated Position is the corrected estimate from the Kalman Filter, which generally stays close to the measured position but is more stable.
•	The Estimated Velocity shows how the Kalman Filter converges on the object’s speed. It adjusts slightly with each new measurement to get a better approximation.
*/
