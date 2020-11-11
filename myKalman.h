//
// Created by jiangbin on 2020/11/9.
//

#ifndef KALMANFILTER_MYKALMAN_H
#define KALMANFILTER_MYKALMAN_H

class KalmanFilter{
private:
    /*!
     * state function: x(k) = A*x(k-1) + B*u(k) + w(k)  // w(k) is the state noise, w~(0,R), R is tne covariance of the state noise
     * measure function: z(k) = H*x(k) + v(k) // v(k) is the measurement noise, v~(0,Q), Q is the covariance of the measurement noise

     prediction:
     * error covariance prediction: P'(k) = A*P(k-1)*A(T) + Q // P is the error covariance, A is the state variable, Q is the measurement noise
     * state prediction: x'(k) = A*x'(k-1) + B*u(k)

     update:
     * kalman matrix: K(k) = P'(k)*H(T)/(H*P'(k)*H(T)+R)
     * state update: x(k) = x'(k) + K(k)*(z(k)-H*x'(k))
     * error covariance update: P(k) = (I - K(k)*H)*P'(k)

     */
    int stateSize;  // state variable's dimensions
    int measureSize;    // measure variable's dimensions
    int controlSize;    // control variable's dimensions
    Eigen::VectorXd x;  // state
    Eigen::VectorXd z;  // measure
    Eigen::VectorXd u;  // control
    Eigen::MatrixXd A;  // state variable
    Eigen::MatrixXd B;  // control variable
    Eigen::MatrixXd H;  // measurement variable
    Eigen::MatrixXd P;  // error covariance
    Eigen::MatrixXd R;  // state noise covariance
    Eigen::MatrixXd Q;  // measurement noise covariance
public:
    KalmanFilter(int stateSize_, int measureSize_, int controlSize_);
    void init(Eigen::VectorXd& x_,Eigen::MatrixXd P, Eigen::MatrixXd R, Eigen::MatrixXd Q);
    Eigen::VectorXd predict(Eigen::MatrixXd &A_, Eigen::MatrixXd &B_, Eigen::VectorXd &u_);
    Eigen::VectorXd predict(Eigen::MatrixXd &A_);
    void update(Eigen::MatrixXd &H_, Eigen::VectorXd& z_meas);
};

KalmanFilter::KalmanFilter(int stateSize_, int measureSize_, int controlSize_):stateSize(stateSize_), measureSize(measureSize_), controlSize(controlSize_)
{
    if(stateSize == 0 || measureSize == 0){
        std::cerr<<"error! stateSize and measureSize should bigger than 0";
    }

    x.resize(stateSize);
    x.setZero();

    A.resize(stateSize,stateSize);
    A.setIdentity();

    u.resize(controlSize);
    u.transpose();
    u.setZero();

    B.resize(stateSize,controlSize);
    B.setZero();

    z.resize(measureSize);
    z.setZero();

    H.resize(measureSize,stateSize);
    H.setZero();

    P.resize(stateSize,stateSize);
    P.setIdentity();

    R.resize(stateSize,stateSize);
    R.setZero();

    Q.resize(measureSize,measureSize);
    Q.setZero();
}

void KalmanFilter::init(Eigen::VectorXd &x_, Eigen::MatrixXd P_, Eigen::MatrixXd R_, Eigen::MatrixXd Q_) {
    x = x_;
    P = P_;
    R = R_;
    Q = Q_;
}

Eigen::VectorXd KalmanFilter::predict(Eigen::MatrixXd &A_, Eigen::MatrixXd &B_, Eigen::VectorXd &u_) {
    A = A_;
    B = B_;
    u = u_;
    x = A*x + B*u;
    Eigen::MatrixXd A_T = A.transpose();
    P = A*P*A_T + Q;
    return x;
}

Eigen::VectorXd KalmanFilter::predict(Eigen::MatrixXd &A_) {
    A = A_;
    x = A*x;
    Eigen::MatrixXd A_T = A.transpose();
    P = A*P*A_T + Q;
    return x;
}

void KalmanFilter::update(Eigen::MatrixXd &H_, Eigen::VectorXd& z_meas){
    H = H_;
    Eigen::MatrixXd temp1, temp2,Ht;
    Ht = H.transpose();
    temp1 = H*P*Ht + R;
    temp2 = temp1.inverse();//(H*P*H'+R)^(-1)
    Eigen::MatrixXd K = P*Ht*temp2;
    z = H*x;
    x = x + K*(z_meas-z);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(stateSize, stateSize);
    P = (I - K*H)*P;
}

#endif //KALMANFILTER_MYKALMAN_H
