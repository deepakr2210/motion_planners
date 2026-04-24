#pragma once

#include <string>
#include <vector>
#include <stdexcept>

#include <Eigen/Dense>
#include <mujoco/mujoco.h>


class DiffIKControl{

public:
    DiffIKControl(mjModel* model, mjData* data,  
                    const std::string& ee_site,
                    double dt, double lam = 0.1)
        : model_(model), data_(data), dt_(dt), lam_(lam) //basically model_=model; etc
    {
        nv_ = model->nv;
        ee_id_ = mj_name2id(model, mjOBJ_SITE, ee_site.c_str());
        if (ee_id_ < 0)
            throw std::invalid_argument("Site '" + ee_site + "' not found in model");
    }

    std::vector<double> execute(const std::vector<double>& q_current,
                                const std::vector<double>& v_in)
    {
        const int ndof = static_cast<int>(q_current.size()); //convert to int

        for (int i=0; i < ndof; ++i)
            data_->qpos[i] = q_current[i]; // update the mjData with the current q_pos
        mj_kinematics(model_, data_);      // update Kinematics
        mj_comPos(model_, data_);          // needed when updating kinematics 

        Eigen::MatrixXd J = siteJacobian();
        Eigen::MatrixXd J_inv = dampedPinv(J);

        // zero-copy wrapper to map std::vector to Eigen::vector 
        Eigen::Map<const Eigen::VectorXd> v(v_in.data(), 6); 
        Eigen::VectorXd q_dot = J_inv * v; // (nv, )

        std::vector<double> q_new(ndof);
        for (int i=0; i<ndof; ++i)
            q_new[i] = q_current[i] + q_dot[i] * dt_;
        
            return q_new;    
    }

private:
    mjModel* model_;
    mjData* data_;
    int nv_;
    int ee_id_;
    double dt_;
    double lam_;

    Eigen::MatrixXd siteJacobian() // get site Jacobian
    {
        // As Mujoco expects (3 x nv) row-major arrays
        using RowMat3N = Eigen::Matrix<double, 3, Eigen::RowMajor>;
        RowMat3N jacp(3, nv_), jacr(3, nv_); // Translation and Rotation DOF Jacobians
        jacp.setZero();
        jacr.setZero();
        mj_jacSite(model_, data_, jacp.data(), jacr.data(), ee_id_);

        Eigen::MatrixXd J(6, nv_);
        J.topRows(3) = jacp;
        J.bottomRows(3) = jacr;
        return J;
    }

    Eigen::MatrixXd dampedPinv(const Eigen::MatrixXd& J)
    {
        const int n = J.rows();
        Eigen::MatrixXd A = J * J.transpose() + lam_ * lam_ * Eigen::MatrixXd::Identity(n,n);
        return J.transpose() * A.inverse();
    }
};