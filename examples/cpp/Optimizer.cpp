#include <MParT/ConditionalMapBase.h>

using namespace mpart; 

void LevenbergMarquardtSolver(std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> map, 
                             Eigen::MatrixXd                                 const& x, 
                             Eigen::VectorXd                                 const& y)
{
    const unsigned int numPts = x.cols();
    assert(y.size() == numPts);

    Eigen::VectorXd coeffs = map->CoeffMap();
    Eigen::MatrixXd sens = Eigen::MatrixXd::Ones(1,numPts);
    
    Eigen::MatrixXd jac = map->CoeffGrad(x,sens);
    Eigen::VectorXd objGrad = y - map->Evaluate(x).row(0).transpose();
    double obj = 0.5*objGrad.squaredNorm();
    Eigen::VectorXd paramGrad = jac * objGrad;

    double stepSize;
    double newObj;

    const double ftol = 1e-6;
    const double gtol = 1e-4;
    double lambda = 1e-5;
    const double lambdaScale = 5;

    Eigen::VectorXd newObjGrad;
    Eigen::MatrixXd hess;

    printf("Iteration, Objective, Grad Norm,   Lambda\n");

    for(unsigned int optIt=0; optIt<5000; ++optIt){

        hess = jac * jac.transpose();
        hess += lambda * hess.diagonal().asDiagonal(); 

        map->CoeffMap() = coeffs + hess.ldlt().solve(paramGrad);
        newObjGrad = y - map->Evaluate(x).row(0).transpose();
        newObj = 0.5*newObjGrad.squaredNorm();

        if(newObj < obj){

            // Check for convergence
            if(std::abs(obj-newObj)<ftol){
                std::cout << "SUCCESS! Terminating due to small change in objective." << std::endl;
                return;
            }

            if(paramGrad.norm()<gtol){
                std::cout << "SUCCESS! Terminating due to small gradient norm." << std::endl;
                return;
            }
            
            coeffs = map->CoeffMap();
            lambda /= lambdaScale;

            objGrad = newObjGrad;// y - map->Evaluate(x).row(0).transpose();
            obj = newObj; //0.5*objGrad.squaredNorm();
            jac = map->CoeffGrad(x,sens);
            paramGrad = jac * objGrad;

        }else{
            map->CoeffMap() = coeffs;
            lambda *= lambdaScale;
        }

        printf("%9d, %9.2e, %9.2e, %6.2e\n", optIt, obj,paramGrad.norm(), lambda );
    }

}
