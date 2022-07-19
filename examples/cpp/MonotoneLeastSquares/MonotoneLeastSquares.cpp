/**
TODO: ADD DESCRIPTION
 */


#include <random>
#include <fstream>
#include <stdio.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <MParT/ConditionalMapBase.h>
#include <MParT/MapFactory.h>
#include <MParT/MultiIndices/MultiIndexSet.h>

#include "../Optimizer.cpp"

using namespace mpart; 

int main(int argc, char* argv[]){

    Kokkos::initialize(argc,argv);
    {

    if (argc < 3) {
        std::cerr << "usage: MonotoneLeastSquares NOISESTD MAXDEGREE\n";
        return EXIT_FAILURE;
    }

    double noise_std = atof(argv[1]);
    unsigned int maxDegree = atof(argv[2]);

    // Generate noisy data
    unsigned int num_points = 1000;
    int xmin = 0;
    int xmax = 4;
    Eigen::MatrixXd x(1,num_points);
    x.row(0).setLinSpaced(num_points, xmin, xmax);

    Eigen::VectorXd y_true = 2*(x.row(0).array() > 2).cast<double>();

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, noise_std);
    auto normal = [&] (int) {return distribution(generator);};
    Eigen::VectorXd y_noise = Eigen::VectorXd::NullaryExpr(num_points, normal);

    Eigen::VectorXd y_measured = y_true + y_noise;

    // Create the map
    MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(x.rows(), maxDegree);

    MapOptions opts;
    opts.quadMinSub = 2;
    std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> map = MapFactory::CreateComponent(mset.Fix(), opts);

    // Solve the regression problem for the map coefficients
    Eigen::VectorXd map_of_x_before = map->Evaluate(x).row(0);
    LevenbergMarquadtSolver(map, x, y_measured);
    Eigen::VectorXd map_of_x_after = map->Evaluate(x).row(0);

    // Save the data to a csv file for plotting
    std::ofstream file("data.dat");
    assert(file.is_open());

    file << "X" << "\t" << "Y_True" << "\t" << "Y_Obs" << "\t"<< "Map_Initial" << "\t" << "Map_Optimized" << "\n";
    for (size_t i = 0; i < num_points; ++i){
        file << x(i) << "\t" << y_true(i) << "\t" << y_measured(i) << "\t"<< map_of_x_before(i) << "\t" << map_of_x_after(i) << "\n";
    }

    }
    Kokkos::finalize();
	
    return 0;
}

