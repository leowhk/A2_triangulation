/**
 * Copyright (C) 2015 by Liangliang Nan (liangliang.nan@gmail.com)
 * https://3d.bk.tudelft.nl/liangliang/
 *
 * This file is part of Easy3D. If it is useful in your research/work,
 * I would be grateful if you show your appreciation by citing it:
 * ------------------------------------------------------------------
 *      Liangliang Nan.
 *      Easy3D: a lightweight, easy-to-use, and efficient C++
 *      library for processing and rendering 3D data. 2018.
 * ------------------------------------------------------------------
 * Easy3D is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License Version 3
 * as published by the Free Software Foundation.
 *
 * Easy3D is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "triangulation.h"
#include "matrix_algo.h"
#include <easy3d/optimizer/optimizer_lm.h>

using namespace easy3d;

void find_centroid_translation(const std::vector<Vector2D> &points, Vector2D &c, std::vector<Vector2D> &translated_pts) {
    double sum_x, sum_y, sx, sy;
    for (Vector2D po: points) {
        sum_x += po.x();
        sum_y += po.y();
    }

    sx = sum_x / points.size();
    sy = sum_y / points.size();
    c = {sx, sy};

    for (Vector2D pt : points){
        double tx = pt.x() - c.x();
        double ty = pt.y() - c.y();
        Vector2D t_pt = {tx, ty};
        translated_pts.emplace_back(t_pt);
    }
}

void find_scale_factor(const std::vector<Vector2D> &points, const Vector2D &c, double &s){
    double sum;
    for(Vector2D po: points){
        sum += pow(po.x() - c.x(), 2) + pow(po.y() - c.y(),2);
    }
    s = sqrt(sum/points.size());
}

void ST_transform(const std::vector<Vector2D> &points, const double &s, std::vector<Vector2D> &scaled_points){
    for(Vector2D pt : points){
        double sx = pt.x()/s;
        double sy = pt.y()/s;
        Vector2D s_pt = {sx, sy};
        scaled_points.emplace_back(s_pt);
    }
}

void homo_coordinates(const std::vector<Vector2D> &points2D, std::vector<Vector3D> &h_points3D){
    for (Vector2D pt : points2D){
        Vector3D ya = pt.homogeneous();
        h_points3D.emplace_back(ya);
//        std::cout << ya << std::endl;
    }
}

void find_f(const Matrix &W, Matrix33 &f){
    // Compute SVD of W, where Wf = 0
    int m = W.rows();
    int n = W.cols();

    Matrix U(m, m, 0.0);
    Matrix S(m, n, 0.0);
    Matrix V(n, n, 0.0);

    svd_decompose(W, U, S, V);

    //Last column of V gives you the f
    Vector V_last = V.get_column(n-1);

    f = {V_last[0], V_last[1], V_last[2],
         V_last[3], V_last[4], V_last[5],
         V_last[6], V_last[7], V_last[8]};
}

void compute_F(const std::vector<Vector3D> &points2D_0, const std::vector<Vector3D> &points2D_1,
               Matrix &W, Matrix &F){

    std::vector<double> line;
    for(int i = 0; i < points2D_0.size(); i++){
        double p0_x = points2D_0[i].x();
        double p0_y = points2D_0[i].y();
        double p0_z = points2D_0[i].z();
        double p1_x = points2D_1[i].x();
        double p1_y = points2D_1[i].y();
        double p1_z = points2D_1[i].z();

        line = {p1_x * p0_x, p1_y * p0_x, p1_z * p0_x,
                p1_x * p0_y, p1_y * p0_y, p1_z * p0_y,
                p1_x * p0_z, p1_y * p0_z, p1_z * p0_z};

        W.set_row(i, line);
    }

    Matrix33 f;
    find_f(W, f);
    std::cout << f << std::endl;

    int m = f.rows();
    int n = f.cols();

    Matrix U(m, m, 0.0);
    Matrix S(m, n, 0.0);
    Matrix V(n, n, 0.0);

    svd_decompose(f, U, S, V);

    S(2,2) = 0;
    std::cout << S << std::endl;
    F = U * S * V.transpose();

//    std::vector<double> Fo(9, 0.0);
//    std::vector<double> Zero(points2D_0.size(), 0.0);
//
//    solve_least_squares(W, Zero, Fo);
//
//    Matrix33 Fi (Fo[0],Fo[1],Fo[2],
//                 Fo[3],Fo[4],Fo[5],
//                 Fo[6],Fo[7],Fo[8]);
//
//    std::cout << Fi << std::endl;
//
//
//    Matrix U(3, 3, 0.0);
//    Matrix S(3, 3, 0.0);
//    Matrix V(3, 3, 0.0);
//
//    svd_decompose(Fi, U, S, V);
//
//    std::cout << Fi << std::endl;
//    std::cout << U << std::endl;
//    std::cout << S << std::endl;
//    std::cout << V.transpose() << std::endl;

}

void transformation_matrix(const std::vector<Vector2D> &points, const std::vector<Vector2D> &transformed_points,
                           Matrix &original_matrix, Matrix &transformed_matrix){
    for(int i=0; i<points.size(); i++){
        double transformed_px = transformed_points[i].x() - points[i].x();
        double transformed_py = transformed_points[i].y() - points[i].y();
        double original_px = points[i].x();
        double original_py = points[i].y();
        original_matrix.set_row(i, {original_px, original_py, 1});
        transformed_matrix.set_row(i, {transformed_px, transformed_py, 1});
    }
}


/**
 * TODO: Finish this function for reconstructing 3D geometry from corresponding image points.
 * @return True on success, otherwise false. On success, the reconstructed 3D points must be written to 'points_3d'
 *      and the recovered relative pose must be written to R and t.
 */
bool Triangulation::triangulation(
        double fx, double fy,     /// input: the focal lengths (same for both cameras)
        double cx, double cy,     /// input: the principal point (same for both cameras)
        const std::vector<Vector2D> &points_0,  /// input: 2D image points in the 1st image.
        const std::vector<Vector2D> &points_1,  /// input: 2D image points in the 2nd image.
        std::vector<Vector3D> &points_3d,       /// output: reconstructed 3D points
        Matrix33 &R,   /// output: 3 by 3 matrix, which is the recovered rotation of the 2nd camera
        Vector3D &t    /// output: 3D vector, which is the recovered translation of the 2nd camera
) const
{
    /// NOTE: there might be multiple workflows for reconstructing 3D geometry from corresponding image points.
    ///       This assignment uses the commonly used one explained in our lecture.
    ///       It is advised to define a function for the sub-tasks. This way you have a clean and well-structured
    ///       implementation, which also makes testing and debugging easier. You can put your other functions above
    ///       triangulation(), or put them in one or multiple separate files.

    std::cout << "\nTODO: I am going to implement the triangulation() function in the following file:" << std::endl
              << "\t    - triangulation_method.cpp\n\n";

    std::cout << "[Liangliang]:\n"
                 "\tFeel free to use any provided data structures and functions. For your convenience, the\n"
                 "\tfollowing three files implement basic linear algebra data structures and operations:\n"
                 "\t    - Triangulation/matrix.h  Matrices of arbitrary dimensions and related functions.\n"
                 "\t    - Triangulation/vector.h  Vectors of arbitrary dimensions and related functions.\n"
                 "\t    - Triangulation/matrix_algo.h  Determinant, inverse, SVD, linear least-squares...\n"
                 "\tPlease refer to the above files for a complete list of useful functions and their usage.\n\n"
                 "\tIf you choose to implement the non-linear method for triangulation (optional task). Please\n"
                 "\trefer to 'Tutorial_NonlinearLeastSquares/main.cpp' for an example and some explanations.\n\n"
                 "\tIn your final submission, please\n"
                 "\t    - delete ALL unrelated test or debug code and avoid unnecessary output.\n"
                 "\t    - include all the source code (and please do NOT modify the structure of the directories).\n"
                 "\t    - do NOT include the 'build' directory (which contains the intermediate files in a build step).\n"
                 "\t    - make sure your code compiles and can reproduce your results without ANY modification.\n\n" << std::flush;

    /// Below are a few examples showing some useful data structures and APIs.

    /// define a 2D vector/point
    Vector2D b(1.1, 2.2);

    /// define a 3D vector/point
    Vector3D a(1.1, 2.2, 3.3);

    /// get the Cartesian coordinates of a (a is treated as Homogeneous coordinates)
    Vector2D p = a.cartesian();

    /// get the Homogeneous coordinates of p
    Vector3D q = p.homogeneous();

    /// define a 3 by 3 matrix (and all elements initialized to 0.0)
    Matrix33 A;

    /// define and initialize a 3 by 3 matrix
    Matrix33 T(1.1, 2.2, 3.3,
               0, 2.2, 3.3,
               0, 0, 1);

    /// define and initialize a 3 by 4 matrix
    Matrix34 M(1.1, 2.2, 3.3, 0,
               0, 2.2, 3.3, 1,
               0, 0, 1, 1);

    /// set first row by a vector
    M.set_row(0, Vector4D(1.1, 2.2, 3.3, 4.4));

    /// set second column by a vector
    M.set_column(1, Vector3D(5.5, 5.5, 5.5));

    /// define a 15 by 9 matrix (and all elements initialized to 0.0)
    Matrix WW(15, 9, 0.0);
    /// set the first row by a 9-dimensional vector
    WW.set_row(0, {0, 1, 2, 3, 4, 5, 6, 7, 8}); // {....} is equivalent to a std::vector<double>

    /// get the number of rows.
    int num_rows = WW.rows();

    /// get the number of columns.
    int num_cols = WW.cols();

    /// get the the element at row 1 and column 2
    double value = WW(1, 2);

    /// get the last column of a matrix
    Vector last_column = WW.get_column(WW.cols() - 1);

    /// define a 3 by 3 identity matrix
    Matrix33 I = Matrix::identity(3, 3, 1.0);

    /// matrix-vector product
    Vector3D v = M * Vector4D(1, 2, 3, 4); // M is 3 by 4

    ///For more functions of Matrix and Vector, please refer to 'matrix.h' and 'vector.h'

    // TODO: delete all above example code in your final submission

    //--------------------------------------------------------------------------------------------------------------
    // implementation starts ...

    // TODO: check if the input is valid (always good because you never known how others will call your function).


    // TODO: Estimate relative pose of two views. This can be subdivided into
    //      - estimate the fundamental matrix F;

    // Step 1: FInd the centroid in the both images and translate points to new origin: centroid
    Vector2D centroid_0, centroid_1;
    std::vector<Vector2D> Tpoints_0, Tpoints_1;

    find_centroid_translation(points_0, centroid_0, Tpoints_0);
    find_centroid_translation(points_1, centroid_1, Tpoints_1);

//    std::cout << "Centroid 0: " << centroid_0 << std::endl;
//    std::cout << "Centroid 1: " << centroid_1 << std::endl;
//    for(int i=0; i < points_0.size(); i++){
//        std::cout << points_0[i] << "\t\t" << Tpoints_0[i] << std::endl;
//    }

    // Step 2: Calculate the mean distance to centre from each of the two centroid points and scale
    double s_0, s_1;
    find_scale_factor(points_0, centroid_0, s_0);
    find_scale_factor(points_1, centroid_1, s_1);
//    std::cout << "scale factor on image 0: " << s_0 << std::endl;
//    std::cout << "scale factor on image 1: " << s_1 << std::endl;

    // Step 3: Translate the points with origin at the centroids with regards to scale
    std::vector<Vector2D> STpoints_0, STpoints_1;
    ST_transform(Tpoints_0, s_0, STpoints_0);
    ST_transform(Tpoints_1, s_1, STpoints_1);

    // Transformed and scaled points at image 0: STpoints_0
    // Transformed and scaled points at image 0: STpoints_1


    // Step 4: Normalise the points
    std::vector<Vector3D> nSTpoints_0, nSTpoints_1;
    homo_coordinates(STpoints_0, nSTpoints_0);
    homo_coordinates(STpoints_1, nSTpoints_1);

    // normalised points of image 0: nSTpoints_0
    // normalised points of image 1: nSTpoints_1


    // Step 5: Compute Fundametal Matrix
    Matrix W(points_0.size(), 9,0.0);
    Matrix F;
    compute_F(nSTpoints_0, nSTpoints_1, W, F);
    std::cout << F << std::endl;

    // Check Det(F) = 0
    std::cout << determinant(F) << std::endl; //very very close to zero, but not zero

    //  Step 6: Calculate T1, and T2 (slide 14) think about how you are integrating it to F
    Matrix O_matrix_0(points_0.size(),3,0.0);
    Matrix O_matrix_1(points_1.size(),3,0.0);
    Matrix T_matrix_0(points_0.size(), 3,0.0); // Transformed T0
    Matrix T_matrix_1(points_1.size(), 3,0.0); // Transformed T1

    transformation_matrix(points_0, STpoints_0, O_matrix_0, T_matrix_0);
    transformation_matrix(points_1, STpoints_1, O_matrix_1, T_matrix_1);

    Matrix33 new_F = O_matrix_0.transpose() * F * T_matrix_0;

    // Step 7: integrating into F to find new_F - "new_F = O_matrix_0.transpose() * F * T_matrix_0;"
    std::cout << new_F << std::endl;

//    7 - scale scale_invariant F   where F(2,2) = 1.

//    8 - calculate E and 4 Rt settings from it (slide 20 -27)

//    9 - triangulate & compute inliers (slide 27)

//    10 - choose best RT setting & evaluate

//    Denormalise(F)





    //      - compute the essential matrix E;




    //      - recover rotation R and t.




    // TODO: Reconstruct 3D points. The main task is
    //      - triangulate a pair of image points (i.e., compute the 3D coordinates for each corresponding point pair)

    // TODO: Don't forget to
    //          - write your recovered 3D points into 'points_3d' (so the viewer can visualize the 3D points for you);
    //          - write the recovered relative pose into R and t (the view will be updated as seen from the 2nd camera,
    //            which can help you check if R and t are correct).
    //       You must return either 'true' or 'false' to indicate whether the triangulation was successful (so the
    //       viewer will be notified to visualize the 3D points and update the view).
    //       There are a few cases you should return 'false' instead, for example:
    //          - function not implemented yet;
    //          - input not valid (e.g., not enough points, point numbers don't match);
    //          - encountered failure in any step.


    return points_3d.size() > 0;
}