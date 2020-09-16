#include "refinement.h"
#include "partitioning.h"
#include <cmath>        // round (ceil)
#include "etimer.h"
#include "loader.h"
#include "common_funcs.h"
#include "solver.h"
#include "k_fold.h"
#include <cassert>

#include <thread>

#include <flann/flann.hpp>	//FLANN

#define debug_level  0
#define debug_iter  2
#define debug_next_level  0

using std::cout; using std::endl;

struct selected_agg
{
    int index;
    double value;

    selected_agg(int col_index, double fraction_val) :
        index(col_index), value(fraction_val) {}

    bool operator > (const selected_agg sel_agg) const {
        return (value > sel_agg.value);
    }
};

struct Better_Gmean_SN
{
//    bool operator () (const summary& a, const summary& b) const
    bool operator()(const ref_results& A, const ref_results& B) const
    {
        summary a = A.validation_data_summary;
        summary b = B.validation_data_summary;
        float filter_range = 0.02;
        // a has completely better gmean than b
        if( (a.perf.at(Gmean) - b.perf.at(Gmean)) > filter_range )
            return true;
        else{
            // b has completely better gmean than a
            if( (b.perf.at(Gmean) - a.perf.at(Gmean)) > filter_range )
                return false;
            else{                               //similar gmean
                if(paramsInst->get_ms_best_selection() == 1)
                    // a has less nSV than b which is better
                    return (a.perf.at(Sens) >  b.perf.at(Sens) );
                else
                    // the gmeans  are similar and we don't care for nSV  ???
                    return false;
            }
        }
    }
};

struct Better_Gmean_SN_nSV
{
//    bool operator () (const summary& a, const summary& b) const
    bool operator()(const ref_results& A, const ref_results& B) const
    {
        summary a = A.validation_data_summary;
        summary b = B.validation_data_summary;
        float filter_range = 0.02;
        //a has completely better gmean than b
        if( (a.perf.at(Gmean) - b.perf.at(Gmean)) > filter_range )
            return true;
        else{
            //b has completely better gmean than a
            if( (b.perf.at(Gmean) - a.perf.at(Gmean)) > filter_range )
                return false;
            else{                               //similar gmean
                if ( a.perf.at(Sens) -  b.perf.at(Sens) > filter_range ) // we have a winner
                    return true;
                else{
                // if b has higher sensitivity, return false to select b
                    if (b.perf.at(Sens) -  a.perf.at(Sens) > filter_range)
                        return false;
                    else // return true if number of (A's SV < B's SV)
                        return (a.num_SV_p + a.num_SV_n <  b.num_SV_p + b.num_SV_n );
                }
            }
        }
    }
};


// Needed for using FLANN to find nearest neighbors of misclassified points
void run_flann(Mat& m_data, Mat& m_indices, Mat& m_dists, int num_nn) {
    // - - - - load the data into flann matrix - - - -
    PetscInt num_row, num_col;
    PetscInt i, j, ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;
//    int num_nn = Config_params::getInstance()->get_nn_number_of_neighbors();
    
    MatGetSize(m_data, &num_row, &num_col);

    assert(("Number of rows in the data are zero which is a problem!", num_row!=0));
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row, num_nn, num_nn, PETSC_NULL, &m_indices);
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row, num_nn, num_nn, PETSC_NULL, &m_dists);
//    std::cout << "[KNN][RF] num_nn:"<< num_nn << std::endl;

    std::vector<float>vecData;

    for(i =0; i <num_row; i++){
        int tmp_nnz_indices=0;
        // std::cout << "\nrow " << i <<": ";
        std::vector<float> v_nn_vals;
        MatGetRow(m_data,i,&ncols,&cols,&vals);
        for (j=0; j<ncols; j++) {
            if(j < ncols && j==cols[tmp_nnz_indices]){
                v_nn_vals.push_back(vals[j]);
                tmp_nnz_indices++;
            }else{
                v_nn_vals.push_back(0);
            }
            // std::cout << "(" << j << ", "<< vals[j]  << ")  ";
        }
        MatRestoreRow(m_data,i,&ncols,&cols,&vals);
        vecData.insert(vecData.end(), v_nn_vals.begin(), v_nn_vals.end());
    }
    flann::Matrix<float> data(vecData.data(),num_row,num_col);
//    std::cout << "[KNN][RF] "<< num_row << " data points are loaded successfully!\n";

    // call flann
    flann::Index<flann::L2<float> > index_(data, flann::KDTreeIndexParams(1));
    index_.buildIndex();

    flann::Matrix<int> indicies(new int[num_row * num_nn], num_row, num_nn);
    flann::Matrix<float> dists(new float[num_row * num_nn], num_row, num_nn);

    flann::SearchParams params(64);
//    flann::seed_random(0);          //set the random seed to 1 for debug

    params.cores = 0; //automatic core selection
    index_.knnSearch(data, indicies, dists, num_nn, params);

    //store the indices, dists to 2 separate matrices
    for(int row_idx = 0; row_idx < num_row; row_idx++){
        // std::cout << "\nrow " << row_idx <<": ";
        for(j = 0; j < num_nn; j++){
            unsigned int node_idx = indicies[row_idx][j];
            double dist = dists[row_idx][j];
            // std::cout << "(" << j << ", "<< node_idx << " - " << dist << ")  ";
            MatSetValue(m_indices,row_idx,j,node_idx,INSERT_VALUES);
            MatSetValue(m_dists,row_idx,j,dist,INSERT_VALUES);
        }
    }
    MatAssemblyBegin(m_indices,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_indices,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(m_dists,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_dists,MAT_FINAL_ASSEMBLY);
}//

// Run FLANN on combined training data and misclassified validation data matrices
// Then find nearest neighbor for each misclassified point from training data and add to set
//int find_misclassified_neighbors(Mat& m_td, Mat& m_vd, std::vector<int>& mc_indices, std::set<int>& aug_neighbors, int flann_nn_count) {
PetscInt find_misclassified_neighbors(Mat& m_td, Mat& m_vd, std::vector<PetscInt>& mc_indices, std::set<PetscInt>& aug_neighbors, int flann_nn_count) {
    CommonFuncs cf;
    Mat m_td_and_vd, m_mc_vd, m_nn_indices, m_nn_dists;
    IS is_mc_indices;

    PetscInt num_td_rows;//, num_;	// $Debug Not sure why 'num_' is here - potentially intended to be removed and forgotten?
    MatGetSize(m_td, &num_td_rows, NULL);

    mc_indices.shrink_to_fit();
    PetscInt num_mc = mc_indices.size();
    ISCreateGeneral(PETSC_COMM_SELF, num_mc, mc_indices.data(), PETSC_COPY_VALUES, &is_mc_indices);
    MatGetSubMatrix(m_vd, is_mc_indices, NULL, MAT_INITIAL_MATRIX, &m_mc_vd);

    cf.combine_Mats(m_td, m_mc_vd, m_td_and_vd);
    run_flann(m_td_and_vd, m_nn_indices, m_nn_dists, flann_nn_count);

    const PetscInt *cols;
    const PetscScalar *indices_vals, *dists_vals;
    PetscInt i, j;
    PetscInt num_new_neighbors = 0;

    for(i=num_td_rows; i<num_td_rows+num_mc; i++) {
        MatGetRow(m_nn_indices, i, NULL, NULL, &indices_vals);
        MatGetRow(m_nn_dists, i, NULL, NULL, &dists_vals);

        // First neighbor index (j=0) for a point is always itself, so skip it
        for(j=1; j<flann_nn_count; j++) {
            PetscInt idx_val = indices_vals[j];
            PetscScalar dist_val = dists_vals[j];

            // Make sure index is in training data
            if(dist_val != 0 && idx_val < num_td_rows) {
                auto result = aug_neighbors.insert(idx_val);
                if(result.second) {
                    num_new_neighbors++;
//                    break;
                }
                break;
            }
        }
        MatRestoreRow(m_nn_indices, i, NULL, NULL, &indices_vals);
        MatRestoreRow(m_nn_dists, i, NULL, NULL, &dists_vals);
    }

    MatDestroy(&m_mc_vd);
    MatDestroy(&m_td_and_vd);
    MatDestroy(&m_nn_indices);
    MatDestroy(&m_nn_dists);
    ISDestroy(&is_mc_indices);

//    printf("num_new_neighbors= %d\n", num_new_neighbors);
    return num_new_neighbors;
}




solution Refinement::main(Mat& m_data_p, Mat& m_P_p, Vec& v_vol_p, Mat&m_WA_p
                          , Mat& m_data_n, Mat& m_P_n, Vec& v_vol_n, Mat&m_WA_n
                          , Mat& m_VD_p, Mat& m_VD_n
                          , solution& sol_coarser, int level
                          , std::vector<ref_results>& v_ref_results){


    paramsInst->set_main_current_level_id(level);
    PetscInt num_row_p_data =0, num_row_n_data =0, num_col_data =0;
//    MatGetSize(m_data_p,&num_row_p_data,NULL);


    MatGetSize(m_data_p,&num_row_p_data,&num_col_data);
    MatGetSize(m_data_n,&num_row_n_data,NULL);


    if (level == debug_next_level){ cout << "exit [RF] 230:" << endl; exit(1);}


    // ------------- Debug for adding feature selection -----------
/*
    printf("[RF][main]{beginnig} num fine data p:%d, n:%d\n"
           ,num_row_p_data,num_row_n_data);


    PetscInt num_row_p_P =0, num_row_n_P =0, num_col_p_P =0, num_col_n_P =0;
    MatGetSize(m_P_p,&num_row_p_P, &num_col_p_P);
    MatGetSize(m_P_n,&num_row_n_P, &num_col_n_P);
    printf("[RF][main]{beginnig} Minority class P matrix dim [%d,%d]\n"
           ,num_row_p_P,num_col_p_P);
    printf("[RF][main]{beginnig} Majority class P matrix dim [%d,%d]\n"
           ,num_row_n_P,num_col_n_P);

    printf("[RF][main] sol_coarser.p_index.size():%lu, "
           "sol_coarser.n_index.size():%lu\n"
           , sol_coarser.p_index.size()
           , sol_coarser.n_index.size());
*/

    if (level == debug_next_level){ cout << "exit [RF] 253:" << endl; exit(1);}
#if dbl_RF_exp_data_ml == 0 /// - - -  normal case  - - -
#if dbl_RF_main >=5
// Moved these above
//    PetscInt num_row_p_data =0, num_row_n_data =0;
//    MatGetSize(m_data_p,&num_row_p_data,NULL);
//    MatGetSize(m_data_n,&num_row_n_data,NULL);
    printf("[RF][main]{beginnig} num fine data p:%d, n:%d\n"
           ,num_row_p_data,num_row_n_data);


    PetscInt num_row_p_P =0, num_row_n_P =0, num_col_p_P =0, num_col_n_P =0;
    MatGetSize(m_P_p,&num_row_p_P, &num_col_p_P);
    MatGetSize(m_P_n,&num_row_n_P, &num_col_n_P);
    printf("[RF][main]{beginnig} Minority class P matrix dim [%d,%d]\n"
           ,num_row_p_P,num_col_p_P);
    printf("[RF][main]{beginnig} Majority class P matrix dim [%d,%d]\n"
           ,num_row_n_P,num_col_n_P);

    printf("[RF][main] sol_coarser.p_index.size():%lu, "
           "sol_coarser.n_index.size():%lu\n"
           , sol_coarser.p_index.size()
           , sol_coarser.n_index.size());
#endif
    // these are fix for all levels as the sum
    // of volumes for all the points are preserved
    PetscScalar sum_all_vol_p,sum_all_vol_n;
    VecSum(v_vol_p, &sum_all_vol_p);
    VecSum(v_vol_n, &sum_all_vol_n);


    Mat m_new_neigh_p, m_new_neigh_n;
    IS IS_neigh_p, IS_neigh_n;

    ref_results refinement_results;

    solution sol_refine;
    // this cause the SV to increase after each augmentation
    // Ehsan reset it inside the while loop to for each try tp only store
    // the current SV indices  112619-2057


    // Enable augmenting training set with nearest neighbors of misclassified validation points
//    bool augment_with_neighbors = true;
    bool augment_with_neighbors = Config_params::getInstance()->get_rf_aug_nn();

    // Enable augmenting training set with nearest neighbors from both classes of misclassified validation points
    // False: finds a nearest neighbor only from the corresponding class (At most, one nearest neighbor per misclassified point)
    // True: finds a nearest neighbor from the corresponding and opposite classes (At most, two nearest neighbors per misclassified point)
//    bool augment_with_both_neighbors = true;
    bool augment_with_both_neighbors = Config_params::getInstance()->get_rf_aug_both_classes();


    bool allow_augment = false;	// Initial value; updated elsewhere based on num of new neighbors found
    bool should_loop = true;
    bool looped = false;

    int augment_count = 0;	// Number of times new models have been trained using augmented training data
    int loop_count = 0;

    // Currently limiting to up to one time based on num of new neighbors found instead
    int max_loops = 5;	// Arbitrary max iterations of the loop for right now

    bool use_flann = true;
    int flann_nn_count = 5;


    // For using ISs to store the found neighbors   $Performance?
//    IS IS_aug_neigh_indices_p, IS_aug_neigh_indices_n;

    // For using a STL set to store the found neighbors   $Performance?
//    std::set<int> aug_neigh_indices_p, aug_neigh_indices_n;
    std::set<PetscInt> aug_neigh_indices_p, aug_neigh_indices_n;
 
    std::vector<ref_results> v_extra_ref_results;

    // Start of while loop for potentially adding neighbors of misclassified points to the set of neighbors of support vectors
    while(should_loop && loop_count < max_loops) {
        should_loop = false;

    /// - - - - - - - get new points for finer level - - - - - - -
/*  Old signature calls
    find_SV_neighbors(m_data_p,m_P_p,coarser_p_index, m_WA_p
                      , m_new_neigh_p,"Minority",IS_neigh_p);
    find_SV_neighbors(m_data_n,m_P_n,coarser_n_index, m_WA_n
                      , m_new_neigh_n,"Majority",IS_neigh_n);
*/

//    find_SV_neighbors(m_data_p,m_P_p,sol_coarser.p_index, m_WA_p
//                      , m_new_neigh_p,"Minority", IS_neigh_p, aug_neigh_indices_p);
//    find_SV_neighbors(m_data_n,m_P_n,sol_coarser.n_index, m_WA_n
//                      , m_new_neigh_n,"Majority", IS_neigh_n, aug_neigh_indices_n);

    find_SV_neighbors(m_data_p,m_P_p,sol_coarser.p_index, m_WA_p
                      , m_new_neigh_p,"Minority", IS_neigh_p, aug_neigh_indices_p
                      , sol_coarser);
    find_SV_neighbors(m_data_n,m_P_n,sol_coarser.n_index, m_WA_n
                      , m_new_neigh_n,"Majority", IS_neigh_n, aug_neigh_indices_n
                      , sol_coarser);

//    v_extra_ref_results.trackSV.p_index =

    if (level == debug_level){ cout << "exit [RF] 351:" << endl; exit(1);}
    if (level == debug_next_level){ cout << "exit [RF] 352:" << endl; exit(1);}
    int iteration = paramsInst->get_main_num_kf_iter();

    // - - - - get the size of neighbors - - - -
    PetscInt num_neigh_row_p_ =0, num_neigh_row_n_ =0;
    MatGetSize(m_new_neigh_p,&num_neigh_row_p_,NULL);
    MatGetSize(m_new_neigh_n,&num_neigh_row_n_,NULL);


    printf("[RF][main][loop: %d] num new neighbor p:%d, n:%d\n"
           ,loop_count,num_neigh_row_p_,num_neigh_row_n_);
    printf("[RF][main][loop: %d] num coarse SV p:%d, n:%d\n"
           ,loop_count,sol_coarser.p_index.size(),sol_coarser.n_index.size());

//    if(num_neigh_row_p_ == 0 || num_neigh_row_n_ == 0){
//        printf("\n[Error]:[MR][refinement] Empty matrices for new neighbors,\n Exit!\n");
//        exit(1);
//    }
    assert (num_neigh_row_p_ && "[MR][refinement] Empty matrices for new neighbors");
    assert (num_neigh_row_n_ && "[MR][refinement] Empty matrices for new neighbors");

    // the solution SV vectors should reset for each try    112619-2057

//    solution sol_refine;	// Moved above
    sol_refine.p_index.clear();
    sol_refine.n_index.clear();

    summary summary_TD;
    /***********************************************************************************************/
    /*                                  Start Partitioning                                         */
    /***********************************************************************************************/

    /*  - The information for partitioning is related to points in the finer level(current)
     *      which are going to pass to the SVM to creat a model
     *  - The WA, Vol contains all the points which are not needed.
     *  - Only the information for the points which are neighbor to SV of coarser level are required
     *  m_neigh_WA: only specific rows and columns
     *  v_nei_vol: volumes for the points in the m_neigh_WA
     */

//    cout << "exit [RF] 378:" << endl; exit(1);
    if( (num_neigh_row_p_ + num_neigh_row_n_)  > paramsInst->get_pr_start_partitioning() ){


        // Terminate iteration
       MatDestroy(&m_P_p);
       MatDestroy(&m_P_n);
       MatDestroy(&m_WA_p);
       MatDestroy(&m_WA_n);
       VecDestroy(&v_vol_p);
       VecDestroy(&v_vol_n);


       v_extra_ref_results.push_back(refinement_results);

       int best_result_idx = 0;
       int num_extra_results = v_extra_ref_results.size();
       cout << num_extra_results << endl;
       if (num_extra_results == 1){
            cout << "[RF][MAIN] There is no result here" << endl;
       }else{
            cout << "[RF][MAIN] reporting the best of results before"
                 << " early termination" << endl;
            std::sort(v_extra_ref_results.begin(), v_extra_ref_results.end(), 
                      Better_Gmean_SN_nSV());
            v_ref_results.push_back(v_extra_ref_results[0]);	// Best result in v_extra_ref_results should now be at idx 0
       }
       solution curr_refine_solution;

       curr_refine_solution.terminated_early = true;
       return curr_refine_solution;

       /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        *
        *                       EARLY TERMINATION
        *
        * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    }
    else
    {
        /***********************************************************************************************/
        /*                                     No Partitioning                                         */
        /***********************************************************************************************/
#if dbl_RF_main_no_partition >=1
        printf("\n[RF][main][loop: %d] * * * * No Partitioning, level:%d * * * * \n",loop_count,level);
#endif

        if(paramsInst->get_ms_status() &&
            (num_neigh_row_p_ + num_neigh_row_n_) < paramsInst->get_ms_limit()    ){

            if (level == debug_level){ cout << "exit [RF] 694:" << endl; exit(1);}

            cout << "[RF] DEBUG L703 before UDSV sol_refine: nSV_p:" << sol_refine.p_index.size()
                 << ",nSV_n:" << sol_refine.n_index.size() << endl;

            // ------- call Model Selection (SVM) -------
            ModelSelection ms_refine;
            ms_refine.uniform_design_separate_validation(m_new_neigh_p, v_vol_p
                                                         , m_new_neigh_n
                                                         , v_vol_n, true
                                                         , sol_coarser.C
                                                         , sol_coarser.gamma
                                                         , m_VD_p, m_VD_n
                                                         , level, sol_refine
                                                         , refinement_results);
//                                                         , v_ref_results);	// No longer passing the full vector

            // added by Ehsan 112619-1900 to track the SV from the best try not the last one

            refinement_results.trackSV.p_index = sol_refine.p_index;
            refinement_results.trackSV.n_index = sol_refine.n_index;

            cout << "[RF] DEBUG L723 after UDSV: nSV_p:" << refinement_results.trackSV.p_index.size()
                 << ",nSV_n:" << refinement_results.trackSV.n_index.size() << endl;

            if (level == debug_level){ cout << "exit [RF] 706:" << endl; exit(1);    }

#if dbl_RF_main_no_partition >=1
            cout << "[RF]{no partitioning} ms_active uniform design is finished!\n";
#endif

        }else{  // No model selection either because it is disabled or the threshold is reached
            Solver sv_refine;
            svm_model * trained_model;
            trained_model = sv_refine.train_model(m_new_neigh_p, v_vol_p
                                                  , m_new_neigh_n
                                                  , v_vol_n, true
                                                  , sol_coarser.C
                                                  , sol_coarser.gamma) ;
            /* Added 091618_1821
            The validation is required to add this run to the rest of
            the models for different levels which I can select the best
            one in the end of refinement.
            */

            //map that contains all the measures
            summary current_val_summary;
            sv_refine.predict_validation_data(m_VD_p, m_VD_n
                                              , current_val_summary
                                              , 0, true);
            sv_refine.evaluate_testdata(level,summary_TD);

// As above, handling these differently now
//            ref_results refinement_results ;
            refinement_results.validation_data_summary = current_val_summary;
            refinement_results.test_data_summary = summary_TD;
            refinement_results.level = level;
            cout << "WARNING: no model selection for parameter fitting!"<< endl;

            sv_refine.prepare_solution_single_model(trained_model,num_neigh_row_p_,sol_refine);

            // added by Ehsan 112619-1900 to track the SV from the best try not the last one
            refinement_results.trackSV.p_index = sol_refine.p_index;
            refinement_results.trackSV.n_index = sol_refine.n_index;

            sv_refine.free_solver("[RF]");
            // models are exported in the solver class if it is needed


        }

    #if dbl_RF_main >=5
        if(level > 1 ){         //skip this info for the finest level (since in the MS, they are not provided)
            printf("[RF][main][loop: %d] sol_refine.C:%g, sol_refine.gamma:%g\n",
                   loop_count,sol_refine.C,sol_refine.gamma);
            printf("[RF][main][loop: %d] sol_refine.p size:%lu, sol_refine.n size:%lu\n",
                   loop_count,sol_refine.p_index.size() ,sol_refine.n_index.size());
        }
    #endif

// Moved further below to ensure they are only destroy when no longer needed
//        MatDestroy(&m_WA_p);
//        MatDestroy(&m_WA_n);
//        VecDestroy(&v_vol_p);
//        VecDestroy(&v_vol_n);
    }
    // free ISs for both situation with or without partitioning
    ISDestroy(&IS_neigh_p);
    ISDestroy(&IS_neigh_n);


        // Mark the results as having been augmented with additional neighbors
        if(looped) {
            refinement_results.validation_data_summary.augmented = true;
            refinement_results.test_data_summary.augmented = true;
        }


        /*                                                                 *
         * CODE TO ADD NEIGHBORS OF MISCLASSIFIED VALIDATION POINTS STARTS *
         *                 vvvvvvvvvvv BELOW vvvvvvvvvv                    */

        if(augment_count == 0 && num_neigh_row_p_ + num_neigh_row_n_ < 50000) {
            // Allow augmentation, if needed
            augment_count++;
            allow_augment = true;
        }
        else if(augment_count && num_neigh_row_p_ + num_neigh_row_n_ >= 50000) {
            // Terminate iteration
            MatDestroy(&m_P_p);
            MatDestroy(&m_P_n);
            MatDestroy(&m_WA_p);
            MatDestroy(&m_WA_n);
            VecDestroy(&v_vol_p);
            VecDestroy(&v_vol_n);


            v_extra_ref_results.push_back(refinement_results);

            int best_result_idx = 0;
            int num_extra_results = v_extra_ref_results.size();

//            std::sort(v_extra_ref_results.begin(), v_extra_ref_results.end(), Better_Gmean_SN());
            std::sort(v_extra_ref_results.begin(), v_extra_ref_results.end(), Better_Gmean_SN_nSV());
            // Currently just finds the result with the best g-mean, and really only needed when the max_loop_count is reached
/*            for(int i=0; i<num_extra_results; i++) {
                if(v_extra_ref_results[i].validation_data_summary.perf[Gmean]
		   > v_extra_ref_results[best_result_idx].validation_data_summary.perf[Gmean])
                    best_result_idx = i;
            }
*/
            // Only add best result to v_ref_results
//            v_ref_results.push_back(v_extra_ref_results[best_result_idx]);
            v_ref_results.push_back(v_extra_ref_results[0]);	// Best result in v_extra_ref_results should now be at idx 0
//            v_ref_results.push_back(refine_results);

//            sol_refine.C = -1;
//            sol_refine.gamma = 0;
            sol_refine.terminated_early = true;
            return sol_refine;
        }

        // Add ref_results to a vector that holds all results found at this refinement level
        v_extra_ref_results.push_back(refinement_results);

        // Determine if new solution is significantly lower quality than previous/best level's solution
        int results_size = v_ref_results.size();


        // TODO CONVERT BELOW TO USING CMD LINE PARAMETERS!!!!!
//        double drop_threshold = 0.05;
//        double ratio = 1.0;	// Ratio of misclassified points to add
        double drop_threshold = Config_params::getInstance()->get_rf_aug_drop_threshold();
        double ratio = Config_params::getInstance()->get_rf_aug_nn_ratio();	// Ratio of misclassified points to add

        bool drop_from_best = Config_params::getInstance()->get_rf_aug_drop_from_best();
//        bool drop_from_best = false;
        double coarser_sol_perf;		// Results from previous/best level
//        double coarser_sol_perf = v_ref_results[results_size-1].validation_data_summary.perf[Gmean];	// Results from previous level

        if(!drop_from_best) {
            // Validation results from previous level
            coarser_sol_perf = v_ref_results[results_size-1].validation_data_summary.perf[Gmean];
        }
        else {
            // Need to implement -> Now verify/validate works correctly
//            std::sort(v_ref_results.begin(), v_ref_results.end(), Better_Gmean_SN());	// Sort v_ref_results => element @ idx 0 should be best of current iter

            // Sort v_ref_results => element @ idx 0 should be best of current iter
            std::sort(v_ref_results.begin(), v_ref_results.end(), Better_Gmean_SN_nSV());

            // Validation results from best level
            coarser_sol_perf = v_ref_results[0].validation_data_summary.perf[Gmean];
        }

        // Most recent validation results from this level
        double curr_sol_perf = refinement_results.validation_data_summary.perf[Gmean];

        /* ---------------------------------------------------------------------- *
         *                                Augmentation
         * ---------------------------------------------------------------------- *
         */
        if (iteration == debug_iter){ cout << "exit [RF] 861:" << endl; exit(1);}

        if(allow_augment && augment_with_neighbors && loop_count == 0 &&
                coarser_sol_perf - curr_sol_perf >= drop_threshold) {

            ETimer augment;
            CommonFuncs cf;

            allow_augment = false;

            printf("\n[RF][main][loop: %d] l:%d, Quality drop >= %.3f: coarser- %.3f -- current- %.3f; Adding neighbors of misclassified validation points\n",
            loop_count, level, drop_threshold, coarser_sol_perf, curr_sol_perf);

            int curr_loop_nn_added_p=0, curr_loop_nn_added_n=0;

            /*  Positive class  */
            // Get misclassified validation point indices from current level
//            std::vector<int> misclassified_p((std::vector<int>)refinement_results.validation_data_summary.v_fn_idx);
            std::vector<PetscInt> &misclassified_p = refinement_results.validation_data_summary.v_fn_idx;

            if (iteration == debug_iter){ cout << "exit [RF] 870:" << endl; exit(1);}
            if(misclassified_p.size() > 0) {
                // Find nearest neighbor in current level's training data for each/some amount
                int max_num_points_to_add = misclassified_p.size() * ratio;

                // For ratios <1.0, need to decide on a way to determine which misclassified points to find neighbors for
                // -Start at first and continue sequentially until hit max?
                // -Randomly select them?
                // -Select an even distribution?

                // Finds nearest neighbor in training data for each misclassified validation point and adds the idx to a std::set

                // Use FLANN; Add neighbor from same class:
                // Positive training data and positive found neighbor idx set for 1st and 4th args, respectively
                curr_loop_nn_added_p += find_misclassified_neighbors(m_data_p, m_VD_p
                                                                     , misclassified_p, aug_neigh_indices_p
                                                                     , flann_nn_count);

                // Add a neighbor from the opposite class as well:
                // Negative training data and negative found neighbor idx set for 1st and 4th args, respectively
                if(augment_with_both_neighbors) {
                    curr_loop_nn_added_n += find_misclassified_neighbors(m_data_n, m_VD_p
                                                                     , misclassified_p, aug_neigh_indices_n
                                                                     , flann_nn_count);
//                    }
                }
            } // End Positive class

            if (iteration == debug_iter){ cout << "exit [RF] 897:" << endl; exit(1);}

            /*  Negative class  */
            // Get misclassified validation point indices from current level
//            std::vector<int> misclassified_n((std::vector<int>)refinement_results.validation_data_summary.v_fp_idx);
            std::vector<PetscInt> &misclassified_n = refinement_results.validation_data_summary.v_fp_idx;

            if(misclassified_n.size() > 0) {
                // Find nearest neighbor in current level's training data for each/some amount
                int max_num_points_to_add = misclassified_n.size() * ratio;

                // For ratios <1.0, need to decide on a way to determine which misclassified points to find neighbors for
                // -Start at first and continue sequentially until hit max?
                // -Randomly select them?
                // -Select an even distribution?

                // Use FLANN; Add neighbor from same class:
                // Negative training data and negative found neighbor idx set for 1st and 4th args, respectively
                curr_loop_nn_added_n += find_misclassified_neighbors(m_data_n, m_VD_n
                                                        , misclassified_n, aug_neigh_indices_n
                                                        , flann_nn_count);

                // Add a neighbor from the opposite class as well:
                // Positive training data and positive found neighbor idx set for 1st and 4th args, respectively
                if(augment_with_both_neighbors) {
                    curr_loop_nn_added_p += find_misclassified_neighbors(m_data_p, m_VD_n
                                                        , misclassified_n, aug_neigh_indices_p
                                                        , flann_nn_count);
                }
//                }
            } // End Negative class

            printf("[RF][main][loop: %d] l:%d, Added %d new finer positive class neighbors, %d in total this level.\n", loop_count, level,
			curr_loop_nn_added_p, aug_neigh_indices_p.size());
            printf("[RF][main][loop: %d] l:%d, Added %d new finer negative class neighbors, %d in total this level.\n", loop_count, level,
			curr_loop_nn_added_n, aug_neigh_indices_n.size());

            // Indicate that code should loop, update count
            should_loop = curr_loop_nn_added_p + curr_loop_nn_added_n > 0 ? true : false;
//            should_loop = true;
            if(!looped) looped = true;
            loop_count++;

            augment.stop_timer("[RF][main] Finding nearest neighbors for augmenting at level ",
                               std::to_string(level));
            printf("\n");
        }
    }	// end of while(should_loop) (starts at line 73)



    // Find best result for this level
    // Only gets this far if found a better solution or max_loops was reached
    // In the latter case, need to find best result of those found.
    if(looped) {
/*
        int best_result_idx = 0;
        int num_extra_results = v_extra_ref_results.size();
        // Currently just finds the result with the best g-mean, and really only needed when the max_loop_count is reached
        for(int i=0; i<num_extra_results; i++) {
            if(v_extra_ref_results[i].validation_data_summary.perf[Gmean] > v_extra_ref_results[best_result_idx].validation_data_summary.perf[Gmean])
                best_result_idx = i;
        }
        // Only add best result to v_ref_results
        v_ref_results.push_back(v_extra_ref_results[best_result_idx]);
*/
//        cout << "exit [RF] 1067:" << endl; exit(1);

        // Sort v_extra_ref_results so best model found this level is at idx 0
        std::sort(v_extra_ref_results.begin(), v_extra_ref_results.end(), Better_Gmean_SN());
        v_ref_results.push_back(v_extra_ref_results[0]);


        sol_refine.p_index = v_extra_ref_results[0].trackSV.p_index;
        sol_refine.n_index = v_extra_ref_results[0].trackSV.n_index;
        cout << "[RF] DEBUG L995: 0 nSV_p:" << v_extra_ref_results[0].trackSV.p_index.size()
             << ",nSV_n:" << v_extra_ref_results[0].trackSV.n_index.size() << endl;

        cout << "[RF] DEBUG L998: 1 nSV_p:" << v_extra_ref_results[1].trackSV.p_index.size()
             << ",nSV_n:" << v_extra_ref_results[1].trackSV.n_index.size() << endl;

        cout << "__DB__ [RF] Best selected result has val G-mean:"
             << v_extra_ref_results[0].validation_data_summary.perf[Gmean]
             << " test G-mean:"
             << v_extra_ref_results[0].test_data_summary.perf[Gmean] << endl;

        cout << "__DB__ [RF] improved VAL G-mean:"
             << v_extra_ref_results[1].validation_data_summary.perf[Gmean]
             << " to:"
             << v_extra_ref_results[0].validation_data_summary.perf[Gmean]
             << endl;

        cout << "__DB__ [RF] improved TEST G-mean:"
             << v_extra_ref_results[1].test_data_summary.perf[Gmean]
             << " to:"
             << v_extra_ref_results[0].test_data_summary.perf[Gmean]
             << endl;
    }
    else  {	// Never looped; first result did not drop more than the threshold
        v_ref_results.push_back(refinement_results);
    }

    // DONE LOOPING

//    cout << "exit [RF] 1077:" << endl; exit(1);

    // Can't destroy following until done looping
    // Previously in find_SV_neighbors()
    MatDestroy(&m_P_p);
    MatDestroy(&m_P_n);

    // Previously above in partitioning section
    MatDestroy(&m_WA_p);
    MatDestroy(&m_WA_n);
    VecDestroy(&v_vol_p);
    VecDestroy(&v_vol_n);

//    cout << "exit [RF] 1080:" << endl; // exit(1);

    return sol_refine;


#else   // dbl_RF_exp_data_ml /// - - -  debug/experimental cases  - - -
    k_fold kf;
    Mat m_train_data_label;
    // false: don't destroy the input matrices inside the function
    // the data files are destroyed in the MainRecursion class
    kf.combine_two_classes_in_one(m_train_data_label, m_data_p, m_data_n,
                                  false);
    std::string out_prefix = paramsInst->get_exp_info() +
            "_exp:" + std::to_string(paramsInst->get_main_current_exp_id()) +
            "_kf:" + std::to_string(paramsInst->get_main_current_kf_id()) +
            "_level:"+std::to_string(paramsInst->get_main_current_level_id());
    std::string out_train_label_fname = out_prefix + "_traindata_label.dat";
    std::string out_min_vol_fname = out_prefix + "_min_vol.dat";
    std::string out_maj_vol_fname = out_prefix + "_maj_vol.dat";

    CommonFuncs cf;
    cf.exp_matrix(m_train_data_label, "./debug/", out_train_label_fname,
                  "[RF][main]");
    cf.exp_vector(v_vol_p, "./debug/", out_min_vol_fname, "[RF][main]");
    cf.exp_vector(v_vol_n, "./debug/", out_maj_vol_fname, "[RF][main]");
    cout << "DEBUG refinement, no classification!" << endl;
    if (level < 3){
        cout << "[RF][PCL] There is a problem of double free for the 1st level"<<
		", exit manually" << endl;
        exit(1);
    }
#endif
}



/*
 * @input:
 *      cc_name: class name used for logging information
 * @output:
 *      m_neighbors is the matrix of neighbor points
 *      neigh_id is IS type which contain the indices for neighbor points
 */

/* Old signature
void Refinement::find_SV_neighbors(Mat& m_data, Mat& m_P, std::vector<int>& seeds_ind,
                                      Mat& m_SV, Mat& m_neighbors, std::string cc_name,
                                      IS& IS_neigh_id){
*/
//void Refinement::find_SV_neighbors(Mat& m_data, Mat& m_P, std::vector<int>& seeds_ind,
//                                      Mat& m_WA, Mat& m_neighbors, std::string cc_name,
//                                      IS& IS_neigh_id, std::set<int>& aug_indices){

void Refinement::find_SV_neighbors(Mat& m_data, Mat& m_P, std::vector<int>& seeds_ind,
                                      Mat& m_WA, Mat& m_neighbors, std::string cc_name,
                                      IS& IS_neigh_id, std::set<PetscInt>& aug_indices,
                                      solution& sol_coarsest){

    // create the index set to get the sub matrix in the end
    PetscInt        * ind_;         //arrays of Int that contains the row indices
    PetscInt        num_row_fine_points=0;
    unsigned int num_seeds;
    PetscInt ncols=0;
    const PetscInt    *cols;                        //if not NULL, the column numbers
    const PetscScalar *vals;

    MatGetSize(m_data,&num_row_fine_points,NULL);
#if dbl_RF_FSN >=5
    cout << "[RF][FSN]{" << cc_name << "} m_data num row as num_row_fine_points:"<< num_row_fine_points <<endl;
#endif
    /// - - - - - - - - - Create P Transpose matrix - - - - - - - -
    // P' : find fine points in rows instead of columns due to performance issues with Aij matrix
    num_seeds = (int) seeds_ind.size();
#if dbl_RF_FSN >=3
    cout  << "[RF][FSN]{" << cc_name << "} initialize num_seeds:" << num_seeds << "\n";
#endif
    PetscMalloc1(num_row_fine_points,&ind_);
    Mat m_Pt_;
    MatTranspose(m_P,MAT_INITIAL_MATRIX,&m_Pt_);
    // Shouldn't be destroyed here, moved to very end of main()
//    MatDestroy(&m_P);                           //to make sure I don't use it by mistake

    PetscInt num_row_m_Pt_, num_col_m_Pt_;
    MatGetSize(m_Pt_,&num_row_m_Pt_,&num_col_m_Pt_);
#if dbl_RF_FSN >=5
    cout << "[RF][FSN]{" << cc_name << "} P transpose dim ["<< num_row_m_Pt_ <<","<< num_col_m_Pt_ << "]" <<endl;
    cout << "[RF][FSN]{" << cc_name << "} m_data num rows:"<< num_row_fine_points << endl;
    #if dbl_RF_FSN >=7            //should be above 7
        cout << "[RF][FSN]{" << cc_name << "} list of all SVs are:\n";
//        if(cc_name == "Majority"){
            cout << "[RF][FSN]{" << cc_name << "} [HINT]for no fake point, they should start from zero, not couple hundreds:\n";
//        }
        // num_seeds comes from number of SV from the solution from model selection
        for(unsigned int i=0; i < num_seeds ; i++){
            printf("%d, ",seeds_ind[i]);
        }
        printf("\n");
    #endif
#endif

    // a temporary vector for parts in a selected aggregate
    //maximum number of points(columns) in each row of P'
    std::vector<selected_agg > v_agg_;
    v_agg_.reserve(num_row_fine_points);

    /// - - - - - reserve as the number of rows in finer data set (for each class) - - - - -
    std::vector<int> v_fine_neigh_id(num_row_fine_points);

//    exit(1);
    /// - - - - - - - - - Select fine points - - - - - - - -
    // Loop over indices of SV's in coarser level in P' matrix (Oct 2, #bug, fixed)
    for(unsigned int i=0; i < num_seeds ; i++){
        MatGetRow(m_Pt_,seeds_ind[i],&ncols, &cols, &vals);

#if dbl_RF_FSN >=1
        if(ncols == 0){
            cout  << "[RF][FSN]{" << cc_name << "} empty row in P' at row i:"<< i
                       << " seeds_ind[i]:" << seeds_ind[i] << " ncols:" << ncols << endl;
            exit(1);
        }
        #if dbl_RF_FSN >=3
            cout  << "[RF][FSN]{" << cc_name << "} MatGetRow of P' matrix in loop seeds_ind[i]:"
                           << seeds_ind[i] << " ncols:" << ncols << endl;
        #endif
#endif
        // - - - - if there is only one node in this aggregate, select it - - - -
        if(ncols == 1){
            v_fine_neigh_id[cols[0]] = 1;
        }
        else {                  // multiple nodes participate this aggregate
            for(int j=0; j < ncols ; j++){  // for each row
                // - - - - - create a vector of pairs - - - - -
                // (fine index, participation in aggregate)
                v_agg_.push_back( selected_agg(cols[j], vals[j]) );
            }

            // - - - sort the vector of multiple participants in this aggregate - - -
            std::sort(v_agg_.begin(), v_agg_.end(), std::greater<selected_agg>());

#if dbl_RF_FSN >=7
    printf("==== [MR][inside selecting agg]{after sort each row of P'} i:%d ====\n",i);
    for (auto it = v_agg_.begin(); it != v_agg_.end(); it++){
        printf("index:%d, value:%g\n", it->index, it-> value);
    }
//index is the column number and important part
//value is only used to find the important indices (selected indices)
#endif

            // - - - select fraction of participants - - -
            float add_frac_ = ceil(paramsInst->get_rf_add_fraction() * ncols); // read add_fraction from parameters
            for (auto it = v_agg_.begin(); it != v_agg_.begin() + add_frac_ ; it++){
                v_fine_neigh_id[it->index] =1 ;
            }

            v_agg_.clear();
        } // end of else for multiple participants in this aggregate

        MatRestoreRow(m_Pt_,seeds_ind[i],&ncols, &cols, &vals);
    }
    MatDestroy(&m_Pt_);
#if dbl_RF_FSN >=9
    cout<<"[RF][find_SV_neighbors] num_seeds:"<<num_seeds<<endl;
#endif

//    cout << "exit [RF] 1241:" << endl; exit(1);
    // 4-12-2019 - Korey
    // add extra neighbors to try to improve solution
    if(aug_indices.size() > 0) {
        for(auto it=aug_indices.begin(); it != aug_indices.end(); it++)
            v_fine_neigh_id[*it] = 1;
    }

    /// - - - - - - - - - Add distant points - - - - - - - - -

    // - - - - - calc average edge weight - - - - -
    Vec     v_sum_edge_weight;
    VecCreateSeq(PETSC_COMM_SELF,num_row_fine_points,&v_sum_edge_weight);
    MatGetRowSum(m_WA,v_sum_edge_weight);
    PetscScalar         * a_sum_edge_weight;
    PetscMalloc1(num_row_fine_points, &a_sum_edge_weight);
    VecGetArray(v_sum_edge_weight, &a_sum_edge_weight);

    double rf_add_dist_threshold = 0;                           // should be a parameter
    for(unsigned int i=0; i < v_fine_neigh_id.size(); i++){
        if(v_fine_neigh_id[i] == 1 ){        // if this row is selected
            MatGetRow(m_WA,i, &ncols, &cols, &vals);            // Get the selected row
            double avg_edge_weight = a_sum_edge_weight[i] / ncols;
            for(int j=0; j<ncols; j++){
                /// - - - - If I want to add another parameter, I can use the vals[j] to select ratio of the closest points - - - - - //TODO
                if(  (vals[j] >= rf_add_dist_threshold * avg_edge_weight) && (v_fine_neigh_id[cols[j]]!=1)  ){
                        v_fine_neigh_id[cols[j]]=2;
                }
            }
            MatRestoreRow(m_WA,i, &ncols, &cols, &vals);
        }
    }

    /// - - - - - - - - - Get the output submatrix - - - - - - - - -
    // Add points which are selected to the array for the output submatrix
    int cnt_agg_part_distant_neighbor = 0;
    int cnt_total=0;
    for(unsigned int i=0; i < v_fine_neigh_id.size(); i++){
        if(paramsInst->get_rf_add_distant_point_status()){
            if(v_fine_neigh_id[i] > 0 ){        // it participate directly(1) or it is a distant neighbor(2)
                ind_[cnt_total] = i;
                cnt_total++;
                if(v_fine_neigh_id[i] == 2 ){
                    cnt_agg_part_distant_neighbor++;
                }
            }
        }else{                                  // distant neighbors are ignored
            if(v_fine_neigh_id[i] == 1 ){        // if this row is selected
                ind_[cnt_total] = i;
                cnt_total++;
            }
        }
    }   // the ind_ is sorted as it fills in sorted order (i is sorted in the above loop)


    // Using WA matrix, find the neighbors of points which are participated in SV's aggregate
#if dbl_RF_FSN >=1      // this should be 1
    if(paramsInst->get_rf_add_distant_point_status()){
        cout  << "[RF][FSN]{" << cc_name << "} num of points participated in SV aggregates are: "<< cnt_total - cnt_agg_part_distant_neighbor  << endl;
        cout  << "[RF][FSN]{" << cc_name << "} num of distant 1 neighbor of above points are::"<< cnt_agg_part_distant_neighbor << endl;
    }else{
        cout  << "[RF][FSN]{" << cc_name << "} num of points participated in SV aggregates are: "<< cnt_total <<
                      " and distant neighbors are ignored!" << endl;
    }

#endif




    //create the IS (Index Set)
    ISCreateGeneral(PETSC_COMM_SELF,cnt_total,ind_,PETSC_COPY_VALUES,&IS_neigh_id);
    PetscFree(ind_);      //free the indices as I have created the IS

#if dbl_RF_FSN >=7          //default is 7
    printf("[MR] IS is created \n");               //$$debug
    ISView(IS_neigh_id,PETSC_VIEWER_STDOUT_WORLD);
//        MatGetSize(m_data,&num_row_fine_points,NULL);
//        printf("[MR] m_data num rows: %d\n",num_row_fine_points);
#endif


    MatGetSubMatrix(m_data,IS_neigh_id, NULL,MAT_INITIAL_MATRIX,&m_neighbors);


#if dbl_RF_FSN >=3
    PetscInt m_neighbors_num_row =0, m_neighbors_num_col;
    MatGetSize(m_neighbors ,&m_neighbors_num_row,&m_neighbors_num_col);

    cout  << "[RF][FSN]{" << cc_name
                  << "} new sub matrix dimension #row:" << m_neighbors_num_row
                  << ",#col:" <<m_neighbors_num_col << endl;
#endif
#if dbl_RF_FSN >=7      //default is 7
    cout  << "[RF][FSN]{" << cc_name << "} m_neighbors matrix:\n";                       //$$debug
    MatView(m_neighbors,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif
//    ISDestroy(&IS_neigh_id);  // Don't destroy it. It is required later in the partitioning
}


void Refinement::process_coarsest_level(Mat& m_data_p, Vec& v_vol_p,
                                        Mat& m_data_n, Vec& v_vol_n,
                                        Mat& m_VD_p, Mat& m_VD_n, int level,
                                        solution& sol_coarsest,
                                        std::vector<ref_results>& v_ref_results){

#if dbl_RF_exp_data_ml == 0 /// - - -  normal case  - - -
    PetscInt check_num_row_VD;
    MatGetSize(m_VD_p, &check_num_row_VD, NULL );
    assert (check_num_row_VD && "[RF][PCL] minority validation data is empty");
    MatGetSize(m_VD_n, &check_num_row_VD, NULL );
    assert (check_num_row_VD && "[RF][PCL] majority validation data is empty");
    // - - - - for long runs - - - -
    bool l_inh_param=false;
    double local_param_c=1;
    double local_param_gamma=1;
    // use the best parameters from past trainings
    if(paramsInst->get_best_params_status()){
        l_inh_param = true;
        local_param_c = paramsInst->get_best_C();
        local_param_gamma = paramsInst->get_best_gamma();
    }
                  /// - - - - - load the validation data - - - - -
    if(paramsInst->get_ms_status()){      // - - - - model selection - - - -
        // call model selection method
        ref_results refinement_results;
        ModelSelection ms_coarsest;
        ms_coarsest.uniform_design_separate_validation(m_data_p, v_vol_p,
                                                       m_data_n, v_vol_n,
                                                       l_inh_param,
                                                       local_param_c,
                                                       local_param_gamma,
                                                       m_VD_p, m_VD_n,
                                                       level, sol_coarsest,
//                                                       v_ref_results);
                                                       refinement_results);
        v_ref_results.push_back(refinement_results);
    }else{        /// - - - - No model selection (call solver directly) - - - -

        Solver sv_coarsest;
        struct svm_model *coarsest_model;
        coarsest_model = sv_coarsest.train_model(m_data_p, v_vol_p,
                                                 m_data_n, v_vol_n,
                                                 l_inh_param,
                                                 local_param_c,
                                                 local_param_gamma);

        PetscInt num_row_p;
        MatGetSize(m_data_p, &num_row_p, NULL);
        prepare_single_solution(&coarsest_model, num_row_p, sol_coarsest);
        // Notice, the sv_coarsest is only availabel in this scope,
        //  and not accessible outside the else clause
        sv_coarsest.free_solver("[RF][PCL]");
        cout << "the process_coarsest_level without " <<
                "model selection is incomplete" << endl ;
        throw "[RF][PCL] NOT DEVELOPED!";

    }

#else   // dbl_RF_exp_data_ml /// - - -  debug/experimental cases  - - -
    k_fold kf;
    Mat m_train_data_label, m_validation_data_label;
    // false: don't destroy the input matrices inside the function
    // the data files are destroyed in the MainRecursion class
    kf.combine_two_classes_in_one(m_train_data_label, m_data_p, m_data_n,
                                  false);
    kf.combine_two_classes_in_one(m_validation_data_label, m_VD_p, m_VD_n,
                                  false);
    std::string out_prefix = paramsInst->get_exp_info() +
            "_exp:" + std::to_string(paramsInst->get_main_current_exp_id()) +
            "_kf:" + std::to_string(paramsInst->get_main_current_kf_id()) +
            "_level:"+std::to_string(paramsInst->get_main_current_level_id());

    std::string out_train_label_fname = out_prefix + "_traindata_label.dat";
    std::string out_validation_label_fname = out_prefix + "_validationdata_label.dat";
    std::string out_min_vol_fname = out_prefix + "_min_vol.dat";
    std::string out_maj_vol_fname = out_prefix + "_maj_vol.dat";

    CommonFuncs cf;
    cf.exp_matrix(m_train_data_label, "./debug/", out_train_label_fname,
                  "[RF][main]");
    cf.exp_matrix(m_validation_data_label, "./debug/",
                  out_validation_label_fname, "[RF][main]");
    cf.exp_vector(v_vol_p, "./debug/", out_min_vol_fname, "[RF][main]");
    cf.exp_vector(v_vol_n, "./debug/", out_maj_vol_fname, "[RF][main]");
    cout << "DEBUG refinement, no classification!" << endl;
    if (level < 4){
        cout << "[RF][PCL] There is a problem of double free for the 1st level"<<
		", exit manually" << endl;
        exit(1);
    }
	
#endif

}


void Refinement::prepare_single_solution(svm_model **svm_trained_model, int num_row_p, solution& result_solution){
    PetscInt i;
    result_solution.C = (*svm_trained_model)->param.C;
    result_solution.gamma = (*svm_trained_model)->param.gamma;
    cout << "[RF][PSS] params are set C:"<< result_solution.C << endl ;

    result_solution.p_index.reserve((*svm_trained_model)->nSV[0]);   //reserve the space for positive class
    for (i=0; i < (*svm_trained_model)->nSV[0];i++){
        // -1 because sv_indice start from 1, while petsc row start from 0
        result_solution.p_index.push_back((*svm_trained_model)->sv_indices[i] - 1);
    }
    cout << "[RF][PSS] P class is prepared\n";

    result_solution.n_index.reserve((*svm_trained_model)->nSV[1]);   //reserve the space for negative class
    // start from 0 to #SV in majority
    // add the index in the model for it after subtract from number of minority in training data
    for (i=0; i < (*svm_trained_model)->nSV[1];i++){
        // -1 the same as pos class, p_num_row because they are after each other
        result_solution.n_index.push_back((*svm_trained_model)->sv_indices[(*svm_trained_model)->nSV[0] + i] - 1 - num_row_p);
    }
    cout << "[RF][PSS] N class is prepared\n";
}


struct BetterGmean
{
    bool operator () (const ref_results& a, const ref_results& b) const
    {
        return (a.validation_data_summary.perf.at(Gmean) > b.validation_data_summary.perf.at(Gmean));         //a has larger gmean than b
    }
};


void Refinement::add_best_model(std::vector<ref_results>& v_ref_results) const{
    // - - - - - find the best model - - - - -
#if dbl_RF_ABM >=5
    printf("\n[RF][SBM] final model at each level of refinement (before sort)\n");
    for(auto it=v_ref_results.begin() ; it != v_ref_results.end(); ++it){
        paramsInst->print_summary(it->validation_data_summary, "[RF][SBM] (A v-cycle) VD", it-> level);
    }
    for(auto it=v_ref_results.begin() ; it != v_ref_results.end(); ++it){
        paramsInst->print_summary(it->test_data_summary, "[RF][SBM] (A v-cycle) TD", it-> level);
    }
#endif

    std::sort(v_ref_results.begin(),v_ref_results.end(),BetterGmean());   // select the model with best G-mean or any other preferred performance measure

#if dbl_RF_ABM >=3
    printf("\n[RF][SBM] final model at each level of refinement (after sort)\n");
    for(auto it=v_ref_results.begin() ; it != v_ref_results.end(); ++it){
        paramsInst->print_summary(it->validation_data_summary, "[RF][SBM] (A v-cycle) VD", it-> level);
    }
    for(auto it=v_ref_results.begin() ; it != v_ref_results.end(); ++it){
        paramsInst->print_summary(it->test_data_summary, "[RF][SBM] (A v-cycle) TD", it-> level);
    }
#endif
    // - - - - - add the best model to final results of experiment for this v-cycle - - - - -
    paramsInst->add_final_summary(v_ref_results[0].test_data_summary, v_ref_results[0].level);

}
