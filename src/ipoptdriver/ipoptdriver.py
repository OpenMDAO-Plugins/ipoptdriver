"""
    ipoptdriver.py - Driver for the IPOPT optimizer.

    See Appendix B for additional information on the :ref:`IPOPTDriver`.
"""

# pylint: disable=E0611,F0401,E1101,R0903,E1002,R0903,C0103
# E0611 - name cannot be found in a module
# F0401 - Unable to import module
# E1101 - Used when a variable is accessed for an unexistent member
# R0904 - Too many public methods
# E1002 - disable complaints about .__init__: Use super on an old style class
# R0903 - Disable complaints about Too few public methods
# C0103 - Disable complaints Invalid name "setUp"
#              (should match [a-z_][a-z0-9_]{2,30}$)

import functools
import sys

#public symbols
__all__ = ['IPOPTdriver']

from numpy import zeros, array, append

import pyipopt
pyipopt.set_loglevel(0)  # Avoid wrapper entry/return trace.

from openmdao.main.api import Driver
from openmdao.main.datatypes.api import Enum, Float, Int, Dict
from openmdao.main.exceptions import RunStopped
from openmdao.main.interfaces import IHasParameters, IHasConstraints, \
                                     IHasObjective, implements, IOptimizer
from openmdao.main.hasconstraints import HasConstraints
from openmdao.main.hasobjective import HasObjective
from openmdao.main.hasparameters import HasParameters

from openmdao.util.decorators import add_delegate


class IpoptReturnStatus(object):
    '''A fake enum for the possible values of
    the status variable returned by Ipopt
    '''

    Solve_Succeeded = 0
    Solved_To_Acceptable_Level = 1
    Infeasible_Problem_Detected = 2
    Search_Direction_Becomes_Too_Small = 3
    Diverging_Iterates = 4
    User_Requested_Stop = 5
    Feasible_Point_Found = 6

    Maximum_Iterations_Exceeded = -1
    Restoration_Failed = -2
    Error_In_Step_Computation = -3
    Maximum_CpuTime_Exceeded = -4
    Not_Enough_Degrees_Of_Freedom = -10
    Invalid_Problem_Definition = -11
    Invalid_Option = -12
    Invalid_Number_Detected = -13

    Unrecoverable_Exception = -100
    NonIpopt_Exception_Thrown = -101
    Insufficient_Memory = -102
    Internal_Error = -199

    Undefined = -9999 # not part of Ipopt. Added for use with wrapper


def eval_f(x, driver):
    '''evaluate objective function'''

    return driver.eval_objective()

def eval_g(x, driver):
    '''evaluate constraint functions'''

    return array(driver.eval_constraints(driver.parent))

def eval_grad_f(x, driver):
    '''gradient of the object function'''

    if driver._saved_j is None:
        driver._recalc_j()
    return driver._saved_j[0, :].ravel()

def eval_jac_g(x, flag, driver):
    '''
    Calculate the Jacobian of the gradients.
    In pyipopt, it sets the flag to indicate whether
    to return iRow and jCol or the actual values of the
    Jacobian
    '''

    if flag:
        # for 4 variables and 2 constraints, for example,
        #   return (array([0, 0, 0, 0, 1, 1, 1, 1]),
        #           array([0, 1, 2, 3, 0, 1, 2, 3]))

        irow = array( [[ ]] )
        for i in range( driver.num_constraints ):
            newrow = array( [ [i] * driver.num_params ] )
            if i:
                irow = append( irow, newrow, axis=0 )
            else:
                irow = newrow

        jcol = array( [[ ]] )
        prange = range( driver.num_params )
        for i in range( driver.num_constraints ):
            newrow = array( [ prange ] )
            if i:
                jcol = append( jcol, newrow, axis=0 )
            else:
                jcol = newrow

        return ( irow, jcol )

    else:
        # Need to return jac_g with dimensions of
        #    driver.num_constraints, driver.num_params

        if driver._saved_j is None:
            driver._recalc_j()
        return driver._saved_j[1:, :]

def eval_h(*args, **kwargs):
    """ Just a placeholder, should not be called. """

    raise NotImplementedError('eval_h')

def apply_new(x, driver):
    """ Apply new parameters and evaluate. """

    driver.set_parameters(x)
    super(IPOPTdriver, driver).run_iteration()
    driver._prev_parameters = x.copy()
    driver._saved_j = None

def intermediate_callback(alg_mod, iteration, obj_value, inf_pr, inf_du,
                          mu, d_norm, regularization_size, alpha_du, alpha_pr,
                          ls_trials, driver):
    ''' Ipopt calls back to this function each iteration'''

    # Incrementing the count here gets a true value of the iterations
    #   done by Ipopt
    driver.iter_count += 1
    driver.record_case()

    return not driver._stop


@add_delegate(HasParameters, HasConstraints, HasObjective)
class IPOPTdriver(Driver):
    """ Driver wrapper of C version of IPOPT.
    """

    implements(IHasParameters, IHasConstraints, IHasObjective, IOptimizer)

    # Control parameters for IPOPT. Specifically list the most common.
    # Leave the rest for the dictionary "options".
    print_level = Enum(5, range(13), iotype='in',
                       desc='Print '
                       'information during IPOPT solution. Higher values are '
                       'more verbose. Use 0 for no output')

    tol = Float(1.0e-8, iotype='in', low=0.0,
                desc='convergence tolerance. '
                'Algorithm terminates if the scaled NLP error becomes '
                'smaller than this value and if additional conditions '
                '(see Ipopt manual) are met')

    max_iter = Int(3000, iotype='in', low=0,
                   desc='Maximum number of iterations')

    max_cpu_time = Float(1.0e6, iotype='in', low=0.0,
                         desc='limit on CPU seconds' )

    constr_viol_tol = Float(0.0001, iotype='in', low=0.0,
                            desc='absolute tolerance on constraint violation' )

    obj_scaling_factor = Float(1.0, iotype='in',
                               desc='scaling factor for the objective function')

    # Previous default of 'ma27' isn't always available.
    linear_solver = Enum('mumps',
                         [ 'ma27', 'ma57', 'ma77',
                           'pardiso', 'wsmp', 'mumps', 'custom'],
                         iotype='in', desc='linear algebra package used' )

    status = Enum(IpoptReturnStatus.Undefined,
                  [
        IpoptReturnStatus.Solve_Succeeded,
        IpoptReturnStatus.Solved_To_Acceptable_Level,
        IpoptReturnStatus.Infeasible_Problem_Detected,
        IpoptReturnStatus.Search_Direction_Becomes_Too_Small,
        IpoptReturnStatus.Diverging_Iterates,
        IpoptReturnStatus.User_Requested_Stop,
        IpoptReturnStatus.Feasible_Point_Found,
        IpoptReturnStatus.Maximum_Iterations_Exceeded ,
        IpoptReturnStatus.Restoration_Failed ,
        IpoptReturnStatus.Error_In_Step_Computation ,
        IpoptReturnStatus.Maximum_CpuTime_Exceeded ,
        IpoptReturnStatus.Not_Enough_Degrees_Of_Freedom ,
        IpoptReturnStatus.Invalid_Problem_Definition ,
        IpoptReturnStatus.Invalid_Option ,
        IpoptReturnStatus.Invalid_Number_Detected ,
        IpoptReturnStatus.Unrecoverable_Exception,
        IpoptReturnStatus.NonIpopt_Exception_Thrown,
        IpoptReturnStatus.Insufficient_Memory,
        IpoptReturnStatus.Internal_Error,
        IpoptReturnStatus.Undefined,
        ],
                  iotype='out',
                  desc='Ipopt return code indicating status of optimization result' )

    # Available options can be listed by 'ipopt --print-options'.
    options = Dict({
        # this would just turn off copyright banner
        #    self.nlp.str_option( "sb", 'yes' )
        # to suppress all output set the following to 'yes'
        'suppress_all_output': 'no',

        # Output.
        # 'print_level' : 5,  # Made into variable.
        'output_file' : "",
        'file_print_level' : 5,
        'print_user_options' : "no",
        'print_options_documentation' : "no",
        'print_timing_statistics' : "no",
        'option_file_name' : "",  # Normal default is 'ipopt.opt'.
        'replace_bounds' : "no",
        'skip_finalize_solution_call' : "no",
        'print_info_string' : "no",
        'inf_pr_output' : "original",
        'print_frequency_iter' : 1,
        'print_frequency_time' : 0.,

        # Convergence.
        #'tol' : 1e-08,               # Made into variable.
        's_max' : 100.0,
        #'max_iter' : 3000,           # Made into variable.
        #'max_cpu_time' : 1e+06,      # Made into variable.
        'dual_inf_tol' : 1.0,
        #'constr_viol_tol' : 0.0001,  # Made into variable.
        'compl_inf_tol' : 0.0001,
        'acceptable_tol' : 1e-06,
        'acceptable_iter' : 15,
        'acceptable_dual_inf_tol' : 1e+10,
        'acceptable_constr_viol_tol' : 0.01,
        'acceptable_compl_inf_tol' : 0.01,
        'acceptable_obj_change_tol' : 1e+20,
        'diverging_iterates_tol' : 1e+20,
        'mu_target' : 0.0,

        # NLP Scaling.
        'nlp_scaling_method' : "gradient-based",
        #'obj_scaling_factor' : 1.,  # Made into variable.
        'nlp_scaling_max_gradient' : 100.0,
        'nlp_scaling_obj_target_gradient' : 0.0,
        'nlp_scaling_constr_target_gradient' : 0.0,
        'nlp_scaling_min_value' : 1e-08,

        # NLP.
        'nlp_lower_bound_inf' : -1e+19,
        'nlp_upper_bound_inf' : 1e+19,
        'fixed_variable_treatment' : "make_parameter",
        'dependency_detector' : "none",
        'dependency_detection_with_rhs' : "no",
        'num_linear_variables' : 0,
        'kappa_d' : 1e-05,
        'bound_relax_factor' : 1e-08,
        'honor_original_bounds' : "yes",
        'check_derivatives_for_naninf' : "no",
        'jac_c_constant' : "no",
        'jac_d_constant' : "no",
        'hessian_constant' : "no",

        # Initialization.
        'bound_push' : 0.01,
        'bound_frac' : 0.01,
        'slack_bound_push' : 0.01,
        'slack_bound_frac' : 0.01,
        'constr_mult_init_max' : 1000.0,
        'bound_mult_init_val' : 1.0,
        'bound_mult_init_method' : "constant",
        'least_square_init_primal' : "no",
        'least_square_init_duals' : "no",

        # Barrier parameter update.
        'mu_max_fact' : 1000.0,
        'mu_max' : 100000.0,
        'mu_min' : 1e-11,
        'adaptive_mu_globalization' : "obj-constr-filter",
        'adaptive_mu_kkterror_red_iters' : 4,
        'adaptive_mu_kkterror_red_fact' : 0.9999,
        'filter_margin_fact' : 1e-05,
        'filter_max_margin' : 1.0,
        'adaptive_mu_restore_previous_iterate' : "no",
        'adaptive_mu_monotone_init_factor' : 0.8,
        'adaptive_mu_kkt_norm_type' : "2-norm-squared",
        'mu_strategy' : "monotone",
        'mu_oracle' : "quality-function",
        'fixed_mu_oracle' : "average_compl",
        'mu_init' : 0.1,
        'barrier_tol_factor' : 10.0,
        'mu_linear_decrease_factor' : 0.2,
        'mu_superlinear_decrease_power' : 1.5,
        'mu_allow_fast_monotone_decrease' : "yes",
        'tau_min' : 0.99,
        'sigma_max' : 100.0,
        'sigma_min' : 1e-06,
        'quality_function_norm_type' : "2-norm-squared",
        'quality_function_centrality' : "none",
        'quality_function_balancing_term' : "none",
        'quality_function_max_section_steps' : 8,
        'quality_function_section_sigma_tol' : 0.01,
        'quality_function_section_qf_tol' : 0.0,

        # Line Search.
        'line_search_method' : "filter",
        'alpha_red_factor' : 0.5,
        'accept_every_trial_step' : "no",
        'accept_after_max_steps' : -1,
        'alpha_for_y' : "primal",
        'alpha_for_y_tol' : 10.0,
        'tiny_step_tol' : 2.22045e-15,
        'tiny_step_y_tol' : 0.01,
        'watchdog_shortened_iter_trigger' : 10,
        'watchdog_trial_iter_max' : 3,
        'theta_max_fact' : 10000.0,
        'theta_min_fact' : 0.0001,
        'eta_phi' : 1e-08,
        'delta' : 1.0,
        's_phi' : 2.3,
        's_theta' : 1.1,
        'gamma_phi' : 1e-08,
        'gamma_theta' : 1e-05,
        'alpha_min_frac' : 0.05,
        'max_soc' : 4,
        'kappa_soc' : 0.99,
        'obj_max_inc' : 5.0,
        'max_filter_resets' : 5,
        'filter_reset_trigger' : 5,
        'corrector_type' : "none",
        'skip_corr_if_neg_curv' : "yes",
        'skip_corr_in_monotone_mode' : "yes",
        'corrector_compl_avrg_red_fact' : 1.0,
        'nu_init' : 1e-06,
        'nu_inc' : 0.0001,
        'rho' : 0.1,
        'kappa_sigma' : 1e+10,
        'recalc_y' : "no",
        'recalc_y_feas_tol' : 1e-06,
        'slack_move' : 1.81899e-12,
        'constraint_violation_norm_type' : "1-norm",

        # Warm Start.
        'warm_start_init_point' : "no",
        'warm_start_same_structure' : "no",
        'warm_start_bound_push' : 0.001,
        'warm_start_bound_frac' : 0.001,
        'warm_start_slack_bound_push' : 0.001,
        'warm_start_slack_bound_frac' : 0.001,
        'warm_start_mult_bound_push' : 0.001,
        'warm_start_mult_init_max' : 1e+06,
        'warm_start_entire_iterate' : "no",

        # Linear Solver.
        #'linear_solver' : "mumps",  # Made into variable.
        'linear_system_scaling' : "none",  # Had been "mc19", but not always available.
        'linear_scaling_on_demand' : "yes",

        # Step Calculation.
        'mehrotra_algorithm' : "no",
        'fast_step_computation' : "no",
        'min_refinement_steps' : 1,
        'max_refinement_steps' : 10,
        'residual_ratio_max' : 1e-10,
        'residual_ratio_singular' : 1e-05,
        'residual_improvement_factor' : 1.0,
        'neg_curv_test_tol' : 1.0,
        'max_hessian_perturbation' : 1e+20,
        'min_hessian_perturbation' : 1e-20,
        'perturb_inc_fact_first' : 100.0,
        'perturb_inc_fact' : 8.0,
        'perturb_dec_fact' : 0.333333,
        'first_hessian_perturbation' : 0.0001,
        'jacobian_regularization_value' : 1e-08,
        'jacobian_regularization_exponent' : 0.25,
        'perturb_always_cd' : "no",

        # Restoration Phase.
        'expect_infeasible_problem' : "no",
        'expect_infeasible_problem_ctol' : 0.001,
        'expect_infeasible_problem_ytol' : 1e+08,
        'start_with_resto' : "no",
        'soft_resto_pderror_reduction_factor' : 0.9999,
        'max_soft_resto_iters' : 10,
        'required_infeasibility_reduction' : 0.9,
        'max_resto_iter' : 3000000,
        'evaluate_orig_obj_at_resto_trial' : "yes",
        'resto_penalty_parameter' : 1000.0,
        'resto_proximity_weight' : 1.0,
        'bound_mult_reset_threshold' : 1000.0,
        'constr_mult_reset_threshold' : 0.0,
        'resto_failure_feasibility_threshold' : 0.0,

        # Derivative Checker.
        'derivative_test' : "none",
        'derivative_test_first_index' : -2,
        'derivative_test_perturbation' : 1e-08,
        'derivative_test_tol' : 0.0001,
        'derivative_test_print_all' : "no",
        'jacobian_approximation' : "exact",
        'findiff_perturbation' : 1e-07,
        'point_perturbation_radius' : 10.0,

        # Hessian Approximation.
        'limited_memory_aug_solver' : "sherman-morrison",
        'limited_memory_max_history' : 6,
        'limited_memory_update_type' : "bfgs",
        'limited_memory_initialization' : "scalar1",
        'limited_memory_init_val' : 1.0,
        'limited_memory_init_val_max' : 1e+08,
        'limited_memory_init_val_min' : 1e-08,
        'limited_memory_max_skipping' : 2,
        'limited_memory_special_for_resto' : "no",
        #'hessian_approximation' : "exact",  # Forced to "limited-memory"
        'hessian_approximation_space' : "nonlinear-variables",

        # MA27 Linear Solver.
        'ma27_pivtol' : 1e-08,
        'ma27_pivtolmax' : 0.0001,
        'ma27_liw_init_factor' : 5.0,
        'ma27_la_init_factor' : 5.0,
        'ma27_meminc_factor' : 10.0,
        'ma27_skip_inertia_check' : "no",
        'ma27_ignore_singularity' : "no",

        # MA57 Linear Solver.
        'ma57_pivtol' : 1e-08,
        'ma57_pivtolmax' : 0.0001,
        'ma57_pre_alloc' : 1.05,
        'ma57_pivot_order' : 5,
        'ma57_automatic_scaling' : "yes",  # ipopt default is "no".
        'ma57_block_size' : 16,
        'ma57_node_amalgamation' : 16,
        'ma57_small_pivot_flag' : 0,

        # Paridiso Linear Solver.
        'pardiso_matching_strategy' : "complete+2x2",
        'pardiso_redo_symbolic_fact_only_if_inertia_wrong' : "no",
        'pardiso_repeated_perturbation_means_singular' : "no",
        'pardiso_out_of_core_power' : 0,
        'pardiso_msglvl' : 0,
        'pardiso_skip_inertia_check' : "no",
        'pardiso_max_iter' : 500,
        'pardiso_iter_relative_tol' : 1e-06,
        'pardiso_iter_coarse_size' : 5000,
        'pardiso_iter_max_levels' : 10000,
        'pardiso_iter_dropping_factor' : 0.5,
        'pardiso_iter_dropping_schur' : 0.1,
        'pardiso_iter_max_row_fill' : 10000000,
        'pardiso_iter_inverse_norm_factor' : 5e+06,
        'pardiso_iterative' : "no",
        'pardiso_max_droptol_corrections' : 4,

        # Mumps Linear Solver.
        'mumps_pivtol' : 1e-06,
        'mumps_pivtolmax' : 0.1,
        'mumps_mem_percent' : 1000,
        'mumps_permuting_scaling' : 7,
        'mumps_pivot_order' : 7,
        'mumps_scaling' : 77,
        'mumps_dep_tol' : -1.0,

        # MA28 Linear Solver.
        'ma28_pivtol' : 0.01,

        # Uncategorized.
        'warm_start_target_mu' : 0.0,
        }, iotype='in',
                   desc='Dictionary of additional optimization parameters' )


    def __init__(self):
        super(IPOPTdriver, self).__init__()

        self.iter_count = 0

        # define the IPOPTdriver's private variables
        # note, these are all resized in config_ipopt

        self.design_vals = zeros(0, 'd')
        self.nlp = None

        self.num_params = 0
        self.num_eq_constraints = 0
        self.num_constraints = 0

        self.obj = 0.0

        self._prev_parameters = None
        self._saved_j = None

    def set_option(self, name, value):
        '''Set one of the options in the large dict of options'''

        if name in self.options:
            self.options[ name ] = value
        else:
            self.raise_exception( '%s is not a valid option for Ipopt' % name,
                                  ValueError )

    def start_iteration(self):
        """Perform initial setup before iteration loop begins."""

        self._prev_parameters = None
        self._saved_j = None
        self._config_ipopt()

        # get the initial values of the parameters
        self.design_vals = self.eval_parameters(self.parent)

        x_L = self.get_lower_bounds()
        x_U = self.get_upper_bounds()
        # Ipopt treats equality and inequality constraints together.
        # For the equality constraints, both g_L and g_U are set to zero.
        # For the inequality constraints, g_L is set to -(largest float) and
        # g_U is set to zero.
        g_L = zeros( self.num_constraints, 'd' )
        g_U = zeros( self.num_constraints, 'd' )
        g_L[self.num_eq_constraints:] = -sys.float_info.max

        # number of non zeros in Jacobian
        nnzj = self.num_params * self.num_constraints
                           # of constraints. Assumed to be dense
        # number of non zeros in hessian
        nnzh = self.num_params * ( self.num_params + 1 ) / 2

        try:
            self.nlp = pyipopt.create(
               self.num_params, x_L, x_U,
               self.num_constraints, g_L, g_U,
               nnzj, nnzh,
               eval_f, eval_grad_f,
               eval_g, eval_jac_g,
               eval_h,  # Placeholder, should not be called.
               functools.partial(apply_new, driver=self)
               )

            self.nlp.set_intermediate_callback( intermediate_callback )

        except Exception, err:
            self._logger.error(str(err))
            raise

        # Set optimization options
        self.nlp.int_option( 'print_level', self.print_level )
        self.nlp.num_option( 'tol', self.tol )
        self.nlp.int_option( 'max_iter', self.max_iter )
        self.nlp.num_option( 'max_cpu_time', self.max_cpu_time )
        self.nlp.num_option( 'constr_viol_tol', self.constr_viol_tol )
        self.nlp.num_option( 'obj_scaling_factor', self.obj_scaling_factor )
        self.nlp.str_option( 'linear_solver', self.linear_solver )

        # Set optimization options set via the options dict
        for option, value in self.options.iteritems():
            if isinstance( value, int ):
                self.nlp.int_option( option, value )
            elif isinstance( value, str ):
                self.nlp.str_option( option, value )
            elif isinstance( value, float ):
                self.nlp.num_option( option, value )
            else:
                self.raise_exception("Cannot handle option '%s' of type '%s'"
                                     % (option, type(value)), ValueError)

        # Ipopt does the Hessian calculation so we do not have to
        self.nlp.str_option( "hessian_approximation", "limited-memory" )

    def continue_iteration(self):
        """Returns True if iteration should continue.
             Get info from the optimizer to see if it
             is done iterating
        """

        return self.iter_count == 0

    def pre_iteration(self):
        """Checks or RunStopped and evaluates objective"""

        super(IPOPTdriver, self).pre_iteration()
        if self._stop:
            self.raise_exception('Stop requested', RunStopped)

    def run_iteration(self):
        """ The IPOPT driver iteration"""

        try:
            ( self.design_vals,
              _zl, _zu, _lambda,  # lambda is a 'recent' addition
              self.obj,
              self.status ) = self.nlp.solve(self.design_vals, self)

        # so we can check for stops
        except Exception as err:
            self._logger.error(str(err))
            raise

        # update the model if necessary
        if self._prev_parameters is None or \
           not (self.design_vals == self._prev_parameters).all():
            self.set_parameters(self.design_vals)
            super(IPOPTdriver, self).run_iteration()

        # release storage
        self._prev_parameters = None
        self._saved_j = None

    def _config_ipopt(self):
        """Set up arrays, and perform some
        validation and make sure that array sizes are consistent.
        """

        # size arrays based on number of parameters
        self.num_params = self.total_parameters()
        self.design_vals = zeros(self.num_params, 'd')

        if self.num_params < 1:
            self.raise_exception('no parameters specified', RuntimeError)

        # size constraint related arrays
        self.num_eq_constraints = len( self.get_eq_constraints() )
        self.num_constraints = self.num_eq_constraints + \
                               len( self.get_ineq_constraints() )

    def _recalc_j(self):
        """Update _saved_j."""

        inputs = self.list_param_group_targets()
        obj    = self.list_objective_targets()
        con    = self.list_constraint_targets()

        self._saved_j = self.workflow.calc_gradient(inputs, obj + con)

